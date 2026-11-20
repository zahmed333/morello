"""Sampling inside the warehouse. Only the sampled rows leave the engine.

A Source wraps a connection and knows three things: how to run a query into pandas, how to
append a small DataFrame to a table, and which SQL dialect it speaks."""
import numpy as np
import pandas as pd

RAND = {"snowflake": "UNIFORM(0::FLOAT, 1::FLOAT, RANDOM())", "bigquery": "RAND()", "spark": "rand()",
        "databricks": "rand()", "sqlite": "(ABS(RANDOM()) / 9223372036854775807.0)", "mysql": "RAND()"}
STR = {"bigquery": "STRING", "spark": "STRING", "databricks": "STRING", "sqlite": "TEXT", "mysql": "CHAR"}


class Source:
    dialect = "generic"

    def query(self, sql):
        raise NotImplementedError

    def append(self, table, df):
        raise NotImplementedError

    @property
    def rand(self):
        return RAND.get(self.dialect, "RANDOM()")

    @property
    def str_type(self):
        return STR.get(self.dialect, "VARCHAR")


class DuckDBSource(Source):
    dialect = "duckdb"

    def __init__(self, con):
        self.con = con

    def query(self, sql):
        return self.con.execute(sql).df()

    def append(self, table, df):
        self.con.register("_morello_df", df)
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _morello_df LIMIT 0")
        self.con.execute(f"INSERT INTO {table} SELECT * FROM _morello_df")
        self.con.unregister("_morello_df")


class SparkSource(Source):
    """A SparkSession: Databricks, or any Spark cluster."""
    dialect = "spark"

    def __init__(self, spark):
        self.spark = spark

    def query(self, sql):
        return self.spark.sql(sql).toPandas()

    def append(self, table, df):
        self.spark.createDataFrame(df).write.mode("append").saveAsTable(table)

    def frame(self, sql):
        return self.spark.sql(sql)


class SnowparkSource(Source):
    """A Snowpark session, for code running inside Snowflake (stored procedures, dbt Python models)."""
    dialect = "snowflake"

    def __init__(self, session):
        self.session = session

    def query(self, sql):
        return self.session.sql(sql).to_pandas()

    def append(self, table, df):
        df = df.copy(); df.columns = [c.upper() for c in df.columns]
        self.session.write_pandas(df, table, auto_create_table=True, overwrite=False, quote_identifiers=False)


class SqlAlchemySource(Source):
    def __init__(self, engine, dialect=None):
        self.engine = engine
        self.dialect = dialect or {"postgresql": "postgres"}.get(engine.dialect.name, engine.dialect.name)

    def query(self, sql):
        import sqlalchemy as sa
        with self.engine.connect() as c:
            return pd.read_sql(sa.text(sql), c)

    def append(self, table, df):
        schema, _, name = table.rpartition(".")
        df.to_sql(name, self.engine, schema=schema or None, if_exists="append", index=False)


class DbapiSource(Source):
    """Any DB-API connection (snowflake-connector, databricks-sql-connector, psycopg ...)."""
    def __init__(self, conn, dialect):
        self.conn, self.dialect = conn, dialect

    def query(self, sql):
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            for fast in ("fetch_pandas_all", "fetchall_arrow"):
                if hasattr(cur, fast):
                    out = getattr(cur, fast)()
                    return out if isinstance(out, pd.DataFrame) else out.to_pandas()
            return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        finally:
            cur.close()

    def append(self, table, df):
        types = {"i": "BIGINT", "u": "BIGINT", "f": "DOUBLE", "b": "BOOLEAN"}
        cols = ", ".join(f"{c} {types.get(df[c].dtype.kind, self.str_type)}" for c in df.columns)
        lit = lambda v: "NULL" if v is None or (isinstance(v, float) and np.isnan(v)) else (
            str(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            else "'" + str(v).replace("'", "''") + "'")
        cur = self.conn.cursor()
        try:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
            for i in range(0, len(df), 50):
                rows = ", ".join("(" + ", ".join(lit(v) for v in r) + ")" for r in df.iloc[i:i + 50].itertuples(index=False))
                cur.execute(f"INSERT INTO {table} ({', '.join(df.columns)}) VALUES {rows}")
            if hasattr(self.conn, "commit"):
                self.conn.commit()
        finally:
            cur.close()


def connect(obj, dialect=None):
    """Wrap a connection, session or URL in a Source."""
    if isinstance(obj, Source):
        return obj
    if isinstance(obj, str):
        if obj.startswith("duckdb:"):
            import duckdb
            path = obj.split(":", 1)[1]
            path = path[2:] if path.startswith("//") else path          # duckdb:///abs/path or duckdb:rel/path
            return DuckDBSource(duckdb.connect(path or ":memory:"))
        import sqlalchemy as sa
        return SqlAlchemySource(sa.create_engine(obj), dialect)
    mod, name = type(obj).__module__, type(obj).__name__
    if mod.startswith("duckdb"):
        return DuckDBSource(obj)
    if mod.startswith("pyspark"):
        return SparkSource(obj)
    if mod.startswith("snowflake.snowpark"):
        return SnowparkSource(obj)
    if mod.startswith("sqlalchemy"):
        return SqlAlchemySource(obj, dialect)
    if hasattr(obj, "cursor"):
        if dialect is None:
            dialect = "snowflake" if "snowflake" in mod else "databricks" if "databricks" in mod else "generic"
        return DbapiSource(obj, dialect)
    raise TypeError(f"don't know how to query a {mod}.{name}")


# ---------------------------------------------------------------------------- query building
def _lit(v):
    return "'" + str(v).replace("'", "''") + "'"


class TableSampler:
    """Builds the sample queries for one table.

    as_date=True  batches are calendar days of batch_col (a date, a timestamp, or ISO text)
    as_date=False batches are the distinct values of batch_col
    where         optional extra filter, e.g. "region = 'EU'"
    """

    def __init__(self, source, table, batch_col, as_date=True, where=None):
        self.src, self.table, self.batch_col, self.as_date, self.where = source, table, batch_col, as_date, where

    @property
    def key_expr(self):
        s = f"CAST({self.batch_col} AS {self.src.str_type})"
        return f"SUBSTR({s}, 1, 10)" if self.as_date else s

    def _pred(self, lo, hi=None):
        """Rows of batch lo, or of batches lo..hi inclusive. Day ranges compare the raw column so
        partition pruning still works."""
        hi = lo if hi is None else hi
        if self.as_date:
            nxt = (pd.Timestamp(hi) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            p = f"{self.batch_col} >= {_lit(lo)} AND {self.batch_col} < {_lit(nxt)}"
        elif lo == hi:
            p = f"{self.key_expr} = {_lit(lo)}"
        else:
            p = f"{self.key_expr} >= {_lit(lo)} AND {self.key_expr} <= {_lit(hi)}"
        return f"({p}) AND ({self.where})" if self.where else f"({p})"

    def batches(self, slice_col=None, since=None):
        """Row count per batch (and per slice value when slice_col is given), oldest first.
        since limits the scan to batches at or after that key."""
        sl = f", CAST({slice_col} AS {self.src.str_type}) AS s" if slice_col else ""
        grp = "1, 2" if slice_col else "1"
        w = f"WHERE {self.batch_col} IS NOT NULL" + (f" AND ({self.where})" if self.where else "")
        if since is not None:
            w += f" AND {self.batch_col if self.as_date else self.key_expr} >= {_lit(since)}"
        df = self.src.query(f"SELECT {self.key_expr} AS b{sl}, COUNT(*) AS n FROM {self.table} {w} GROUP BY {grp} ORDER BY {grp}")
        df.columns = [c.lower() for c in df.columns]
        df["b"] = df["b"].astype(str); df["n"] = df["n"].astype(int)
        return df

    def sample_sql(self, lo, hi, total_rows, want_rows):
        p = min(1.0, 1.3 * want_rows / max(total_rows, 1))
        cond = "" if p >= 1.0 else f" AND {self.src.rand} < {p:.8f}"
        return f"SELECT * FROM {self.table} WHERE {self._pred(lo, hi)}{cond} LIMIT {int(want_rows * 1.3) + 10}"

    def sample(self, lo, hi, total_rows, want_rows, seed=0):
        df = self.src.query(self.sample_sql(lo, hi, total_rows, want_rows))
        return df.sample(want_rows, random_state=seed) if len(df) > want_rows else df

    def sample_slices(self, key, slice_col, counts, want_rows, seed=0):
        """One query returning up to want_rows rows for each slice value in counts {value: rows}."""
        if not counts:
            return {}
        s = f"CAST({slice_col} AS {self.src.str_type})"
        whens = " ".join(f"WHEN {_lit(v)} THEN {min(1.0, 1.3 * want_rows / max(n, 1)):.8f}" for v, n in counts.items())
        df = self.src.query(f"SELECT * FROM {self.table} WHERE {self._pred(key)} AND {s} IN ({', '.join(_lit(v) for v in counts)}) "
                            f"AND {self.src.rand} < CASE {s} {whens} ELSE 0 END")
        col = next(c for c in df.columns if c.lower() == slice_col.lower())
        out = {}
        for v in counts:
            g = df[df[col].astype(str) == str(v)]
            out[v] = g.sample(want_rows, random_state=seed) if len(g) > want_rows else g
        return out
