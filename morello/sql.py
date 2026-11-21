"""Reading a table where it lives. Only sampled rows leave the engine.

A Source wraps a connection and knows how to run a query into pandas, append a small DataFrame
to a table, and which SQL dialect it speaks. A TableSampler builds the queries for one table.

Table and column names are checked against a conservative pattern before they are put into a
query. The optional `where` filter is inserted as written, so it must come from whoever
configures the monitor and never from untrusted input.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

RANDOM_FUNCTION = {
    "snowflake": "UNIFORM(0::FLOAT, 1::FLOAT, RANDOM())",
    "bigquery": "RAND()",
    "spark": "rand()",
    "databricks": "rand()",
    "sqlite": "(ABS(RANDOM()) / 9223372036854775807.0)",
    "mysql": "RAND()",
}
STRING_TYPE = {
    "bigquery": "STRING",
    "spark": "STRING",
    "databricks": "STRING",
    "sqlite": "TEXT",
    "mysql": "CHAR",
}
_NAME_PART = r"([A-Za-z_][A-Za-z0-9_$]*|\"[^\"]+\"|`[^`]+`)"  # plain, "quoted" or `quoted`
IDENTIFIER = re.compile(rf"^{_NAME_PART}(\.{_NAME_PART})*$")
OVERSAMPLE = 1.3  # ask for this many times the rows wanted, then trim to the exact count
INSERT_ROWS_PER_STATEMENT = 50


def identifier(name: str) -> str:
    """Return a table or column name unchanged if it is safe to put into a query."""
    if not isinstance(name, str) or not IDENTIFIER.match(name):
        raise ValueError(
            f"{name!r} is not a table or column name: use letters, digits, _ and $, with dots "
            "between parts, or quote a part with double quotes or backticks"
        )
    return name


def literal(value: Any) -> str:
    """A value as a quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


class Source:
    dialect = "generic"

    def query(self, sql: str) -> pd.DataFrame:
        raise NotImplementedError

    def append(self, table: str, rows: pd.DataFrame) -> None:
        """Append rows to a table, creating it if needed."""
        raise NotImplementedError

    def table_exists(self, table: str) -> bool:
        """Whether the table can be read. Engines disagree on catalog queries, so this asks
        for zero rows and treats a failure as "no"."""
        try:
            self.query(f"SELECT 1 FROM {identifier(table)} LIMIT 0")
        except Exception:  # every driver raises its own error type for a missing table
            return False
        return True

    @property
    def random_function(self) -> str:
        return RANDOM_FUNCTION.get(self.dialect, "RANDOM()")

    @property
    def string_type(self) -> str:
        return STRING_TYPE.get(self.dialect, "VARCHAR")


class DuckDBSource(Source):
    dialect = "duckdb"
    STAGING_VIEW = "_morello_rows"

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def query(self, sql: str) -> pd.DataFrame:
        return self.connection.execute(sql).df()

    def append(self, table: str, rows: pd.DataFrame) -> None:
        table = identifier(table)
        self.connection.register(self.STAGING_VIEW, rows)
        try:
            self.connection.execute(
                f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {self.STAGING_VIEW} LIMIT 0"
            )
            self.connection.execute(f"INSERT INTO {table} SELECT * FROM {self.STAGING_VIEW}")
        finally:
            self.connection.unregister(self.STAGING_VIEW)


class SparkSource(Source):
    """A SparkSession: Databricks, or any Spark cluster."""

    dialect = "spark"

    def __init__(self, spark: Any) -> None:
        self.spark = spark

    def query(self, sql: str) -> pd.DataFrame:
        return self.spark.sql(sql).toPandas()

    def append(self, table: str, rows: pd.DataFrame) -> None:
        self.spark.createDataFrame(rows).write.mode("append").saveAsTable(identifier(table))

    def frame(self, sql: str) -> Any:
        """The query as a Spark DataFrame, left on the cluster."""
        return self.spark.sql(sql)


class SnowparkSource(Source):
    """A Snowpark session, for code running inside Snowflake."""

    dialect = "snowflake"

    def __init__(self, session: Any) -> None:
        self.session = session

    def query(self, sql: str) -> pd.DataFrame:
        return self.session.sql(sql).to_pandas()

    def append(self, table: str, rows: pd.DataFrame) -> None:
        rows = rows.rename(columns=str.upper)
        self.session.write_pandas(
            rows,
            identifier(table),
            auto_create_table=True,
            overwrite=False,
            quote_identifiers=False,
        )


class SqlAlchemySource(Source):
    def __init__(self, engine: Any, dialect: str | None = None) -> None:
        self.engine = engine
        name = engine.dialect.name
        self.dialect = dialect or {"postgresql": "postgres"}.get(name, name)

    def query(self, sql: str) -> pd.DataFrame:
        import sqlalchemy

        with self.engine.connect() as connection:
            return pd.read_sql(sqlalchemy.text(sql), connection)

    def append(self, table: str, rows: pd.DataFrame) -> None:
        schema, _, name = identifier(table).rpartition(".")
        rows.to_sql(name, self.engine, schema=schema or None, if_exists="append", index=False)


class DbapiSource(Source):
    """Any DB-API connection: snowflake-connector, databricks-sql-connector, psycopg and so on."""

    COLUMN_TYPE = {"i": "BIGINT", "u": "BIGINT", "f": "DOUBLE", "b": "BOOLEAN"}

    def __init__(self, connection: Any, dialect: str) -> None:
        self.connection, self.dialect = connection, dialect

    def query(self, sql: str) -> pd.DataFrame:
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            for fast_fetch in ("fetch_pandas_all", "fetchall_arrow"):
                if hasattr(cursor, fast_fetch):
                    result = getattr(cursor, fast_fetch)()
                    return result if isinstance(result, pd.DataFrame) else result.to_pandas()
            names = [description[0] for description in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=names)
        finally:
            cursor.close()

    def append(self, table: str, rows: pd.DataFrame) -> None:
        table = identifier(table)
        columns = [identifier(name) for name in rows.columns]
        definitions = ", ".join(
            f"{name} {self.COLUMN_TYPE.get(rows[name].dtype.kind, self.string_type)}"
            for name in columns
        )
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({definitions})")
            for start in range(0, len(rows), INSERT_ROWS_PER_STATEMENT):
                chunk = rows.iloc[start : start + INSERT_ROWS_PER_STATEMENT]
                values = ", ".join(
                    "(" + ", ".join(_sql_value(value) for value in row) + ")"
                    for row in chunk.itertuples(index=False)
                )
                cursor.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES {values}")
            if hasattr(self.connection, "commit"):
                self.connection.commit()
        finally:
            cursor.close()


def connect(target: Any, dialect: str | None = None) -> Source:
    """Wrap a connection, a session or a URL in a Source.

    Accepts a DuckDB connection, a SparkSession, a Snowpark session, a SQLAlchemy engine or URL,
    "duckdb:path/to/file.db", or any DB-API connection (give `dialect` for those).
    """
    if isinstance(target, Source):
        return target
    if isinstance(target, str):
        if target.startswith("duckdb:"):
            import duckdb

            path = target.split(":", 1)[1]
            path = path[2:] if path.startswith("//") else path  # duckdb:///abs or duckdb:rel
            return DuckDBSource(duckdb.connect(path or ":memory:"))
        import sqlalchemy

        return SqlAlchemySource(sqlalchemy.create_engine(target), dialect)

    module = type(target).__module__.lstrip("_")  # DuckDB's connection class lives in _duckdb
    if module.startswith("duckdb"):
        return DuckDBSource(target)
    if module.startswith("pyspark"):
        return SparkSource(target)
    if module.startswith("snowflake.snowpark"):
        return SnowparkSource(target)
    if module.startswith("sqlalchemy"):
        return SqlAlchemySource(target, dialect)
    if hasattr(target, "cursor"):
        if dialect is None:
            dialect = next((d for d in ("snowflake", "databricks") if d in module), "generic")
        return DbapiSource(target, dialect)
    raise TypeError(f"don't know how to query a {module}.{type(target).__name__}")


class TableSampler:
    """Builds the queries that count and sample the batches of one table.

    as_date=True   batches are calendar days of batch_col (a date, a timestamp or ISO text)
    as_date=False  batches are the distinct values of batch_col
    where          optional extra filter, for example "region = 'EU'". Inserted as written.
    """

    def __init__(
        self,
        source: Source,
        table: str,
        batch_col: str,
        as_date: bool = True,
        where: str | None = None,
    ) -> None:
        self.source = source
        self.table = identifier(table)
        self.batch_col = identifier(batch_col)
        self.as_date = as_date
        self.where = where

    @property
    def key_expression(self) -> str:
        as_text = f"CAST({self.batch_col} AS {self.source.string_type})"
        return f"SUBSTR({as_text}, 1, 10)" if self.as_date else as_text

    def _batch_filter(self, first: str, last: str | None = None) -> str:
        """Rows of batch `first`, or of batches first..last inclusive.

        Day ranges compare the raw column, so the engine can still prune partitions.
        """
        last = first if last is None else last
        if self.as_date:
            day_after = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            condition = (
                f"{self.batch_col} >= {literal(first)} AND {self.batch_col} < {literal(day_after)}"
            )
        elif first == last:
            condition = f"{self.key_expression} = {literal(first)}"
        else:
            condition = (
                f"{self.key_expression} >= {literal(first)} "
                f"AND {self.key_expression} <= {literal(last)}"
            )
        return f"({condition}) AND ({self.where})" if self.where else f"({condition})"

    def batch_counts(self, slice_col: str | None = None, since: str | None = None) -> pd.DataFrame:
        """Rows per batch, oldest first, in columns `batch` and `n_rows`.

        With slice_col, rows per batch and slice value, with the value in column `slice`.
        `since` limits the scan to batches at or after that key.
        """
        select = f"SELECT {self.key_expression} AS batch"
        group_by = "1"
        if slice_col:
            select += f", CAST({identifier(slice_col)} AS {self.source.string_type}) AS slice"
            group_by = "1, 2"
        conditions = [f"{self.batch_col} IS NOT NULL"]
        if self.where:
            conditions.append(f"({self.where})")
        if since is not None:
            compared = self.batch_col if self.as_date else self.key_expression
            conditions.append(f"{compared} >= {literal(since)}")
        counts = self.source.query(
            f"{select}, COUNT(*) AS n_rows FROM {self.table} "
            f"WHERE {' AND '.join(conditions)} GROUP BY {group_by} ORDER BY {group_by}"
        )
        counts.columns = [column.lower() for column in counts.columns]
        counts["batch"] = counts["batch"].astype(str)
        counts["n_rows"] = counts["n_rows"].astype(int)
        return counts

    def sample_query(self, first: str, last: str, total_rows: int, want_rows: int) -> str:
        """A query returning roughly OVERSAMPLE x want_rows random rows of batches first..last."""
        share = min(1.0, OVERSAMPLE * want_rows / max(total_rows, 1))
        randomly = "" if share >= 1.0 else f" AND {self.source.random_function} < {share:.8f}"
        limit = int(want_rows * OVERSAMPLE) + 10
        return (
            f"SELECT * FROM {self.table} WHERE {self._batch_filter(first, last)}{randomly} "
            f"LIMIT {limit}"
        )

    def sample(
        self, first: str, last: str, total_rows: int, want_rows: int, seed: int = 0
    ) -> pd.DataFrame:
        """At most want_rows random rows of batches first..last."""
        rows = self.source.query(self.sample_query(first, last, total_rows, want_rows))
        return _trim(rows, want_rows, seed)

    def sample_slices(
        self,
        batch: str,
        slice_col: str,
        rows_per_slice: dict[str, int],
        want_rows: int,
        seed: int = 0,
    ) -> dict[str, pd.DataFrame]:
        """One query returning up to want_rows rows of the batch for each slice value given.

        rows_per_slice maps each slice value to its row count in the batch, which sets how
        large a share of that slice to draw.
        """
        if not rows_per_slice:
            return {}
        as_text = f"CAST({identifier(slice_col)} AS {self.source.string_type})"
        share_by_slice = " ".join(
            f"WHEN {literal(value)} THEN {min(1.0, OVERSAMPLE * want_rows / max(n_rows, 1)):.8f}"
            for value, n_rows in rows_per_slice.items()
        )
        values = ", ".join(literal(value) for value in rows_per_slice)
        rows = self.source.query(
            f"SELECT * FROM {self.table} WHERE {self._batch_filter(batch)} "
            f"AND {as_text} IN ({values}) "
            f"AND {self.source.random_function} < CASE {as_text} {share_by_slice} ELSE 0 END"
        )
        column = next(name for name in rows.columns if name.lower() == slice_col.lower())
        return {
            value: _trim(rows[rows[column].astype(str) == str(value)], want_rows, seed)
            for value in rows_per_slice
        }


def _trim(rows: pd.DataFrame, want_rows: int, seed: int) -> pd.DataFrame:
    return rows.sample(want_rows, random_state=seed) if len(rows) > want_rows else rows


def _sql_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    is_number = isinstance(value, (int, float, np.integer, np.floating))
    if is_number and not isinstance(value, bool):
        return str(value)
    return literal(value)
