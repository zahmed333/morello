"""Where a monitor's state lives: a local folder, cloud storage, or a table in the warehouse."""
import base64, os, time
import pandas as pd

CHUNK = 700_000        # characters of base64 per row in a table store


class LocalStore:
    def __init__(self, root):
        self.root = str(root)

    def put(self, name, data):
        os.makedirs(self.root, exist_ok=True)
        tmp = os.path.join(self.root, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.join(self.root, name + ".tbz"))

    def get(self, name):
        p = os.path.join(self.root, name + ".tbz")
        return open(p, "rb").read() if os.path.exists(p) else None


class FsspecStore:
    """Any fsspec URL: s3://bucket/path, abfss://..., gs://..., dbfs:/..."""
    def __init__(self, root):
        import fsspec
        self.fs, self.root = fsspec.core.url_to_fs(str(root))

    def put(self, name, data):
        self.fs.makedirs(self.root, exist_ok=True)
        with self.fs.open(f"{self.root}/{name}.tbz", "wb") as f:
            f.write(data)

    def get(self, name):
        p = f"{self.root}/{name}.tbz"
        if not self.fs.exists(p):
            return None
        with self.fs.open(p, "rb") as f:
            return f.read()


class TableStore:
    """State kept in a warehouse table, so scheduled jobs anywhere can share it. Append-only:
    every put writes a new version and get reads the newest, which works on engines without DELETE."""
    def __init__(self, source, table):
        self.src, self.table = source, table

    def put(self, name, data):
        b64 = base64.b64encode(data).decode("ascii")
        version = int(time.time() * 1000)
        parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
        self.src.append(self.table, pd.DataFrame({
            "name": name, "version": version, "part": range(len(parts)), "n_parts": len(parts), "chunk": parts}))

    def get(self, name):
        q = name.replace("'", "''")
        try:
            df = self.src.query(
                f"SELECT part, n_parts, chunk FROM {self.table} WHERE name = '{q}' AND version = "
                f"(SELECT MAX(version) FROM {self.table} WHERE name = '{q}') ORDER BY part")
        except Exception:
            return None                       # table does not exist yet
        df.columns = [c.lower() for c in df.columns]
        if df.empty or len(df) != int(df["n_parts"].iloc[0]):
            return None
        return base64.b64decode("".join(df.sort_values("part")["chunk"]))


def open_store(target):
    if hasattr(target, "put") and hasattr(target, "get"):
        return target
    t = str(target)
    return FsspecStore(t) if "://" in t or t.startswith("dbfs:/") else LocalStore(t)
