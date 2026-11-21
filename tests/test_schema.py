"""How columns of each kind are read."""

import numpy as np
import pandas as pd

from morello.schema import TableSchema, as_datetime


def table(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    created = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "price": rng.normal(10, 2, n).round(2),
            "quantity_text": rng.integers(1, 5, n).astype(str),
            "paid": rng.random(n) < 0.5,
            "country": rng.choice(["US", "DE", None], n),
            "created": created,
            "shipped": (created + pd.Timedelta(hours=5)).astype(str),
            "comment": [f"free text {i}" for i in range(n)],
            "unused": None,
        }
    )


def test_columns_are_planned_by_kind():
    schema = TableSchema(max_categories=50).fit(table())
    kinds = {plan.column: plan.kind for plan in schema.plans}
    assert kinds == {
        "price": "num",
        "quantity_text": "numstr",
        "paid": "bool",
        "country": "cat",
        "created": "datetime",
        "shipped": "datetime",
    }
    assert schema.skipped == {
        "row_id": "identifier",
        "comment": "identifier or free text",
        "unused": "empty",
    }
    assert "shipped__minus__created" in schema.features
    assert schema.source_column["shipped__minus__created"] == "shipped"


def test_transform_gives_numbers_and_keeps_nulls():
    rows = table()
    schema = TableSchema().fit(rows)
    encoded = schema.transform(rows)
    assert list(encoded.columns) == schema.features
    assert all(dtype == "float64" for dtype in encoded.dtypes)
    assert encoded["country"].isna().sum() == rows["country"].isna().sum()
    assert (encoded["shipped__minus__created"] == 5 * 3600).all()


def test_a_missing_column_becomes_nulls_and_new_values_get_the_other_code():
    rows = table()
    schema = TableSchema().fit(rows)
    changed = rows.drop(columns="price").assign(country="FR")
    encoded = schema.transform(changed)
    assert encoded["price"].isna().all()
    assert (encoded["country"] == -1.0).all()


def test_the_schema_survives_saving():
    rows = table()
    schema = TableSchema().fit(rows)
    restored = TableSchema.from_state(schema.to_state())
    pd.testing.assert_frame_equal(restored.transform(rows), schema.transform(rows))


def test_only_iso_text_counts_as_datetime():
    assert as_datetime(pd.Series(["2024-01-01", "2024-01-02 10:00:00"])) is not None
    assert as_datetime(pd.Series(["2024-01-01-17", "A-2024-01"])) is None
    assert as_datetime(pd.Series([20240101, 20240102])) is None
