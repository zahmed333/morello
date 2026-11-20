"""Dagster asset check. NOT RUN INSIDE A DAGSTER DEPLOYMENT YET; the function it calls is tested.

    @asset_check(asset=orders)
    def orders_quality():
        return morello_check_result(my_connection(), table="analytics.orders", store="s3://bucket/morello")
"""
from ..table import check_table


def morello_check_result(source, table, store, key=None, results_table=None, dialect=None):
    from dagster import AssetCheckResult
    reports = check_table(source, store, key=key, table=table, results_table=results_table, dialect=dialect)
    bad = [r for r in reports if r.alert]
    return AssetCheckResult(
        passed=not bad,
        metadata={"batches_checked": len(reports), "alerts": len(bad),
                  "suspect_columns": ", ".join(sorted({c for r in bad for c in r.suspects[:5]})),
                  "report": "\n".join(r.text() for r in bad)[:5000]})
