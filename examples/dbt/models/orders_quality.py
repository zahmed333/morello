from morello.integrations.dbt import run


def model(dbt, session):
    dbt.config(materialized="table")
    return run(
        session,
        dbt.ref("orders"),
        name="orders",
        batch_col="order_ts",
        state_table=f"{dbt.this.schema}.morello_state",
    )
