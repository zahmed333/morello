# Morello in a dbt project

`models/orders_quality.py` is a dbt Python model that checks the upstream model `orders`.
The first run trains the monitor; later runs check the batches that arrived since the last run.
The `morello_ok` test fails the build when a batch alerts.

Local demo with dbt-duckdb (this is how the adapter was tested):

    pip install dbt-core dbt-duckdb
    pip install ../..                     # Morello itself, from this repo
    python make_demo_data.py clean        # 62 days of orders
    dbt build --profiles-dir .            # trains, test passes
    python make_demo_data.py broken       # adds a day where US orders carry EUR
    dbt build --profiles-dir .            # test fails and lists the suspect column

On Snowflake the `morello` package must be available to Snowpark (upload a zip to a stage and
add it under `imports` in `dbt.config`, with `packages=["lightgbm", "pandas", "numpy"]`).
On Databricks install it on the cluster. Neither has been run yet.
