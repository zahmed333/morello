"""Snowflake, running inside the account as a Snowpark Python stored procedure.

NOT RUN AGAINST A SNOWFLAKE ACCOUNT YET. The sampling SQL uses Snowflake's syntax and the same
code path is tested on DuckDB and Spark, but treat this module as a starting point.

Setup (once):
    1. zip the `morello` folder and upload it:   PUT file://morello.zip @my_stage AUTO_COMPRESS=FALSE;
    2. run the statement printed by  print(morello.integrations.snowflake.procedure_sql("@my_stage"))
    3. CALL MORELLO_FIT('DB.SCHEMA.ORDERS', 'ORDER_TS', 'DB.OPS.MORELLO_STATE');
    4. schedule:  CREATE TASK ... SCHEDULE = 'USING CRON 0 6 * * * UTC' AS
                  CALL MORELLO_CHECK('DB.SCHEMA.ORDERS', 'DB.OPS.MORELLO_STATE', 'DB.OPS.MORELLO_RESULTS');
"""
from ..sql import SnowparkSource
from ..store import TableStore
from ..table import fit_table, check_table


def fit_proc(session, table: str, batch_col: str, state_table: str) -> str:
    src = SnowparkSource(session)
    mon = fit_table(src, table, batch_col, TableStore(src, state_table))
    return f"trained on {len(mon.cols)} columns, {len(mon.history)} calibration batches"


def check_proc(session, table: str, state_table: str, results_table: str) -> str:
    src = SnowparkSource(session)
    reports = check_table(src, TableStore(src, state_table), table=table, results_table=results_table)
    alerts = [r for r in reports if r.alert]
    return f"{len(reports)} checks, {len(alerts)} alerts" + "".join("\n" + r.text() for r in alerts)


def procedure_sql(stage, python_version="3.11"):
    common = (f"LANGUAGE PYTHON RUNTIME_VERSION = '{python_version}'\n"
              "PACKAGES = ('snowflake-snowpark-python', 'pandas', 'numpy', 'lightgbm')\n"
              f"IMPORTS = ('{stage}/morello.zip')\n")
    return (
        "CREATE OR REPLACE PROCEDURE MORELLO_FIT(TABLE_NAME STRING, BATCH_COL STRING, STATE_TABLE STRING)\n"
        "RETURNS STRING\n" + common + "HANDLER = 'morello.integrations.snowflake.fit_proc';\n\n"
        "CREATE OR REPLACE PROCEDURE MORELLO_CHECK(TABLE_NAME STRING, STATE_TABLE STRING, RESULTS_TABLE STRING)\n"
        "RETURNS STRING\n" + common + "HANDLER = 'morello.integrations.snowflake.check_proc';\n")
