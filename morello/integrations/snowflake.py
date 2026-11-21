"""Snowflake, running inside the account as Snowpark Python stored procedures.

NOT RUN AGAINST A SNOWFLAKE ACCOUNT YET. The sampling SQL uses Snowflake's syntax and the same
code path is tested on DuckDB and Spark. Treat this module as a starting point.

Setup, once:
    1. zip the `morello` folder and upload it:
           PUT file://morello.zip @my_stage AUTO_COMPRESS=FALSE;
    2. run the statements printed by
           print(morello.integrations.snowflake.procedure_sql("@my_stage"))
    3. CALL MORELLO_FIT('DB.SCHEMA.ORDERS', 'ORDER_TS', 'DB.OPS.MORELLO_STATE');
    4. schedule the check:
           CREATE TASK ... SCHEDULE = 'USING CRON 0 6 * * * UTC' AS
           CALL MORELLO_CHECK('DB.SCHEMA.ORDERS', 'DB.OPS.MORELLO_STATE', 'DB.OPS.MORELLO_RESULTS');
"""

from __future__ import annotations

from typing import Any

from ..sql import SnowparkSource
from ..store import TableStore
from ..table import check_table, fit_table

PACKAGES = ("snowflake-snowpark-python", "pandas", "numpy", "lightgbm")


def fit_procedure(session: Any, table: str, batch_col: str, state_table: str) -> str:
    """Handler of MORELLO_FIT."""
    source = SnowparkSource(session)
    monitor = fit_table(source, table, batch_col, TableStore(source, state_table))
    return f"trained on {len(monitor.columns)} columns, {len(monitor.history)} calibration batches"


def check_procedure(session: Any, table: str, state_table: str, results_table: str) -> str:
    """Handler of MORELLO_CHECK."""
    source = SnowparkSource(session)
    reports = check_table(
        source, TableStore(source, state_table), table=table, results_table=results_table
    )
    alerting = [report for report in reports if report.alert]
    summary = f"{len(reports)} checks, {len(alerting)} alerts"
    return "\n".join([summary] + [report.text() for report in alerting])


def procedure_sql(stage: str, python_version: str = "3.11") -> str:
    """The CREATE PROCEDURE statements, for a stage that holds morello.zip."""
    packages = ", ".join(f"'{package}'" for package in PACKAGES)
    runtime = (
        f"LANGUAGE PYTHON RUNTIME_VERSION = '{python_version}'\n"
        f"PACKAGES = ({packages})\n"
        f"IMPORTS = ('{stage}/morello.zip')\n"
    )
    module = "morello.integrations.snowflake"
    fit = (
        "CREATE OR REPLACE PROCEDURE MORELLO_FIT("
        "TABLE_NAME STRING, BATCH_COL STRING, STATE_TABLE STRING)\n"
        f"RETURNS STRING\n{runtime}HANDLER = '{module}.fit_procedure';\n"
    )
    check = (
        "CREATE OR REPLACE PROCEDURE MORELLO_CHECK("
        "TABLE_NAME STRING, STATE_TABLE STRING, RESULTS_TABLE STRING)\n"
        f"RETURNS STRING\n{runtime}HANDLER = '{module}.check_procedure';\n"
    )
    return fit + "\n" + check
