# Scala gate

`MorelloGate.scala` stops a Scala Spark job while a table has an unresolved Morello alert. It
reads the results table that the Python check writes.

Compile it against the Spark you run on. With a pip-installed PySpark no other tools are needed,
because its jars include the Scala compiler:

    JARS=$(python -c "import pyspark, os; print(os.path.dirname(pyspark.__file__) + '/jars')")
    mkdir -p classes
    java -cp "$JARS/*" scala.tools.nsc.Main -usejavacp -d classes MorelloGate.scala
    (cd classes && zip -q ../morello-gate.jar *.class)

Run it as a step after the Python check. It exits with 0 when no batch is alerting and with a
non-zero code and the list of batches when one is:

    spark-submit --class MorelloGate morello-gate.jar ops.morello_results analytics.orders

Or call it from your own job:

    MorelloGate.requireClean(spark, "ops.morello_results", "analytics.orders")

This was run with Scala 2.13.18 and Spark 4.3.0 in local mode, with the Python check and the
gate sharing a Hive metastore. It has not been run on Databricks.
