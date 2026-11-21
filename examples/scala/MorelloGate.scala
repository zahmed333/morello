// Stop a Scala Spark job while a table has an unresolved Morello alert.
//
// Morello itself runs as a Python step (a Databricks workflow task, spark-submit of a .py file,
// or a notebook cell) that appends one row per check to a results table:
//
//     from morello.integrations import spark as morello_spark
//     morello_spark.check(spark, "catalog.sales.orders", state_table="ops.morello_state",
//                         results_table="ops.morello_results")
//
// A Scala job that runs afterwards calls MorelloGate.requireClean, or runs this file as a step
// of its own:
//
//     spark-submit --class MorelloGate morello-gate.jar ops.morello_results catalog.sales.orders
//
// The results table is an append-only log, so the gate reads the latest row of every batch
// and segment. A batch stops blocking when it is checked again and passes, or when a person
// accepts it (`morello accept --results-table ...` writes a row with status "accepted").
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number}

object MorelloGate {

  /** Throws IllegalStateException naming every batch of `tableName` whose latest check alerted. */
  def requireClean(spark: SparkSession, resultsTable: String, tableName: String): Unit = {
    val newestFirst = Window.partitionBy("batch", "segment").orderBy(col("checked_at").desc)
    val alerts = spark
      .table(resultsTable)
      .filter(col("table_name") === tableName)
      .withColumn("recency", row_number().over(newestFirst))
      .filter(col("recency") === 1 && col("status") === "alert")
      .select("batch", "segment", "suspect_columns", "top_check")
      .orderBy("batch", "segment")
      .collect()

    if (alerts.nonEmpty) {
      val lines = alerts.map { row =>
        val segment = if (row.getString(1).isEmpty) "" else s" [${row.getString(1)}]"
        s"batch ${row.getString(0)}$segment: ${row.getString(2)} (${row.getString(3)})"
      }
      throw new IllegalStateException(s"morello alert on $tableName:\n${lines.mkString("\n")}")
    }
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 2, "usage: MorelloGate <results table> <table name>")
    val spark = SparkSession.builder().appName("morello-gate").getOrCreate()
    try requireClean(spark, args(0), args(1))
    finally spark.stop()
  }
}
