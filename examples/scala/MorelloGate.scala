// Gate a Scala Spark job on morello results.
//
// morello itself runs as a Python step (a Databricks workflow task, spark-submit of a .py file, or a
// notebook cell) that writes one row per checked batch to a results table:
//
//     from morello.integrations import spark as morello_spark
//     morello_spark.check(spark, "catalog.sales.orders", state_table="ops.morello_state",
//               results_table="ops.morello_results")
//
// A Scala job that runs afterwards reads that table and stops if the newest batch alerted.
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object MorelloGate {
  def requireClean(spark: SparkSession, resultsTable: String, tableName: String): Unit = {
    val results = spark.table(resultsTable).filter(col("table_name") === tableName)
    val newest = results.agg(max("checked_at")).first().getString(0)
    val alerts = results.filter(col("checked_at") === newest && col("status") === "alert")
      .select("batch", "segment", "suspect_columns", "top_check", "top_ratio").collect()
    if (alerts.nonEmpty) {
      val msg = alerts.map(r => s"batch ${r.getString(0)} ${r.getString(1)}: ${r.getString(2)} (${r.getString(3)})").mkString("\n")
      throw new IllegalStateException(s"morello alert on $tableName:\n$msg")
    }
  }
}
