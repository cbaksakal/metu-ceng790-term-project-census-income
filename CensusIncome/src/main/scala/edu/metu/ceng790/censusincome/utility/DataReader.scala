/**
 * Class to read data from csv file
 *
 */

package edu.metu.ceng790.censusincome.utility

import org.apache.spark.sql.functions.{col, regexp_replace, trim}
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataReader {

  def readAndPreprocessData(spark: SparkSession, path: String, featureNames: Seq[String], intTypeFeatureIndices: Seq[Int]): DataFrame = {
    // read data from file
    var csvDataDF = spark.read.format("csv")
      .option("header", "false")
      .load(path)

    // change the names of all columns. also cast the type of INT columns that were previously fetched as String
    csvDataDF = preProcessDF(csvDataDF, featureNames, intTypeFeatureIndices)

    // print schema of the DF and show some lines to ensure data is fetched correctly
    csvDataDF.printSchema()
    csvDataDF.show(5)

    // return data frame
    csvDataDF
  }

  // function to rename columns and cast integer types
  private def preProcessDF(df: DataFrame, featureNames: Seq[String], intTypeFeatureIndices: Seq[Int]): DataFrame = {
    var newDF = df
    for (i <- newDF.columns.indices) {
      newDF = newDF.withColumnRenamed("_c" + i, featureNames(i))
      if (intTypeFeatureIndices.contains(i))
        newDF = newDF.withColumn(featureNames(i), newDF.col(featureNames(i)).cast("integer"))
    }
    newDF
  }
}
