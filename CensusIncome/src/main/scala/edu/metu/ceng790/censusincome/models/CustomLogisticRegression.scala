/**
 * Class to implement Logistic Regression on Census Income data
 *
 */

package edu.metu.ceng790.censusincome.models

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

object CustomLogisticRegression {

  def train(trainingDataDF: DataFrame, featureNames: Seq[String],
            categoricalFeatureIndices: Seq[Int]): CrossValidatorModel = {

    // index categorical features
    val indexers = categoricalFeatureIndices.map { idx =>
      new StringIndexer()
        .setInputCol(featureNames(idx))
        .setOutputCol(s"${featureNames(idx)}Index")
        .setHandleInvalid("keep")
    }

    // index label column
    val labelIndexer = new StringIndexer()
      .setInputCol("income")
      .setOutputCol("label")

    // assemble all features into a feature vector
    val featureCols = Array("age", "fnlwgt", "educationNum", "capitalGain", "capitalLoss", "hoursPerWeek") ++
      categoricalFeatureIndices.map(idx => featureNames(idx) + "Index")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // initialize Logistic Regression model
    val logisticRegression = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
//      .setStandardization(true)

    // create a pipeline
    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(labelIndexer, assembler, logisticRegression))

    // create a parameter grid for hyperparameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(logisticRegression.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(logisticRegression.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // create a BinaryClassificationEvaluator
    val binaryEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    // create a CrossValidator
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(binaryEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5) // 5-fold cross-validation

    // fit the cross-validator to find the best model
    val cvModel = crossValidator.fit(trainingDataDF)

    cvModel
  }

  // the method to extract importance of features
  def extractLRCoefficients(model: PipelineModel, featureCols: Array[String], outputPath: String)(implicit spark: SparkSession): Unit = {
    import spark.implicits._

    val lrModel = model.stages.last.asInstanceOf[LogisticRegressionModel]
    val coefficients = lrModel.coefficients.toArray
    val featureCoefficients = featureCols.zip(coefficients).sortBy(-_._2.abs)

    val coefficientsDF = featureCoefficients.toSeq.toDF("Feature", "Coefficient")

    // Save the DataFrame to a single CSV file
    coefficientsDF.coalesce(1).write.option("header", "true").csv(outputPath)
  }
}
