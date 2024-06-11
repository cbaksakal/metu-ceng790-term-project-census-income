/**
 * Class to implement SVM on Census Income data
 *
 */

package edu.metu.ceng790.censusincome.models

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}

object CustomSVM {

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
      .setOutputCol("assembledFeatures")

    // initialize SVM model
    val svm = new LinearSVC()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(25)

    // normalize the features
    val scaler = new MinMaxScaler()
      .setInputCol("assembledFeatures")
      .setOutputCol("features")

    // create a parameter grid for hyperparameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(svm.regParam, Array(0.01, 0.1, 1.0))
      .build()

    // create a pipeline
    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(labelIndexer, assembler, scaler, svm))

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
}
