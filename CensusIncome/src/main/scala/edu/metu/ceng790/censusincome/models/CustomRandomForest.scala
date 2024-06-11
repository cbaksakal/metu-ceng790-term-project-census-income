/**
 * Class to implement Random Forest on Census Income data
 *
 */

package edu.metu.ceng790.censusincome.models

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

object CustomRandomForest {

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

    // initialize Random Forest model
    val randomForest = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setFeatureSubsetStrategy("auto")
      .setSeed(1234)

    // create a parameter grid for hyperparameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth, Array(4, 6, 8, 10))
      .addGrid(randomForest.maxBins, Array(48, 64, 80, 96))
      .addGrid(randomForest.impurity, Array("gini", "entropy"))
      .build()

    // create a pipeline
    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(labelIndexer, assembler, randomForest))

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

  def extractRFFeatureImportances(model: PipelineModel, featureCols: Array[String], outputPath: String)(implicit spark: SparkSession): Unit = {
    import spark.implicits._

    val rfModel = model.stages.last.asInstanceOf[RandomForestClassificationModel]
    val featureImportances = rfModel.featureImportances.toArray
    val featureImportancesSorted = featureCols.zip(featureImportances).sortBy(-_._2)

    val importancesDF = featureImportancesSorted.toSeq.toDF("Feature", "Importance")

    // Save the DataFrame to a single CSV file
    importancesDF.coalesce(1).write.option("header", "true").csv(outputPath)
  }

}
