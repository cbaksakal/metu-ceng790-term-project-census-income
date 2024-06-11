/**
 * Class to implement Gradient Boosted Trees on Census Income data
 *
 */

package edu.metu.ceng790.censusincome.models

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object CustomGradientBoostedTrees {

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

    // initialize Gradient-Boosted Tree model
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(25)
      .setFeatureSubsetStrategy("auto")

    // create a parameter grid for hyperparameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(4, 6, 8))
      .addGrid(gbt.maxBins, Array(48, 64, 80))
      .addGrid(gbt.stepSize, Array(0.05, 0.1, 0.2))
      .build()

    // create a pipeline
    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(labelIndexer, assembler, gbt))

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

  def extractGBTFeatureImportances(model: PipelineModel, featureCols: Array[String], outputPath: String)(implicit spark: SparkSession): Unit = {
    import spark.implicits._

    val gbtModel = model.stages.last.asInstanceOf[GBTClassificationModel]
    val featureImportances = gbtModel.featureImportances.toArray
    val featureImportancesSorted = featureCols.zip(featureImportances).sortBy(-_._2)

    val importancesDF = featureImportancesSorted.toSeq.toDF("Feature", "Importance")

    // Save the DataFrame to a single CSV file
    importancesDF.coalesce(1).write.option("header", "true").csv(outputPath)
  }

}
