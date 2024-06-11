/**
 * Driver class
 *
 */

package edu.metu.ceng790.censusincome

import edu.metu.ceng790.censusincome.models.CustomGradientBoostedTrees.extractGBTFeatureImportances
import edu.metu.ceng790.censusincome.models.CustomRandomForest.extractRFFeatureImportances
import edu.metu.ceng790.censusincome.models.{CustomGradientBoostedTrees, CustomLogisticRegression, CustomRandomForest, CustomSVM}
import edu.metu.ceng790.censusincome.utility.DataReader
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

object CensusIncome {
  def main(args: Array[String]): Unit = {

    // paths to data
    val DATA_PATH = "data/adult.data"

    // initialize Spark session
    val spark = SparkSession.builder()
      .appName("CensusIncome")
      .config("spark.master", "local[*]")
      .getOrCreate()

    // feature names and indices
    val featureNames = Seq("age", "workclass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation",
      "relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "income")

    // indices of the numerical features. will later be used to cast String types to Int
    val intTypeFeatureIndices = Seq(0, 2, 4, 10, 11, 12)

    // indices of categorical features
    val categoricalFeatureIndices = Seq(1, 3, 5, 6, 7, 8, 9, 13)

    // read and preprocess data
    val csvDataDF = DataReader.readAndPreprocessData(spark, DATA_PATH, featureNames, intTypeFeatureIndices).cache()

    // count of all data
    val totalSampleCount = csvDataDF.count()
    println(s"Total sample count: $totalSampleCount")

    // check for null values
    val nullCount = csvDataDF.filter(el => el.anyNull).count()
    println(s"Total null value count: $nullCount")

    // check whether dataset is balanced, and show the counts and percentages
    val labelCounts = csvDataDF.groupBy("income").count()

    labelCounts.collect().foreach { row =>
      val label = row.getString(0)
      val count = row.getLong(1)
      val proportion = (count.toDouble / totalSampleCount) * 100
      printf("Label: %s, Count: %d, Proportion: %.2f%%%n", label, count, proportion)
    }

    // before getting into balancing training data, split test data
    val Array(initialTrainingDataDF, testDataDF) = csvDataDF.randomSplit(Array(0.75, 0.25), seed = 1234L)

    // balance the dataset for Logistic Regression
    val classCounts = csvDataDF.groupBy("income").count().collect()

    val majorityClassLabel = classCounts.maxBy(_.getLong(1)).getString(0)
    val minorityClassLabel = classCounts.minBy(_.getLong(1)).getString(0)

    val majorityClass = csvDataDF.filter(col("income") === majorityClassLabel)
    val minorityClass = csvDataDF.filter(col("income") === minorityClassLabel)

    val ratio = majorityClass.count().toDouble / minorityClass.count().toDouble

    // display results
    println(s"Majority Class: ${majorityClassLabel} Count: ${majorityClass.count()}")
    println(s"Minority Class: ${minorityClassLabel} Count: ${minorityClass.count()}")
    println(s"Ratio: $ratio")

    // oversample the minority class
    val oversampledMinorityClass = minorityClass.sample(withReplacement = true, ratio)
    val balancedTrainingDataDF = majorityClass.union(oversampledMinorityClass).cache()

    println(s"Balanced data count: ${balancedTrainingDataDF.count()}")

    /*********************************************** SVM *********************************************************/
    //    // first, train the model with imbalanced data
    //    val svmModelDefault = CustomSVM.train(initialTrainingDataDF, featureNames, categoricalFeatureIndices)
    //    evaluateModel(svmModelDefault, initialTrainingDataDF, testDataDF, "SVM")

    // then, in order to make comparison, train the model with balanced data
    val svmModelBalanced = CustomSVM.train(balancedTrainingDataDF, featureNames, categoricalFeatureIndices)
    evaluateModel(svmModelBalanced, balancedTrainingDataDF, testDataDF, "SVM")

    // select relevant columns and save to CSV
    val predictionsWithProbSVM = getPredictionsWithProbabilities(svmModelBalanced, testDataDF)
    predictionsWithProbSVM.select("label", "probability")
      .write.option("header", "true").csv("producedData/svm_predictions.csv")

    /**************************************** Logistic Regression *************************************************/
    // first, train the model with imbalanced data
    val lrModelDefault = CustomLogisticRegression.train(initialTrainingDataDF, featureNames, categoricalFeatureIndices)
    evaluateModel(lrModelDefault, initialTrainingDataDF, testDataDF, "Logistic Regression")

    //    // then, in order to make comparison, train the model with balanced data
    //    val lrModelBalanced = CustomLogisticRegression.train(balancedTrainingDataDF, featureNames, categoricalFeatureIndices)
    //    evaluateModel(lrModelBalanced, balancedTrainingDataDF, testDataDF, "Logistic Regression")

    // select relevant columns and save to CSV
    val predictionsWithProbLR = getPredictionsWithProbabilities(lrModelDefault, testDataDF)
    predictionsWithProbLR.select("label", "probability")
      .write.option("header", "true").csv("producedData/lr_predictions.csv")

    // Extract and save logistic regression coefficients
    CustomLogisticRegression.extractLRCoefficients(lrModelDefault.bestModel.asInstanceOf[PipelineModel],
      featureNames.toArray, "producedData/lr_feature_importances.csv")(spark)

    /***************************************** Gradient-Boosted Trees **********************************************/
    //    // first, train the model with imbalanced data
    //    val gbtModelDefault = CustomGradientBoostedTrees.train(initialTrainingDataDF, featureNames, categoricalFeatureIndices)
    //    evaluateModel(gbtModelDefault, initialTrainingDataDF, testDataDF, "Gradient-Boosted Trees")

    // then, in order to make comparison, train the model with balanced data
    val gbtModelBalanced = CustomGradientBoostedTrees.train(balancedTrainingDataDF, featureNames, categoricalFeatureIndices)
    evaluateModel(gbtModelBalanced, balancedTrainingDataDF, testDataDF, "Gradient-Boosted Trees")

    // select relevant columns and save to CSV
    val predictionsWithProbGBT = getPredictionsWithProbabilities(gbtModelBalanced, testDataDF)
    predictionsWithProbGBT.select("label", "probability")
      .write.option("header", "true").csv("producedData/gbt_predictions.csv")

    // Extract and save feature importances for Gradient-Boosted Trees
    val bestPipelineModelGBT = gbtModelBalanced.bestModel.asInstanceOf[PipelineModel]
    extractGBTFeatureImportances(bestPipelineModelGBT, featureNames.toArray, "producedData/gbt_feature_importances.csv")(spark)

    /***************************************** Random Forest ********************************************************/
    // train the model with imbalanced data since Random Forest model is robust to data imbalance
    val rfModelDefault = CustomRandomForest.train(initialTrainingDataDF, featureNames, categoricalFeatureIndices)
    evaluateModel(rfModelDefault, initialTrainingDataDF, testDataDF, "Random Forest")

    //    // train the model with balanced data
    //    val rfModelBalanced = CustomRandomForest.train(balancedTrainingDataDF, featureNames, categoricalFeatureIndices)
    //    evaluateModel(rfModelBalanced, balancedTrainingDataDF, testDataDF, "Random Forest")

    // select relevant columns and save to CSV
    val predictionsWithProb = getPredictionsWithProbabilities(rfModelDefault, testDataDF)
    predictionsWithProb.select("label", "probability")
      .write.option("header", "true").csv("producedData/rf_predictions.csv")

    // Extract and save Random Forest feature importances
    val bestPipelineModelRF = rfModelDefault.bestModel.asInstanceOf[PipelineModel]
    extractRFFeatureImportances(bestPipelineModelRF, featureNames.toArray, "producedData/rf_feature_importances.csv")(spark)

    /****************************************************************************************************************/


    // stop Spark
    spark.stop()
  }

  private def evaluateModel(model: CrossValidatorModel, trainingData: DataFrame,
                            testData: DataFrame, modelName: String): Unit = {

    // make predictions on the training and test data
    val trainPredictions = model.transform(trainingData)
    val testPredictions = model.transform(testData)

    // binary classification evaluator for Area Under ROC
    val binaryEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val trainAUC = binaryEvaluator.evaluate(trainPredictions)
    val testAUC = binaryEvaluator.evaluate(testPredictions)

    // multiclass classification evaluator for accuracy, precision, recall, F1-score
    val multiclassEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // accuracy
    multiclassEvaluator.setMetricName("accuracy")
    val trainAccuracy = multiclassEvaluator.evaluate(trainPredictions)
    val testAccuracy = multiclassEvaluator.evaluate(testPredictions)

    // precision
    multiclassEvaluator.setMetricName("weightedPrecision")
    val trainPrecision = multiclassEvaluator.evaluate(trainPredictions)
    val testPrecision = multiclassEvaluator.evaluate(testPredictions)

    // recall
    multiclassEvaluator.setMetricName("weightedRecall")
    val trainRecall = multiclassEvaluator.evaluate(trainPredictions)
    val testRecall = multiclassEvaluator.evaluate(testPredictions)

    // F1-Score
    multiclassEvaluator.setMetricName("f1")
    val trainF1 = multiclassEvaluator.evaluate(trainPredictions)
    val testF1 = multiclassEvaluator.evaluate(testPredictions)

    // print out the results
    println(s"$modelName - Train AUC: %.4f".format(trainAUC))
    println(s"$modelName - Test AUC: %.4f".format(testAUC))
    println()
    println(s"$modelName - Train Accuracy: %.4f".format(trainAccuracy))
    println(s"$modelName - Test Accuracy: %.4f".format(testAccuracy))
    println()
    println(s"$modelName - Train Precision: %.4f".format(trainPrecision))
    println(s"$modelName - Test Precision: %.4f".format(testPrecision))
    println()
    println(s"$modelName - Train Recall: %.4f".format(trainRecall))
    println(s"$modelName - Test Recall: %.4f".format(testRecall))
    println()
    println(s"$modelName - Train F1 Score: %.4f".format(trainF1))
    println(s"$modelName - Test F1 Score: %.4f".format(testF1))
  }

  private def getPredictionsWithProbabilities(model: CrossValidatorModel, dataDF: DataFrame): DataFrame = {
    // Get the best model from cross-validation
    val bestModel = model.bestModel.asInstanceOf[PipelineModel]

    // Make predictions
    val predictions = bestModel.transform(dataDF)

    // Check if the probability column exists
    if (predictions.columns.contains("probability")) {
      // Define UDF to extract the probability of the positive class from the probability column
      val extractProbability: UserDefinedFunction = udf { prob: org.apache.spark.ml.linalg.Vector =>
        prob(1)
      }
      predictions.withColumn("probability", extractProbability(col("probability")))
    } else {
      // Define UDF to extract the probability of the positive class from the rawPrediction column
      val extractProbabilityFromRaw: UserDefinedFunction = udf { rawPrediction: org.apache.spark.ml.linalg.Vector =>
        1.0 / (1.0 + math.exp(-rawPrediction(1)))
      }
      predictions.withColumn("probability", extractProbabilityFromRaw(col("rawPrediction")))
    }
  }
}