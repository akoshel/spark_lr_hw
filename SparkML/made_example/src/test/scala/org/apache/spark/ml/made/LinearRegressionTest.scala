package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.made.StandardScalerTest.sqlc
import org.apache.spark.sql.DataFrame

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta: Double = 0.001
  val mseBnd: Double = 0.0001
  val learningRate: Double = 1
  val numIters: Int = 100
  lazy val df: DataFrame = LinearRegressionTest._data

  private def validateModel(model: LinearRegressionModel, predicts: DataFrame): Unit = {

    val evaluator = new RegressionEvaluator()
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMetricName("mse")

    val mse = evaluator.evaluate(predicts)
    mse should be < mseBnd
  }

  "Estimator" should "validate weights" in {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setLearningRate(learningRate)
      .setNumIters(numIters)

    val model = lr.fit(df)
    val weights = model.getCoefs()

    weights(0) should be(0.5 +- delta)
    weights(1) should be(-0.1 +- delta)
    weights(2) should be(0.2 +- delta)
  }

  "Model" should "validate MSE" in {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setLearningRate(learningRate)
      .setNumIters(numIters)

    val model = lr.fit(df)
    validateModel(model, model.transform(df))
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setLearningRate(learningRate)
        .setNumIters(numIters)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline
      .load(tmpFolder.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]

    validateModel(model, model.transform(df))
  }


  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setLearningRate(learningRate)
        .setNumIters(numIters)
    ))

    val model = pipeline.fit(df)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(df))
  }
}

object LinearRegressionTest extends WithSpark {

  import spark.implicits._
  import breeze.linalg._

  lazy val X = DenseMatrix.rand(1000, 3)
  lazy val label = X * DenseVector(0.5, -0.1, 0.2)
  lazy val data = DenseMatrix.horzcat(X, label.asDenseMatrix.t)
  lazy val dataRaw = data(*, ::).iterator
    .map(x => (x(0), x(1), x(2), x(3)))
    .toSeq.toDF("x1", "x2", "x3", "y")
  lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("x1", "x2", "x3"))
    .setOutputCol("features")

  lazy val _data: DataFrame = assembler
    .transform(dataRaw)
}

