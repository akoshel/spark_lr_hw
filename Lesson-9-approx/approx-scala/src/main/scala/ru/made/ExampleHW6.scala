package ru.made

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, MinHashLSH, RegexTokenizer}
import org.apache.spark.ml.{Pipeline, linalg, evaluation}
import org.apache.spark.ml.made.{RandomHyperplaneLSH, MinHashChild}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ExampleHW6 {

  val preprocessingPipe = new Pipeline()
    .setStages(Array(
      new RegexTokenizer()
        .setInputCol("Review")
        .setOutputCol("tokenized")
        .setPattern("\\W+"),
      new HashingTF()
        .setInputCol("tokenized")
        .setOutputCol("tf")
        .setBinary(true)
        .setNumFeatures(1000),
      new HashingTF()
        .setInputCol("tokenized")
        .setOutputCol("tf2")
        .setNumFeatures(1000),
      new IDF()
        .setInputCol("tf2")
        .setOutputCol("tfidf")
    ))

  val metrics = new RegressionEvaluator()
    .setLabelCol("Rating")
    .setPredictionCol("predict")
    .setMetricName("rmse")

  def main(args: Array[String]): Unit = {
    // Создает сессию спарка
    val spark = SparkSession.builder()
      // адрес мастера
      .master("local[*]")
      // имя приложения в интерфейсе спарка
      .appName("made-demo")
      // взять текущий или создать новый
      .getOrCreate()

    // синтаксический сахар для удобной работы со спарк

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("C:/Homeworks/Lesson-6-spark-sbt-example/data/tripadvisor_hotel_reviews.csv").sample(0.1, 0)

    val Array(train, test) = df.randomSplit(Array(0.8, 0.2), 0)

    val pipe = preprocessingPipe.fit(train)

    val trainFeatures = pipe.transform(train).cache()
    val testFeatures = pipe.transform(test)
    trainFeatures.count()

    val clsh =  new RandomHyperplaneLSH()
      .setInputCol("tfidf")
      .setOutputCol("buckets")
      .setNumHashTables(3)
      .fit(trainFeatures)

    val vec = testFeatures.select("tf").head(1)(0)(0).asInstanceOf[linalg.SparseVector]

    clsh.approxNearestNeighbors(trainFeatures, vec, 5, "distance")
      .agg(count("*"), avg("distance")).collect()

    val st = System.currentTimeMillis()
    val neigh = clsh.approxNearestNeighbors(trainFeatures, vec, 5, "distance")
    val res = neigh.agg(count("*"), avg("distance")).collect()
    println("Results", res(0)(0), res(0)(1), (System.currentTimeMillis() - st) / 1000.0)

  }
}