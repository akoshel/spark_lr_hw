package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BVector}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}

trait LinearRegressionParams extends PredictorParams {

  val learningRate: DoubleParam = new DoubleParam(
    this, "learningRate", "learning rate")
  val numIters: IntParam = new IntParam(
    this, "numIters", "number of iterations")

  setDefault(learningRate -> 1.0)
  setDefault(numIters -> 100)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setNumIters(value: Int): this.type = set(numIters, value)
}


class LinearRegression(override val uid: String)
  extends Regressor[Vector, LinearRegression, LinearRegressionModel]
    with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var coefficients: BVector[Double] = BVector.rand(numFeatures)
    val gradCol = "grad"

    val transformUdf = dataset.sqlContext.udf.register(uid + "_grad",
      (xRaw: Vector, y: Double) => {
        val x = xRaw.asBreeze.toDenseVector
        val grad = x * (sum(x * coefficients) - y)
        Vectors.fromBreeze(grad)
      }
    )

    for (_ <- 0 to $(numIters)) {
      val dataset_transformed = dataset.withColumn(gradCol, transformUdf(dataset($(featuresCol)), dataset($(labelCol))))
      val Row(Row(gradMean)) = dataset_transformed
        .select(Summarizer.metrics("mean").summary(dataset_transformed(gradCol)))
        .first()

      val grad: BVector[Double] = gradMean.asInstanceOf[DenseVector].asBreeze.toDenseVector
      coefficients = coefficients - $(learningRate) * grad
    }
    val params = Vectors.fromBreeze(coefficients)

    copyValues(new LinearRegressionModel(params)).setParent(this)
  }

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel protected[made](override val uid: String, coefficients: Vector)
  extends RegressionModel[Vector, LinearRegressionModel] with PredictorParams with MLWritable {

  def this(coefficients: Vector) = this(Identifiable.randomUID("linearRegressionModel"), coefficients)

  override def predict(features: Vector): Double = {
    sum(features.asBreeze.toDenseVector * coefficients.asBreeze.toDenseVector)
  }

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(coefficients))

  def getCoefs(): BVector[Double] = {
    coefficients.asBreeze.toDenseVector
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val params = Tuple1(coefficients.asInstanceOf[Vector])

      sqlContext.createDataFrame(Seq(params)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (coefficients) = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(coefficients)
      metadata.getAndSetParams(model)
      model
    }
  }
}
