package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => BVector}
import org.apache.spark.ml.feature.{LSH, LSHModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsWriter, Identifiable, MLWriter}
import org.apache.spark.sql.types.StructType

trait RandomHyperplaneLSHParams extends Params {
}

class RandomHyperplaneLSH(override val uid: String)
  extends LSH[RandomHyperplaneLSHModel] {

  def this() = {
    this(Identifiable.randomUID("RandomHyperplaneLSH"))
  }

  override protected[this] def createRawLSHModel(inputDim: Int): RandomHyperplaneLSHModel = {
    val randHyperPlanes: Array[Vector] = {
      Array.fill($(numHashTables)) {
        val randArray = BVector.rand(inputDim).data.map(x => if (x > 0.5) -1.0 else 1.0)
        Vectors.fromBreeze(BVector(randArray))
      }
    }
    new RandomHyperplaneLSHModel(uid, randHyperPlanes)
  }

  override def copy(extra: ParamMap): RandomHyperplaneLSH = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}


class RandomHyperplaneLSHModel private[made](
                                              override val uid: String,
                                              private[made] val randHyperPlanes: Array[Vector]
                                            ) extends LSHModel[RandomHyperplaneLSHModel] {


  private[made] def this(randHyperPlanes: Array[Vector]) =
    this(Identifiable.randomUID("RandomHyperplaneLSH"), randHyperPlanes)

  override protected[ml] def hashFunction(elems: Vector): Array[Vector] = {
    val hashValues = randHyperPlanes.map(
      randHyperPlane => if (elems.dot(randHyperPlane) >= 0) 1 else -1
    )
    hashValues.map(Vectors.dense(_))
  }

  override protected[ml] def keyDistance(x: Vector, y: Vector): Double = {
    if ((Vectors.norm(x, 2) == 0) || (Vectors.norm(y, 2) == 0)) {
      1.0
    } else {
      1.0 - (x.dot(y) / (Vectors.norm(x, 2) * Vectors.norm(y, 2)))
    }
  }

  override protected[ml] def hashDistance(x: Array[Vector], y: Array[Vector]): Double = {
    x.zip(y).map(item => if (item._1 != item._2) 1 else 0).sum.toDouble / x.size
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val params = Tuple1(randHyperPlanes.asInstanceOf[Vector])

      sqlContext.createDataFrame(Seq(params)).write.parquet(path + "/randHyperPlanes")
    }
  }

  override def copy(extra: ParamMap): RandomHyperplaneLSHModel = copyValues(new RandomHyperplaneLSHModel(randHyperPlanes))

}
