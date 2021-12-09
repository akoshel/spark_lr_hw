import breeze.linalg.{DenseMatrix, DenseVector}

  val X = DenseMatrix.rand(100000, 3)
  val label = X * DenseVector(0.5, -0.1, 0.2)
  val data = DenseMatrix.horzcat(X, label.asDenseMatrix.t)

