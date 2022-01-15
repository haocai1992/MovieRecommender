import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

object RecommenderTrain {
  def main(args: Array[String]): Unit={
    println("---------Initializing Spark-----------")
    val sparkConf = new SparkConf()
      .setAppName("RecommenderTrain")
//      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting Up Logger----------")
    setLogger()

    println("----------Setting up data path----------")
//    val dataPath = "/home/caihao/Downloads/ml-100k/u.data" // local for test
//    val modelPath = "/home/caihao/Downloads/ml-100k/ALSmodel" // local for test
//    val checkpointPath = "home/caihao/Downloads/ml-100k/checkpoint/" // local for test
    val dataPath = "hdfs://localhost:9000/user/caihao/movie/u.data" // HDFS
    val modelPath = "hdfs://localhost:9000/user/caihao/movie/ALSmodel" // HDFS
    val checkpointPath = "hdfs://localhost:9000/user/caihao/checkpoint/" // HDFS
    sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)

    println("---------Preparing Data---------")
    val ratingsRDD: RDD[Rating] = PrepareData(sc, dataPath)
    ratingsRDD.checkpoint() // checkpoint data to avoid stackoverflow error

    println("---------Training---------")
    println("Start ALS training, rank=5, iteration=20, lambda=0.1")
    val model: MatrixFactorizationModel = ALS.train(ratingsRDD, 5, 20, 0.1)

    println("----------Saving Model---------")
    saveModel(sc, model, modelPath)
    sc.stop()
  }

  def setLogger(): Unit ={
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
  }

  def PrepareData(sc: SparkContext, dataPath:String): RDD[Rating] ={
    // reads data from dataPath into Spark RDD.
    val file: RDD[String] = sc.textFile(dataPath)
    // only takes in first three fields (userID, itemID, rating).
    val ratingsRDD: RDD[Rating] = file.map(line => line.split("\t") match {
      case Array(user, item, rate, _) => Rating(user.toInt, item.toInt, rate.toDouble)
    })
    println(ratingsRDD.first()) // Rating(196,242,3.0)
    // return processed data as Spark RDD
    ratingsRDD
  }

  def saveModel(context: SparkContext, model:MatrixFactorizationModel, modelPath: String): Unit ={
    try {
      model.save(context, modelPath)
    }
    catch {
      case e: Exception => println("Error Happened when saving model!!!")
    }
  finally {
  }
  }
}
