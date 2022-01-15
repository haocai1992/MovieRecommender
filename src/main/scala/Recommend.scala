import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.rdd.RDD

object Recommend {
  def main(args: Array[String])={
    if (args.length != 2) {
      println("Please input 2 parameters:")
      println("1. recommend type: '--U' for recommending movies to a user; '--M' for recommending users to a movie.")
      println("2. input id: UserID ('--U'); MovieID ('--M')")
      sys.exit(0)
    }
    val recommendType = args(0) // recommend type
    val inputID = args(1) // input id

    println("---------Initializing Spark-----------")
    val sparkConf = new SparkConf()
      .setAppName("Recommend")
      .set("spark.ui.showConsoleProgress", "false")
//      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting Up Logger----------")
    setLogger()

    println("----------Setting Up Data Path----------")
    //    val dataPath = "/home/caihao/Downloads/ml-100k/u.item" // local for test
    //    val modelPath = "/home/caihao/Downloads/ml-100k/ALSmodel" // local for test
    //    val checkpointPath = "home/caihao/Downloads/ml-100k/checkpoint/" // local for test
    val dataPath = "hdfs://localhost:9000/user/caihao/movie/u.item" // HDFS
    val modelPath = "hdfs://localhost:9000/user/caihao/movie/ALSmodel" // HDFS
    val checkpointPath = "hdfs://localhost:9000/user/caihao/checkpoint/" // HDFS
    sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)

    println("---------Preparing Data---------")
    val movieTitle: RDD[(Int, String)] = prepareData(sc, dataPath)
    movieTitle.checkpoint() // checkpoint data to avoid stackoverflow error

    println("---------Loading Model---------")
    val model = loadModel(sc, modelPath)

    println("---------Recommend---------")
    recommend(model, movieTitle, recommendType, inputID)

    sc.stop()
  }

  def setLogger(): Unit ={
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
  }

  def prepareData(sc: SparkContext, dataPath:String): RDD[(Int, String)] ={
    println("Loading Data......")
    // reads data from dataPath into Spark RDD.
    val itemRDD: RDD[String] = sc.textFile(dataPath)
    // only takes in first two fields (movieID, movieName).
    val movieTitle: RDD[(Int, String)] = itemRDD.map(line => line.split("\\|")).map(x => (x(0).toInt, x(1)))
    // return movieID->movieName map as Spark RDD
    movieTitle
  }

  def loadModel(sc: SparkContext, modelPath: String): Option[MatrixFactorizationModel] = {
    try {
      val model: MatrixFactorizationModel = MatrixFactorizationModel.load(sc, modelPath)
      Some(model)
    }
    catch {
      case e: Exception => None
    }
    finally {}
  }

  def recommend(model: Option[MatrixFactorizationModel], movieTitle:RDD[(Int, String)], arg1: String, arg2: String)={
    if (arg1 == "--U") {
      recommendMovies(model.get, movieTitle, arg2.toInt)
    }
    if (arg1 == "--M") {
      recommendUsers(model.get, movieTitle, arg2.toInt)
    }
  }

  def recommendMovies(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputUserID: Int) = {
    val recommendP = model.recommendProducts(inputUserID, 10)
    println(s"Recommending the following movies for user ${inputUserID.toString}:")
    recommendP.foreach(p => println(s"user: ${p.user}, recommended movie: ${movieTitle.lookup(p.product).mkString}, rating: ${p.rating}"))
  }

  def recommendUsers(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputMovieID: Int) = {
    val recommendU = model.recommendUsers(inputMovieID, 10)
    println(s"Recommending the following users for movie ${inputMovieID.toString}:")
    recommendU.foreach(u => println(s"movie: ${movieTitle.lookup(u.product).mkString}, recommended user: ${u.user}, rating: ${u.rating}"))
  }
}
