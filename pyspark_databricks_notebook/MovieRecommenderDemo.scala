package MovieRecommenderDemo

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object MovieRecommenderDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      //      .master("local")
      .appName("movie recommender")
      .enableHiveSupport()
      .config("spark.master", "local")
      .getOrCreate()
    val sc = spark.sparkContext
    spark.sparkContext.setLogLevel("WARN")

    // Write Rdd to HDFS - single partition
    //    val modelNames = Array("FM", "FFM", "DEEPFM", "NFM", "DIN", "DIEN")
    //    val modelNamesRdd =spark.sparkContext.parallelize(modelNames, 1)
    //    modelNamesRdd.saveAsTextFile("hdfs://localhost:9000/user/root/modelNames")

    // Write Rdd to HDFS - two partitions
    //    val modelNames3 = Array("FM", "FFM", "DEEPFM", "NFM", "DIN", "DIEN")
    //    val modelNames3Rdd =spark.sparkContext.parallelize(modelNames3, 2)
    //    modelNames3Rdd.saveAsTextFile("hdfs://localhost:9000/user/root/modelNames3")

    // read file on HDFS
    //    val modelNames2 = spark.sparkContext.textFile("hdfs://localhost:9000/user/root/modelNames/part-00000")
    //    val modelNames4 = spark.sparkContext.textFile("hdfs://localhost:9000/user/root/modelNames3/")
    //    println(modelNames2.count())
    //    println(modelNames4.count())

    // add files to Driver
    //    val files = "hdfs://localhost:9000/user/root/modelNames/part-00000"
    //    spark.sparkContext.addFile(files)
    //    val path = SparkFiles.get("part-00000")
    //    println(path)

    // use scala's IO to read files using this path
    //    val source = scala.io.Source.fromFile(path)
    //    val lineIterator = source.getLines()
    //    val lines = lineIterator.toArray
    //    println(lines.mkString(","))

    // decide if file paths exists on HDFS
    //    val conf = spark.sparkContext.hadoopConfiguration
    //    val path = new Path("hdfs://localhost:9000/user/root/modelNames/part-00000")
    //    val fs = path.getFileSystem(conf) // get file's path information in HDFS
    //    val modelNamesExists = fs.exists(path)
    //    val modelNames1Exists = fs.exists(new Path("hdfs://localhost:9000/user/root/modelNames1/part-00000"))
    //    println(modelNamesExists)
    //    println(modelNames1Exists)

    // read movie data
    val rawUD = spark.sparkContext.textFile("hdfs://localhost:9000/user/root/movie/u.data")
    println(rawUD.count()) // 100000
    println(rawUD.first()) // 196	242	3	881250949

    // read first three words and turn them into Rating object.
    val ratingsRDD: RDD[Rating] = rawUD.map(line => line.split("\t") match {
      case Array(user, item, rate, _) => Rating(user.toInt, item.toInt, rate.toDouble)
    })
    println(ratingsRDD.first()) // Rating(196,242,3.0)


    // look at unique users
    val numUsers: Long = ratingsRDD.map(_.user).distinct().count()
    println(numUsers) // 943

    // train model
    val model = ALS.train(ratingsRDD, 10, 10, 0.01)
    // recommend top 5 films for user 100
    val films = model.recommendProducts(100, 5)
    // look at ratings of film 1311 by user 100
    val rating = model.predict(100, 1311)
    // look at top 5 users who will like film 200
    val users = model.recommendUsers(200, 5)
    println(films.mkString)
    println(rating)
    println(users.mkString)

    // show movie names
    val itemRDD: RDD[String] = spark.sparkContext.textFile("hdfs://localhost:9000/user/root/movie/u.item")
    println(itemRDD.count()) // 1682
    itemRDD.take(3).foreach(println)

    // create map between movie ID and movie name
    val movieTitle: RDD[(Int, String)] = itemRDD.map(line => line.split("\\|")).map(x => (x(0).toInt, x(1)))

    // start recommending!
    val recommendP = model.recommendProducts(100, 5)
    recommendP.foreach(p => println(s"user: ${p.user}, recommended: ${movieTitle.lookup(p.product).mkString}, rating: ${p.rating}"))
  }

}
