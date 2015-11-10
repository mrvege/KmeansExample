/**
 * Created by dm on 11/9/15.
 */

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object KmeansExample {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("KmeansAuthors").setMaster("local[4]")
    val sc = new SparkContext(sparkConf)
    //  val data = sc.textFile("hdfs://localhost:9000/input/feature_1109_2.csv")
    val data = sc.textFile(args(0))
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = args(1).toInt
    val numIterations = args(2).toInt
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors

    //println("Within Set Sum of Squared Errors = " + WSSSE)

    val clusterCenters = clusters.clusterCenters
    val labels = clusters.predict(parsedData)
    labels.saveAsTextFile(args(3))
    // Save and load model
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    //val sameModel = KMeansModel.load(sc, "myModelPath")
  }
}
