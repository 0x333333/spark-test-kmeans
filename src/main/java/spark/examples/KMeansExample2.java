package org.apache.spark.examples;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import org.apache.spark.storage.StorageLevel;

public class KMeansExample2 {
  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setAppName("K-means Example")
                .set("spark.executor.memory", "32M")
                .set("spark.rdd.compress", "true")
                .set("spark.io.compression.codec", "lz4");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load and parse data
    String path = "kmeans_data.txt";
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<Vector> parsedData = data.map(
      new Function<String, Vector>() {
        public Vector call(String s) {
          String[] sarray = s.split(" ");
          double[] values = new double[sarray.length];
          for (int i = 0; i < sarray.length; i++)
            values[i] = Double.parseDouble(sarray[i]);
          return Vectors.dense(values);
        }
      }
    );
    parsedData.persist(StorageLevel.MEMORY_ONLY());

    System.out.println("\n\n\n\n");
    long startTime = System.nanoTime();

    // Cluster the data into two classes using KMeans
    int numClusters = 2;
    int numIterations = 20;
    KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    double WSSSE = clusters.computeCost(parsedData.rdd());
    long endTime = System.nanoTime();
    System.out.println("Execution Time: " + (endTime - startTime)/1000000 + " ms");
    System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
  }
}