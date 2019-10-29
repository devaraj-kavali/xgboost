/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{FloatType, IntegerType}
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.commons.logging.LogFactory

object MortgageTraining {

  def convertDFToArraysRDD(dataFrame: DataFrame, nExecutors: Int):
                      RDD[(Int, Int, Array[Float], Array[Float])] = {
    val df = dataFrame.coalesce(nExecutors)
    val selectedColumns = Seq(col("delinquency_12").cast(FloatType), col("features"))
    df.select(selectedColumns: _*).rdd.mapPartitions(rows => {
      val labelAndFeatures = rows.map {
        case Row(label: Float, features: Vector) => {
          val values = features.toDense.values.map(_.toFloat)
          (label, values)
        }
      }
      val batchSize = 32 << 10
      val labelBuf: ArrayBuffer[Float] = new ArrayBuffer(batchSize)
      val featureBuf: ArrayBuffer[Float] = new ArrayBuffer(batchSize)
      val batch = labelAndFeatures.take(batchSize)
      for (point <- batch) {
        labelBuf += point._1
        featureBuf ++= point._2
      }
      require(!labelBuf.isEmpty && !featureBuf.isEmpty)
      val m = labelBuf.length
      val n = featureBuf.length / m
      Iterator((m, n, labelBuf.toArray, featureBuf.toArray))
    })
  }

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      // scalastyle:off
      println("Usage: program input_path")
      sys.exit(1)
    }

    val inputPath = args(0)
    val sc = new SparkContext(new SparkConf())
    val spark = new SQLContext(sc)

    val nExecutors = sc.getConf.get("spark.executor.instances").toInt
    val nTaskCpus = sc.getConf.get("spark.task.cpus").toInt
    sc.setLogLevel("ERROR")

    val xgbParam = Map("features_col" -> "features",
      "label_col" -> "delinquency_12",
      "num_round" -> 100,
      "max_depth" -> 8,
      "max_leaves" -> 256,
      "alpha" -> 0.9f,
      "eta" -> 0.1f,
      "gamma" -> 0.1f,
      "subsample" -> 1.0f,
      "reg_lambda" -> 1.0f,
      "scale_pos_weight" -> 2.0f,
      "min_child_weight" -> 30.0f,
      "tree_method" -> "hist",
      "objective" -> "reg:squarederror",
      "gorw_policy" -> "lossguide",
      "num_workers" -> nExecutors,
      "nthread" -> nTaskCpus)
    val df = spark.read.format("parquet").load(inputPath)

    val t0 = System.nanoTime
    //val xgbClassifier = new XGBoostClassifier(xgbParam)
    //val xgbClassificationModel = xgbClassifier.fit(xgbInput)
    val tables =  convertDFToArraysRDD(df, nExecutors).cache()
    println(tables.count)
    val elapsed = (System.nanoTime - t0) / 1e9d

    spark.clearCache()
    sc.stop()
    println("Total elapsed time (seconds) = " + elapsed)
  }
}
