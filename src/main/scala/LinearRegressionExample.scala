import com.couchbase.spark.sql._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

/**
  * This is a simple example of how to use  MLlib with Couchbase Spark Connector
  * Predicting house prices
  */
object LinearRegressionExample {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("SparkSQLExample")
      .master("local[*]") // use the JVM as the master, great for testing
      .config("spark.couchbase.nodes", "127.0.0.1") // connect to couchbase on localhost
      .config("spark.couchbase.bucket.houses_prices", "") // open the houses_prices bucket with empty password
      .config("com.couchbase.username", "Administrator")
      .config("com.couchbase.password", "couchbase")
      .getOrCreate()

    //Loading data from the database, you could add some filters here or even write a N1QL query.
    //Please check https://github.com/couchbaselabs/couchbase-spark-samples for more info
    val houses = spark.read.couchbase()

    //handling categorical variables
    val df = transformCategoricalFeatures(houses)

    df.show()

    //just using almost all columns as features, no special feature engineering here
    val features = Array("sqft_living", "bedrooms",
      "gradeVec", "waterfront",
      "bathrooms", "view",
      "conditionVec", "sqft_above",
      "sqft_basement", "zipcode",
      "sqft_lot", "floors",
      "yr_built", "zipcodeVec", "yr_renovatedVec")

    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    //the Linear Regression implementation expect a feature called "label"
    val renamedDF = assembler.transform(df.withColumnRenamed("price", "label"))

    //our training data will be all entries which has price not null
    val data = renamedDF.select("label", "features").filter("price is not null")

    //let's split our data in test and training (a common thing during model selection)
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val trainingData = splits(0).cache()
    //let's ignore the test data for now as we are not doing model selection
    val testData = splits(1)

    val lr = new LinearRegression()
      .setMaxIter(1000)
      .setStandardization(true)
      .setRegParam(0.1)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(trainingData)

    //printing some statistics of the trained model
    //OBS: In a real world application you should never train and use a model on the fly
    // A good model selection is crucial for a reasonable output
    // https://en.wikipedia.org/wiki/Model_selection
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")


    //selecting all the rows in our dataset where the price is null
    val missingPriceData = renamedDF.select("features")
      .filter("price is null")

    missingPriceData.show()

    //printing out the predicted values
    val predictedValues = lrModel.transform(missingPriceData)
    predictedValues.select("prediction").show()
  }

  def transformCategoricalFeatures(dataset: Dataset[_]): DataFrame = {
    val df1 = encodeFeature("zipcode", "zipcodeVec", dataset)
    val df2 = encodeFeature("yr_renovated", "yr_renovatedVec", df1)
    val df3 = encodeFeature("condition", "conditionVec", df2)
    encodeFeature("grade", "gradeVec", df3)
  }

  def encodeFeature(featureName: String, outputName: String, dataset: Dataset[_]): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol(featureName)
      .setOutputCol(featureName + "Index")
      .fit(dataset)

    val indexed = indexer.transform(dataset)

    val encoder = new OneHotEncoder()
      .setInputCol(featureName + "Index")
      .setOutputCol(outputName)

    encoder.transform(indexed)
  }
}
