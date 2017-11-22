name := "couchbase-spark-samples"

organization := "com.couchbase"

version := "1.0.0-SNAPSHOT"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-streaming" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "org.apache.bahir" %% "spark-streaming-twitter" % "2.0.0",
  "com.couchbase.client" %% "spark-connector" % "2.2.0",
  "org.apache.spark" %% "spark-mllib" % "2.2.0",
  "mysql" % "mysql-connector-java" % "5.1.37"
)

resolvers += Resolver.mavenLocal