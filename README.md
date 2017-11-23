Zero Effort Machine Learning with Couchbase and Spark MLlib
=======================


In the last few years we have witnessed the rise of Machine Learning, a 50+ years old 	
technique that has finally reached the masses. Surprisingly a lot of companies are still not doing anything 
in this field, in part I believe due to the lack of knowledge of how it fits in their business and also because 
this topic still seems to be lirrle cloudy in most of the programmers heads, that is why I would like
show you today how you can start with machine learning with almost zero effort.

In the most basic level of machine learning we have something called Linear Regression which is roughly an algorithm 
that tries to "explain" a number by giving weight to a set of features, let's see some examples:

* The price of a house could be explained by things like size, location, number of bedrooms and bathrooms;
* The price of a car could be explained by its model, year, mileage, condition, etc;
* The time spent for a given task could be predicted by the number of subtasks, level of difficulty, worker experience, etc;


There a plenty of use cases were Linear Regression (or other Regression types) can be used, but lets focus in the first
one related to house prices. Imagine we a running a real state company in a particular region of 
the country, as we are not a new company, we do have some data of which were the houses sold in the past and
for how much, in this case each row in our historical data will look like this:

```javascript
  {
    "id": 7129300520,
    "date": "20141013T000000",
    "price": 221900,
    "bedrooms": 3,
    "bathrooms": 1,
    "sqft_living": 1180,
    "sqft_lot": 5650,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1180,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 0,
    "zipcode": 98178,
    "lat": 47.5112,
    "long": -122.257,
    "sqft_living15": 1340,
    "sqft_lot15": 5650
  }
```

## The problem

Now imagine you just joined the company and you have to sell the following house:

```javascript
  {       
  "id": 1000001,
  "date": "20150422T000000",
  "bedrooms": 6,
  "bathrooms": 3,
  "price": null,
  "sqft_living": 2400,
  "sqft_lot": 9373,
  "floors": 2,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 2400,
  "sqft_basement": 0,
  "yr_built": 1991,
  "yr_renovated": 0,
  "zipcode": 98002,
  "lat": 47.3262,
  "long": -122.214,
  "sqft_living15": 2060,
  "sqft_lot15": 7316
 }
```
For how much would you sell it?

Though question right? Luckily that is exactly the question Linear Regression would help you to answer.


## The Answer

For this tutorial you would need to install:

* Couchbase Server 4+
* Spark 2.2
* [SBT](http://www.scala-sbt.org/download.html) (as we are running using scala)

With your Couchbase Server running, go to the administrative portal at http://127.0.0.1:8091 and create a new bucket called
**houses_prices**

![bucket creation](imgs/bucket_creation.png "houses_prices bucket creation")


Now lets clone our tutorial code:
`git clone https://github.com/couchbaselabs/couchbase-spark-mllib-sample.git`

In root folder there is a file called **house_prices_train_data.zip**, it is our dataset which I borrowed from an old machine 
learning course on Coursera. Please unzip it and then run the following command:

`./cbimport json -c couchbase://127.0.0.1 -u YOUR_USER -p YOUR_PASSWORD -b houses_prices -d <PATH_TO_UNZIPED_FILE>/house_prices_train_data -f list -g key::%id% -t 4`

**TIP:** If you are not familiar with **cbimport** please [check this tutorial](https://developer.couchbase.com/documentation/server/current/tools/cbimport.html)


If your command ran successfully, you should notice that your **houses_prices** bucket has been populated:


For the sake of letting you explore this data let's quickly create an index for it, run the following command in the query editor:

```
CREATE PRIMARY INDEX ON `houses_prices`
```

![Index creation](imgs/index_creation.png "Creating indexes for houses_prices bucket")









 









