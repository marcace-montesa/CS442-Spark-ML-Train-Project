#!/usr/bin/env python
# coding: utf-8

# # **Project 2**
# **Marc Ace Montesa** 
# 
# **CS 442**

# In[2]:


import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CS_422_Proj_2").config("spark.executor.memory", "1gb").getOrCreate()

sc = spark.sparkContext

data = spark.read.option("delimiter", ";").csv('./winequality-white.csv', header = True, inferSchema=True)

## Separating the data file from the features and the tested quality column ##
feature_columns = data.columns[:-1]

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")

data_2 = assembler.transform(data)

## Splitting data into an 80/20 training, data split ##
train, test = data_2.randomSplit([0.8, 0.2])

from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

## LINEAR REGRESSION MODEL ##

lr = LinearRegression(featuresCol = "features", labelCol = "quality")
lr_model = lr.fit(train)
lr_summary = lr_model.summary

print("Linear Regression RMSE: ", lr_summary.rootMeanSquaredError)

## RANDOM FOREST CLASSIFIER MODEL ##

rf = RandomForestClassifier(featuresCol = "features", labelCol = "quality")
rf_model = rf.fit(train)

prediction = rf_model.transform(data_2).select("quality", "prediction")

evaluatorMulti = MulticlassClassificationEvaluator(labelCol = "quality", predictionCol="prediction")

weightedPrecision = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName:"weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName:"weightedRecall"})

print("Random Forest Classification Precision: ", weightedPrecision)
print("Random Forest Classification Recall: ", weightedRecall)






