package com.gl.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.collection.mutable.WrappedArray;

import java.io.File;
import java.util.Arrays;
import java.util.List;


/**
 * Created by roman.loyko on 26-Jul-17.
 */
public class Playground {
    public static void main(String[] args) {
        ClassLoader classLoader = Playground.class.getClassLoader();
        System.setProperty("hadoop.home.dir", classLoader.getResource("hadoop").getFile());
        SparkSession spark = SparkSession
                .builder()
                .appName("SampleFeatures")
                .config(new SparkConf()
                        .set("spark.shuffle.service.enabled", "false")
                        .set("spark.dynamicAllocation.enabled", "false")
                        .set("spark.cores.max", "1")
                        .set("spark.executor.instances", "1")
                        .set("spark.executor.memory", "471859200")
                        .set("spark.executor.cores", "1"))
//                .master("spark://46.101.158.118:7077")
                .master("local[2]")
                .getOrCreate();

        //1. feature extraction
        File file = new File(classLoader.getResource("tickets_input.csv").getFile());

        int numFeatures = 100;

        Dataset<Row> dataset = spark.read()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .csv(file.getAbsolutePath());

        Dataset<Row>[] splits = dataset.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("DESCRIPTION")
                .setOutputCol("desc_words");

        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("desc_nosw");

        HashingTF hashingTF = new HashingTF()
                .setInputCol(remover.getOutputCol())
                .setOutputCol("features")
                .setNumFeatures(numFeatures);

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(20)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);


        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {tokenizer, remover, hashingTF, lr});

        PipelineModel model = pipeline.fit(trainingData);

        model.transform(testData);

        Dataset<Row> predictions = model.transform(testData);
        for (Row r : predictions.select("id", "DESCRIPTION", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }

        /* ==================================
        Example from https://spark.apache.org/docs/latest/ml-features.html#tf-idf
         */

//        List<Row> dataEx = Arrays.asList(
//                RowFactory.create(0.0, "Hi I heard about Spark"),
//                RowFactory.create(0.0, "I wish Java could use case classes"),
//                RowFactory.create(2.0, "Logistic regression models are neat")
//        );
//        StructType schema = new StructType(new StructField[]{
//                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
//                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
//        });
//        Dataset<Row> sentenceData = spark.createDataFrame(dataEx, schema);
//
//        Tokenizer tokenizerEx = new Tokenizer().setInputCol("sentence").setOutputCol("words");
//        Dataset<Row> wordsData = tokenizerEx.transform(sentenceData);
//
//        HashingTF hashingTFEx = new HashingTF()
//                .setInputCol("words")
//                .setOutputCol("rawFeatures")
//                .setNumFeatures(numFeatures);
//
//        Dataset<Row> featurizedDataEx = hashingTFEx.transform(wordsData);
//
//        IDF idfEx = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//        IDFModel idfModelEx = idfEx.fit(featurizedDataEx);
//
//        Dataset<Row> rescaledDataEx = idfModelEx.transform(featurizedDataEx);
//        rescaledDataEx.select("label", "words", "rawFeatures", "features").show(false);

        /* ============
        End of example
         */


//        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//        IDFModel idfModel = idf.fit(featurizedData);
//        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
//
//        Word2Vec word2Vec = new Word2Vec()
//                .setInputCol("desc_nosw")
//                .setOutputCol("vectorized")
//                .setVectorSize(3)
//                .setMinCount(0);
//
//        Word2VecModel model = word2Vec.fit(swremoved);
//        Dataset<Row> vectorized = model.transform(swremoved);
//        rescaledData.select("DESCRIPTION", "desc_words", "desc_nosw", "rawFeatures", "features").show(false);
//                .withColumn("countTokens", callUDF("countTokens", col("desc_nosw")))
//                .where("countTokens = 4").show(false);

    }

}
