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
        File file = new File(classLoader.getResource("csv/tickets_input.csv").getFile());

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
                .setOutputCol("features1")
                .setNumFeatures(numFeatures);

        IDF idf = new IDF()
                .setInputCol(hashingTF.getOutputCol())
                .setOutputCol("features");

        StringIndexer stringIndexer1 = new StringIndexer()
                .setInputCol("PRODUCT_CATEGORIZATION_TIER_1")
                .setOutputCol("PCT_1_IDX")
                .setHandleInvalid("skip");
        StringIndexer stringIndexer2 = new StringIndexer()
                .setInputCol("PRODUCT_CATEGORIZATION_TIER_2")
                .setOutputCol("PCT_2_IDX")
                .setHandleInvalid("skip");
        StringIndexer stringIndexer3 = new StringIndexer()
                .setInputCol("PRODUCT_CATEGORIZATION_TIER_3")
                .setOutputCol("PCT_3_IDX")
                .setHandleInvalid("skip");

        Dataset<Row> indexedData = stringIndexer1.fit(trainingData).transform(trainingData);
        indexedData = stringIndexer2.fit(indexedData).transform(indexedData);
        indexedData = stringIndexer3.fit(indexedData).transform(indexedData);

        SQLTransformer sqlTrans = new SQLTransformer().setStatement(
                "SELECT *, 100*PCT_1_IDX + 10*PCT_2_IDX + PCT_3_IDX AS label FROM __THIS__ " +
                        "WHERE PCT_1_IDX is not null AND PCT_2_IDX is not null AND PCT_3_IDX is not null");

        sqlTrans.transform(indexedData).show(false);

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);


        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
                tokenizer,
                remover,
                hashingTF,
                idf,
                stringIndexer1,
                stringIndexer2,
                stringIndexer3,
                sqlTrans,
                lr});

        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);
        for (Row r : predictions.select("DESCRIPTION", "PRODUCT_CATEGORIZATION_TIER_1", "PRODUCT_CATEGORIZATION_TIER_2", "PRODUCT_CATEGORIZATION_TIER_3", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0) + ":"+ r.get(1) + " " + r.get(2) + " " + r.get(3) + ") --> prediction=" + r.get(5));
        }
    }

}
