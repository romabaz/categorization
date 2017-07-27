package com.gl.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import scala.collection.mutable.WrappedArray;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

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
        List<Row> data = new ArrayList<>();


        File file = new File(classLoader.getResource("tickets_input.csv").getFile());

        Dataset<Row> dataset = spark.read()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .csv(file.getAbsolutePath());

        Tokenizer tokenizer = new Tokenizer().setInputCol("DESCRIPTION").setOutputCol("desc_words");
        spark.udf().register("countTokens", (WrappedArray<?> words) -> words.size(), DataTypes.IntegerType);
        Dataset<Row> tokenized = tokenizer.transform(dataset);
        tokenized.select("DESCRIPTION", "desc_words").show(false);

        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("desc_words")
                .setOutputCol("desc_nosw");
        Dataset<Row> swremoved = remover.transform(tokenized);

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("desc_nosw")
                .setOutputCol("vectorized")
                .setVectorSize(3)
                .setMinCount(0);

        Word2VecModel model = word2Vec.fit(swremoved);
        Dataset<Row> vectorized = model.transform(swremoved);
        vectorized.select("DESCRIPTION", "desc_words", "desc_nosw", "vectorized")
                .withColumn("countTokens", callUDF("countTokens", col("desc_nosw")))
                .where("countTokens = 4").show(false);

//
//        Dataset<Row> dataset2 = spark.createDataFrame(
//                dataset.javaRDD().map(row -> RowFactory.create(((String) row.get(4)).split("-"))),
//                dataset.schema());


//        InputStream inputStream = ClassLoader.getSystemResourceAsStream("kickstarter/train_red.csv");
//        Stream<String> stream = new BufferedReader(new InputStreamReader(inputStream)).lines();
//        stream.map(line -> line.split(",(?=([^\"]*\"[^\"]*\")*(?![^\"]*\"))")[4])
//                .forEach(line -> {
//                    data.add(RowFactory.create(Arrays.asList(line.split("-"))));
//                });

//        try {
//            inputStream.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

//        StructType schema = new StructType(new StructField[]{
//                new StructField("text", new ArrayType(StringType, true), false, Metadata.empty())
//        });
//
//        Dataset<Row> documentDF = spark.createDataFrame(data, schema);
//
//        System.out.println(dataset2.columns()[4]);
//        for (Row row : dataset2.takeAsList(10)) {
//            System.out.println(row.getString(4));
//        }

    }

}
