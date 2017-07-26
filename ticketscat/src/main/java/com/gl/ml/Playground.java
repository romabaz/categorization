package com.gl.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by roman.loyko on 26-Jul-17.
 */
public class Playground {
    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "c:\\hadoop\\");
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

        Dataset<Row> dataset = spark.read()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .csv("src/main/resources/kickstarter/train_red.csv");


        Dataset<Row> dataset2 = spark.createDataFrame(
                dataset.javaRDD().map(row -> RowFactory.create(((String) row.get(4)).split("-"))),
                dataset.schema());


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

        System.out.println(dataset2.columns()[4]);
        for (Row row : dataset2.takeAsList(10)) {
            System.out.println(row.getString(4));
        }

    }
}
