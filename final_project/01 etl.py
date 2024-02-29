# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %md 
# MAGIC # Bronze Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical bike trip

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
his_bike_trip_schema = "ride_id STRING, rideable_type STRING, started_at TIMESTAMP, ended_at TIMESTAMP, start_station_name STRING, start_station_id STRING, end_station_name STRING, end_station_id STRING, start_lat DOUBLE, start_lng DOUBLE, end_lat DOUBLE, end_lng DOUBLE, member_casual STRING"

his_bike_trip_checkpoint = f"{GROUP_DATA_PATH}bronze/his_bike_trip/.checkpoint"

query = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format" , "csv")
    .option("cloudFiles.schemaHints", his_bike_trip_schema)
    .option("cloudFiles.schemaLocation", his_bike_trip_checkpoint)
    .option("header", "True")
    .load(BIKE_TRIP_DATA_PATH)
    .filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT))
    .withColumn("started_at", col("started_at").cast(TimestampType()))
    .withColumn("start_mnth", month(col("started_at")))
    .withColumn("start_dow", dayofweek(col("started_at")))
    .writeStream
    .format("delta")
    .option("checkpointLocation", his_bike_trip_checkpoint)
    .partitionBy("start_mnth","start_dow")
    .outputMode("append")
    .option("path", f"{GROUP_DATA_PATH}bronze/his_bike_trip")
    .trigger(once=True)
    .start())

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/G03/bronze/his_bike_trip/")

# COMMAND ----------

spark.read.format("delta").load(f"{GROUP_DATA_PATH}bronze/his_bike_trip").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical weather

# COMMAND ----------

from pyspark.sql.functions import from_unixtime
from pyspark.sql.types import *

his_weather_schema = "dt INT, temp DOUBLE, feels_like DOUBLE, pressure INT, humidity INT, dew_point DOUBLE, uvi DOUBLE, clouds INT, visibility INT, wind_speed DOUBLE, wind_deg INT, pop DOUBLE, snow_1h DOUBLE, id INT, main STRING, description STRING, icon STRING, loc STRING, lat DOUBLE, lon DOUBLE, timezone STRING, timezone_offset INT, rain_1h DOUBLE"

his_weather_checkpoint = f"{GROUP_DATA_PATH}bronze/his_weather/.checkpoint"

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format" , "csv")
    .option("cloudFiles.schemaHints", his_weather_schema)
    .option("cloudFiles.schemaLocation", his_weather_checkpoint)
    .option("header", "True")
    .load(NYC_WEATHER_FILE_PATH)
    .withColumn("dt", col("dt").cast(TimestampType()))
    .withColumn("mnth", month(col("dt")))
    .withColumn("dow", dayofweek(col("dt")))
    .writeStream
    .format("delta")
    .option("checkpointLocation", his_weather_checkpoint)
    .outputMode("append")
    .partitionBy("mnth","dow")
    .option("path", f"{GROUP_DATA_PATH}bronze/his_weather")
    .trigger(once=True)
    .start())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming station info

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

station_info_bronze_df = spark.read.format("delta").load(BRONZE_STATION_INFO_PATH).filter(col("name") == GROUP_STATION_ASSIGNMENT)
display(station_info_bronze_df)
station_info_bronze_df.write.format("delta").mode("overwrite").save(f"{GROUP_DATA_PATH}bronze/stream_station_info")
station_info_bronze_df.write.format("delta").mode("overwrite").saveAsTable("station_info_bronze")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Streaming station status

# COMMAND ----------

station_id_data = station_info_bronze_df.select(col("station_id")).collect()
station_id = [row["station_id"] for row in station_id_data][0]
station_status_bronze_df = spark.read.format("delta").load(BRONZE_STATION_STATUS_PATH).filter(col("station_id") == station_id)
display(station_status_bronze_df)
station_status_bronze_df.write.format("delta").mode("overwrite").save(f"{GROUP_DATA_PATH}bronze/stream_station_status")
station_status_bronze_df.write.format("delta").mode("overwrite").saveAsTable("station_status_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming weather

# COMMAND ----------

nyc_weather_bronze_df = (spark.read.format("delta").load(BRONZE_NYC_WEATHER_PATH)
                            .withColumn("dt", from_unixtime(col("dt")).cast(TimestampType()))
                            .withColumn("weather", explode(col("weather")))
)
nyc_weather_bronze_df = (nyc_weather_bronze_df
                            .withColumn("description", col("weather.description"))
                            .withColumn("icon", col("weather.icon"))
                            .withColumn("id", col("weather.id"))
                            .withColumn("main", col("weather.main"))
                            .drop(col("weather")))
display(nyc_weather_bronze_df)
nyc_weather_bronze_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}bronze/stream_nyc_weather")
nyc_weather_bronze_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable("nyc_weather_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical bike trip

# COMMAND ----------

his_biketrip_df = spark.read.format('delta').load(f"{GROUP_DATA_PATH}bronze/his_bike_trip")

# COMMAND ----------

his_biketrip_df.createOrReplaceTempView("his_biketrip_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW silver_his_bike_trip AS
# MAGIC SELECT ride_id, started_at, DATE(started_at) as start_ts, EXTRACT(year FROM started_at) as start_yr, EXTRACT(month FROM started_at) as start_mnth, EXTRACT(day FROM started_at) as start_day, DAYOFWEEK(started_at) as start_dow, EXTRACT(hour from started_at) as start_hr, ended_at, DATE(ended_at) as end_ts, EXTRACT(year FROM ended_at) as end_yr, EXTRACT(month FROM ended_at) as end_mnth, EXTRACT(day FROM ended_at) as end_day, DAYOFWEEK(ended_at) as end_dow, EXTRACT(hour from ended_at) as end_hr, rideable_type, start_station_name, end_station_name, member_casual 
# MAGIC from his_biketrip_temp;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- compute net bike change
# MAGIC create or replace temp view netchange as
# MAGIC with start as
# MAGIC (
# MAGIC --compute hourly trip count that starts at assigned station 
# MAGIC SELECT
# MAGIC start_ts, start_hr,
# MAGIC COUNT(ride_id) as ride_count_out
# MAGIC FROM silver_his_bike_trip 
# MAGIC WHERE start_station_name = '1 Ave & E 68 St'
# MAGIC GROUP BY 1, 2
# MAGIC ),
# MAGIC
# MAGIC end as(
# MAGIC --compute hourly trip count that ends at assigned station 
# MAGIC SELECT
# MAGIC end_ts, end_hr,
# MAGIC COUNT(ride_id) as ride_count_in
# MAGIC FROM silver_his_bike_trip 
# MAGIC WHERE end_station_name = '1 Ave & E 68 St'
# MAGIC GROUP BY 1, 2 
# MAGIC ),
# MAGIC
# MAGIC net as --compute the difference between the 2 hourly trip count values above 
# MAGIC (select start.start_ts, start.start_hr, end.end_ts, end.end_hr, ride_count_in, ride_count_out, end.ride_count_in - start.ride_count_out as net_bike_change
# MAGIC from start 
# MAGIC join end 
# MAGIC on start.start_ts = end.end_ts and start.start_hr = end.end_hr
# MAGIC ORDER BY 1, 2)
# MAGIC
# MAGIC select s.*, ride_count_in, ride_count_out, net_bike_change
# MAGIC from silver_his_bike_trip s
# MAGIC join net
# MAGIC on s.start_ts = net.start_ts and s.start_hr = net.start_hr
# MAGIC ORDER BY s.started_at;

# COMMAND ----------

his_biketrip_df = spark.read.table("netchange")
his_biketrip_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}silver/his_bike_trip")


# COMMAND ----------

# dbutils.fs.ls(f"{GROUP_DATA_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Historical weather

# COMMAND ----------

his_weather_df = spark.read.format('delta').load(f"{GROUP_DATA_PATH}bronze/his_weather")

# COMMAND ----------

from pyspark.sql.functions import *

his_weather_df = (his_weather_df.withColumn("temp_f", (col("temp") - 273.15) * 9/5 + 32) 
                                .withColumn("feels_like_f", (col("feels_like") - 273.15) * 9/5 + 32)
                                .withColumn("dew_point_f", (col("dew_point") - 273.15) * 9/5 + 32)
                                .withColumn("date", to_timestamp(from_unixtime("dt")))
                                .drop("temp", "feels_like","dew_point","dt","snow_1h") 
                                .withColumnRenamed("temp_f", "temp") 
                                .withColumnRenamed("feels_like_f", "feels_like")
                                .withColumnRenamed("dew_point_f", "dew_point"))
his_weather_df.drop("_rescued_data")

# COMMAND ----------

his_weather_df.display()

# COMMAND ----------

his_weather_df.createOrReplaceTempView("his_weather_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW silver_his_weather as
# MAGIC SELECT DATE(date) as ts, EXTRACT(year FROM date) as yr, EXTRACT(month FROM date) as mnth, EXTRACT(day FROM date) as days, DAYOFWEEK(date) as day_of_week, EXTRACT(hour from date) as hr, temp, feels_like, pressure, humidity, dew_point, wind_speed, main, description, rain_1h from his_weather_temp

# COMMAND ----------

his_weather_df = spark.read.table("silver_his_weather")
his_weather_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}silver/his_weather")


# COMMAND ----------

dbutils.fs.ls(f"{GROUP_DATA_PATH}bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming station info

# COMMAND ----------

station_info = spark.read.format('delta').load(f"{GROUP_DATA_PATH}bronze/stream_station_info")

# COMMAND ----------

silver_station_info = station_info.select(
                        "capacity",
                        "lat",
                        "lon")
silver_station_info.display()

# COMMAND ----------

silver_station_info.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}silver/stream_station_info")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming station status

# COMMAND ----------

station_status = spark.read.format('delta').load(f"{GROUP_DATA_PATH}bronze/stream_station_status")

# COMMAND ----------

silver_station_status = station_status.select(
                    'last_reported',
                    'num_bikes_disabled',
                    'num_bikes_available',
                    'num_docks_disabled',
                    'num_docks_available'
                    )
silver_station_status.display()

# COMMAND ----------

silver_station_status = (silver_station_status 
                        .withColumn("last_reported_ts", to_timestamp(from_unixtime("last_reported"))) 
                        .withColumn("year", year("last_reported_ts")) 
                        .withColumn("month", month("last_reported_ts")) 
                        .withColumn("day", dayofmonth("last_reported_ts")) 
                        .withColumn("dow", dayofweek("last_reported_ts")) 
                        .withColumn("hour", hour("last_reported_ts")) 
                        .drop("last_reported"))

# COMMAND ----------

silver_station_status = silver_station_status.select(
                    'last_reported_ts',
                    'year',
                    'month',
                    'day',
                    'dow',
                    'hour',
                    'num_bikes_disabled',
                    'num_bikes_available',
                    'num_docks_disabled',
                    'num_docks_available'
                    )
silver_station_status.display()

# COMMAND ----------

silver_station_status.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}silver/stream_station_status")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming weather

# COMMAND ----------

stream_weather = spark.read.format('delta').load(f"{GROUP_DATA_PATH}bronze/stream_nyc_weather")
stream_weather.display()

# COMMAND ----------


silver_stream_weather_df = (stream_weather.withColumn("temp_f", (col("temp") - 273.15) * 9/5 + 32) 
                                .withColumn("feels_like_f", (col("feels_like") - 273.15) * 9/5 + 32)
                                .withColumn("dew_point_f", (col("dew_point") - 273.15) * 9/5 + 32)
                                .drop("temp", "feels_like","dew_point","time","uvi","clouds","visibility","pop","wind_deg","wind_gust","icon", "id") 
                                .withColumnRenamed("temp_f", "temp") 
                                .withColumnRenamed("feels_like_f", "feels_like")
                                .withColumnRenamed("dew_point_f", "dew_point")
                                .withColumnRenamed("rain.1h", "rain_1h"))

silver_stream_weather_df = silver_stream_weather_df.fillna(0)

# COMMAND ----------

silver_stream_weather_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}silver/stream_nyc_weather")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Merge `Silver bike trip` and `Silver weather`

# COMMAND ----------

silver_biketrip_df = spark.read.format('delta').load(f"{GROUP_DATA_PATH}silver/his_bike_trip")
silver_weather_df = spark.read.format('delta').load(f"{GROUP_DATA_PATH}silver/his_weather")

# COMMAND ----------

silver_biketrip_df.createOrReplaceTempView("sil_biketrip_temp")
silver_weather_df.createOrReplaceTempView("sil_weather_temp")

# COMMAND ----------

# merge history biketrip table and weather table into a new table called biketrip_weather_merge 
spark.sql("CREATE OR REPLACE TEMPORARY VIEW biketrip_weather_merge AS SELECT started_at, start_yr, start_mnth, start_day, start_dow, start_hr, b.rideable_type,b.start_station_name, b.end_station_name, b.member_casual, w.temp, w.feels_like, w.pressure, w.humidity, w.dew_point, w.wind_speed, w.main, w.description, w.rain_1h, b.ride_count_in, b.ride_count_out,b.net_bike_change from sil_biketrip_temp b join sil_weather_temp w on b.start_yr = w.yr and b.start_mnth = w.mnth and b.start_day = w.days and b.start_hr = hr")

# COMMAND ----------

silver_merge_df = spark.read.table("biketrip_weather_merge")

# COMMAND ----------

silver_merge_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(f"{GROUP_DATA_PATH}silver/silver_merge")

# COMMAND ----------

spark.read.format('delta').load(f"{GROUP_DATA_PATH}silver/silver_merge").count()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


