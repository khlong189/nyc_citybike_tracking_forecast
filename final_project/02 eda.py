# Databricks notebook source
# MAGIC %run ./includes/includes 

# COMMAND ----------

# start_date = str(dbutils.widgets.get('01.start_date')) 
# end_date = str(dbutils.widgets.get('02.end_date'))
# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

# print(start_date,end_date,hours_to_forecast, promote_model)
# print("YOUR CODE HERE...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Historical Bike Trip and Historical Weather

# COMMAND ----------

his_bike_trip = spark.read.format("delta").load(f"{GROUP_DATA_PATH}bronze/his_bike_trip")
his_bike_trip.createOrReplaceTempView("his_bike_trip") #create a temp view of bronze bike trip

from pyspark.sql.functions import *
his_weather = spark.read.format("delta").load(f"{GROUP_DATA_PATH}bronze/his_weather")
his_weather = his_weather.withColumn("datetime", to_timestamp(from_unixtime("dt"))).drop('dt')
his_weather.createOrReplaceTempView("his_weather") #create a temp view of bronze weather

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total Daily Trips

# COMMAND ----------

# MAGIC %sql
# MAGIC --count number of daily trips from 2021 to 2023
# MAGIC SELECT DATE(started_at) start_ts,
# MAGIC count(*) AS records FROM his_bike_trip
# MAGIC group by start_ts
# MAGIC order by start_ts; 

# COMMAND ----------

# MAGIC %md
# MAGIC The highest daily trip count from November 2021 to March 2023 fluctuates between 0 and 1000. The highest daily trip count recorded is 1027 on September 20, 2022 and the lowest count recorded is 3 on January 29, 2022. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total Monthly Trips

# COMMAND ----------

# MAGIC %sql
# MAGIC -- count number of trips by month
# MAGIC SELECT month(started_at) AS months,
# MAGIC count(*) AS records FROM his_bike_trip
# MAGIC GROUP BY 1
# MAGIC order by 1;

# COMMAND ----------

# MAGIC %md
# MAGIC The two months with the highest total monthly trip count are November and March with 32727 and 31496 trips, respectively. April has the lowest total number of trips of 16328.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total Monthly Trips By Each Year

# COMMAND ----------

# MAGIC %sql
# MAGIC -- count number of trips by month and year
# MAGIC SELECT date_format(started_at, 'yyyy-MM') AS months,
# MAGIC count(*) AS records FROM his_bike_trip
# MAGIC group by 1
# MAGIC order by 1;

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the data was recorded in 2022. In 2022, the total trip count increased from January and peaked in August then decreased pretty quickly and hit the bottom in December. The trip count decreased from November to December 2021, and the trip count started to increase from December 2022 to March 2023.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total trips by day of the week

# COMMAND ----------

# MAGIC %sql
# MAGIC -- count number of trips by day of the week
# MAGIC SELECT dayofweek(started_at) AS day_of_week, --date_format(started_at, 'yyyy-MM') AS months,
# MAGIC count(*) AS records FROM his_bike_trip
# MAGIC group by 1
# MAGIC order by 1;

# COMMAND ----------

# MAGIC %md
# MAGIC There is a huge difference in the total trip count between the weekdays and the weekends. The total trip count on the weekend is around 20000 on average and the total trip count on each of the weekdays is at least twice as much.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total trips by hour

# COMMAND ----------

# MAGIC %sql
# MAGIC -- count number of trips by hour
# MAGIC SELECT hour(started_at) as hour,
# MAGIC count(*) AS records FROM his_bike_trip
# MAGIC group by 1
# MAGIC order by 1;

# COMMAND ----------

# MAGIC %md
# MAGIC The highest total trip counts are between 6:00 AM and 9:00 AM and between 4:00 PM and 6:00 PM.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total trips by hour and day of the Week

# COMMAND ----------

# MAGIC %sql
# MAGIC -- count number of trips by hour and by day of week
# MAGIC SELECT dayofweek(started_at) day_of_week, hour(started_at) as hour,
# MAGIC count(*) AS records FROM his_bike_trip
# MAGIC group by 1, 2
# MAGIC order by 1, 2;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily Trip Trend vs Holiday 

# COMMAND ----------

#create a temp view that contains all holiday dates from 2021 to 2023

import holidays
import datetime as dt

holiday_dates = holidays.US(years=[2021, 2022, 2023])

holiday_list = []
for holiday in holiday_dates.items():
    holiday_list.append(holiday)

holiday_df = pd.DataFrame(holiday_list, columns=["date", "holiday"])
spark_df = spark.createDataFrame(holiday_df)
spark_df.createOrReplaceTempView("holidays") 


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from holidays; -- show holiday temp view 

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- join bronze bike trip temp view with holiday temp view 
# MAGIC WITH 
# MAGIC bike_trip AS (
# MAGIC     SELECT date(started_at) start_date, extract(year FROM started_at) year, ride_id FROM his_bike_trip
# MAGIC ),
# MAGIC
# MAGIC bike_holiday_merge as
# MAGIC (
# MAGIC     SELECT *, if(h.holiday is null, 0, 1) as is_holiday
# MAGIC     FROM bike_trip bt LEFT JOIN holidays h on bt.start_date = h.date
# MAGIC )
# MAGIC
# MAGIC SELECT start_date, year, date, ride_id, is_holiday FROM bike_holiday_merge
# MAGIC ORDER BY 1;

# COMMAND ----------

# query to join bronze bike trip temp view with holiday temp view 
query = """
    WITH 
bike_trip AS (
    SELECT date(started_at) start_date, extract(year FROM started_at) year, ride_id FROM his_bike_trip
),

bike_holiday_merge as
(
    SELECT *, if(h.holiday is null, 0, 1) as is_holiday
    FROM bike_trip bt LEFT JOIN holidays h on bt.start_date = h.date
)

SELECT start_date, year, date, ride_id, is_holiday FROM bike_holiday_merge
ORDER BY 1;
"""

#transform the joined table into Pandas dataframe 
df = spark.sql(query).toPandas()

#fill na values in the dt column with 0
df['date'] = df['date'].fillna(0)

#select dates that are holidays 
holiday_dates = [val for val in df['date'] if val != 0]

# COMMAND ----------

 import matplotlib.pyplot as plt

# Load the bike usage data
# Preprocess the data as needed

# Group the data by date
daily_usage = df.groupby('start_date')['ride_id'].count()

# Create a line chart of the daily bike usage trend
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(daily_usage.index, daily_usage.values, label='Daily Bike Usage')

# Add x-axis and y-axis labels and a title
ax.set_xlabel('Date')
ax.set_ylabel('Number of Bikes')
ax.set_title('Daily Bike Usage Trend')

#mark the dates that are holidays on the graph
for holiday_date in holiday_dates:
    ax.axvline(x=holiday_date, color='red', linestyle='--', linewidth=1) 

# Add a legend
ax.legend()

# Show the chart
plt.show()


# COMMAND ----------

import seaborn as sns

gb = df.groupby(['year', 'is_holiday']).count()[['ride_id']]
gb = gb.reset_index()

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(data = gb, x = 'year', y = 'ride_id', hue = 'is_holiday')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There's a drastic difference in the daily trip count between holiday dates and non-holiday dates. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Daily Trend vs Weather

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMP VIEW bike_weather AS
# MAGIC
# MAGIC WITH bike_trip AS (
# MAGIC   SELECT date(started_at) d, hour(started_at) h, count(ride_id) ride_count 
# MAGIC   FROM his_bike_trip 
# MAGIC   GROUP BY 1, 2
# MAGIC ),
# MAGIC
# MAGIC weather AS (
# MAGIC   SELECT date(datetime) date, hour(datetime) hour, (temp - 273.15) * 9/5 + 32 temp, 
# MAGIC   (feels_like - 273.15) * 9/5 + 32 feels_like, pressure, humidity, (dew_point - 273.15) * 9/5 + 32 dew_point, 
# MAGIC   uvi, clouds, visibility, wind_speed,
# MAGIC   snow_1h, rain_1h
# MAGIC   FROM his_weather
# MAGIC )
# MAGIC
# MAGIC SELECT w.*, bt.ride_count from bike_trip bt JOIN weather w ON bt.d = w.date AND bt.h = w.hour
# MAGIC ORDER BY 1, 2; 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bike_weather;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW daily_bike_weather AS 
# MAGIC SELECT date, sum(ride_count) daily_ride_count,
# MAGIC avg(temp) as avg_temp, avg(feels_like) avg_feels_like,
# MAGIC avg(pressure) avg_pressure, avg(humidity) avg_humidity, avg(dew_point) avg_dew_point, avg(uvi) avg_uvi, 
# MAGIC avg(clouds) avg_clouds, avg(wind_speed) avg_wind_speed, avg(snow_1h) avg_snow_1h, avg(rain_1h) avg_rain_1h
# MAGIC FROM bike_weather
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM daily_bike_weather;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average temperature of that day
# MAGIC SELECT date, daily_ride_count, avg_temp
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average feels_like of that day
# MAGIC SELECT date, daily_ride_count, avg_feels_like
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average snow_1h of that day
# MAGIC SELECT date, daily_ride_count, avg_snow_1h
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average rain_1h of that day
# MAGIC SELECT date, daily_ride_count, avg_rain_1h
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average humidity of that day
# MAGIC SELECT date, daily_ride_count, avg_humidity
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average dew_point of that day
# MAGIC SELECT date, daily_ride_count, avg_dew_point
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average wind_speed of that day
# MAGIC SELECT date, daily_ride_count, avg_wind_speed
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average pressure of that day
# MAGIC SELECT date, daily_ride_count, avg_pressure
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average uvi of that day
# MAGIC SELECT date, daily_ride_count, avg_uvi
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC --plot daily trip count vs average clouds of that day
# MAGIC SELECT date, daily_ride_count, avg_clouds
# MAGIC FROM daily_bike_weather
# MAGIC ORDER BY 1;

# COMMAND ----------

# MAGIC %md
# MAGIC The weather features that are positively correlated with the daily trip count are temp, feels_like. The rain_1h feature is negatively correlated with the daily trip count.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Hourly trend vs weather

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly temperature 
# MAGIC SELECT date, hour, ride_count, temp
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly feels_like
# MAGIC SELECT date, hour, ride_count, feels_like
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly rain_1h
# MAGIC SELECT date, hour, ride_count, rain_1h
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly snow_1h 
# MAGIC SELECT date, hour, ride_count, snow_1h
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly dew_point 
# MAGIC SELECT date, hour, ride_count, dew_point
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly humidity 
# MAGIC SELECT date, hour, ride_count, humidity
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly wind_speed 
# MAGIC SELECT date, hour, ride_count, wind_speed
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly pressure
# MAGIC SELECT date, hour, ride_count, pressure
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly uvi 
# MAGIC SELECT date, hour, ride_count, uvi
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %sql 
# MAGIC --plot hourly trip count vs hourly clouds
# MAGIC SELECT date, hour, ride_count, clouds
# MAGIC FROM bike_weather
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# MAGIC %md
# MAGIC The only weather features that are correlated with the hourly trip counts are rain_1h and snow_1h. 

# COMMAND ----------


