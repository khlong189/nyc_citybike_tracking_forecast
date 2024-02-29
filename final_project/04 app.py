# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# DBTITLE 0,YOUR APPLICATIONS CODE HERE...
# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

# print(start_date,end_date,hours_to_forecast, promote_model)

# print("YOUR CODE HERE...")

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Current Timestamp

# COMMAND ----------

import datetime;
spark.conf.set("spark.sql.session.timeZone", "America/New_York")

ct = datetime.datetime.now()
print("Current timestamp:-", ct)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Production Model Version

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()

production_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Production"])[0]
model_uri = f"models:/{GROUP_MODEL_NAME}/{production_model_version.version}"
model = mlflow.prophet.load_model(model_uri)

print(f"Production model version: {production_model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Staging Model Version

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

staging_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Staging"])[0]
model_uri = f"models:/{GROUP_MODEL_NAME}/{staging_model_version.version}"
model = mlflow.prophet.load_model(model_uri)

print(f"Staging model version: {staging_model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Station Name and Map

# COMMAND ----------

# MAGIC %md
# MAGIC ### Station Name: 1 Ave & E 68 St

# COMMAND ----------

!pip install geopandas
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

print(GROUP_STATION_ASSIGNMENT)
df = spark.sql("select * from station_info_bronze").toPandas()
# map
color_scale = [(0, 'orange'), (1,'red')]
fig = px.scatter_mapbox(df, 
                        lat="lat", 
                        lon="lon", 
                        color_continuous_scale=color_scale,
                        zoom=12, 
                        height=800,
                        width=800)
fig.update_traces(marker=dict(size=18))
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Current Weather

# COMMAND ----------

from pyspark.sql.functions import *

weather = spark.read.load(BRONZE_NYC_WEATHER_PATH)
current_time_unix = spark.sql("select to_unix_timestamp(date_trunc('hour', current_timestamp()))").collect()[0][0]

current_weather = (weather.select('dt','temp','pop')
        .filter(weather.dt==current_time_unix)
        .withColumn("Temperature", col('temp'))
        .withColumn('Chance of rain', col('pop'))
        )
display(current_weather.drop('dt','temp','pop'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Station Capacity

# COMMAND ----------

from pyspark.sql.functions import *

station_info_bronze_df = (spark.read.format('delta').load(BRONZE_STATION_INFO_PATH)
                        .filter(col('name') == GROUP_STATION_ASSIGNMENT))
display(station_info_bronze_df.select('capacity'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Total Bikes And Docks Available At The Station

# COMMAND ----------

from pyspark.sql.functions import *

station_id_data = station_info_bronze_df.select(col("station_id")).collect()
station_id = [row["station_id"] for row in station_id_data][0]
station_status_bronze_df = (spark.read.format("delta").load(BRONZE_STATION_STATUS_PATH)
                            .filter(col("station_id") == station_id)
                            .sort(col("last_reported").desc())
                            )
#latest
display(station_status_bronze_df.select('num_docks_available', 'num_bikes_available').limit(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Stream Weather

# COMMAND ----------

stream_weather = spark.read.format('delta').load(f"{GROUP_DATA_PATH}silver/stream_nyc_weather")
stream_weather = stream_weather.withColumn("year", year("dt")).withColumn('month', month('dt')).withColumn('day', dayofmonth('dt')).withColumn('hour', hour('dt'))
stream_weather = stream_weather.orderBy(col('dt').desc())

# COMMAND ----------

stream_weather_df = stream_weather.toPandas()
today = pd.Timestamp.today().normalize()
past_weather_df = stream_weather_df[stream_weather_df['dt'] < today]
future_weather_df = stream_weather_df[stream_weather_df['dt'] >= today]

future_weather_df = future_weather_df[['year', 'month', 'day', 'hour', 'temp', 'feels_like', 'rain_1h']]
future_weather_df['datetime'] = pd.to_datetime(future_weather_df[['year', 'month', 'day', 'hour']])
future_weather_df = future_weather_df.drop(['year', 'month', 'day', 'hour'], axis = 1)
future_weather_df = future_weather_df.rename(columns={'datetime': 'ds'})

baseline_weather_df = future_weather_df[['ds']]

# COMMAND ----------

future_weather_df.head()

# COMMAND ----------

baseline_weather_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Model Prediction for Next 48 Hours

# COMMAND ----------

ARTIFACT_PATH = "G03_model"
model_uri = "models:/G03_model/Production"
baseline_model = mlflow.pyfunc.load_model(model_uri)
baseline_model

# COMMAND ----------

y_pred_baseline = baseline_model.predict(baseline_weather_df)
y_pred_baseline[['ds', 'yhat']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Net bike change prediction plot

# COMMAND ----------

import plotly.graph_objects as pg

fig = pg.Figure()
fig.add_trace(pg.Scatter(x=y_pred_baseline["ds"], y=y_pred_baseline["yhat"], mode='lines'))
fig.update_layout(title='Predicted net bike change for the next 48 hours', xaxis_title='time', yaxis_title='net_bike_change')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuned Model Prediction for Next 48 Hours

# COMMAND ----------

ARTIFACT_PATH = "G03_model"
model_uri = "models:/G03_model/Staging"
tuned_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

y_pred_tuned = tuned_model.predict(future_weather_df)
y_pred_tuned[['ds', 'yhat']]

# COMMAND ----------

import mlflow

def transition(default = False):
    client = mlflow.tracking.MlflowClient()
    if default == True:
        staging_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Staging"])[0]
        model_uri = f"models:/{GROUP_MODEL_NAME}/{staging_model_version.version}"
        model = mlflow.prophet.load_model(model_uri)

        mlflow_client.transition_model_version_stage(
            name=GROUP_MODEL_NAME,
            version=staging_model_version.version,
            stage="Production"
        )
    
        production_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Production"])[0]
        model_uri = f"models:/{GROUP_MODEL_NAME}/{production_model_version.version}"
        model = mlflow.prophet.load_model(model_uri)

        mlflow_client.transition_model_version_stage(
            name=GROUP_MODEL_NAME,
            version=production_model_version.version,
            stage="Archived"
        )
        print("Transitioned")
    print("No changes")
transition()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
