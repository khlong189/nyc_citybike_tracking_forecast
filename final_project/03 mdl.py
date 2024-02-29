# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------


start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)
import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")



# COMMAND ----------

import pandas as pd
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import multiprocessing
from pyspark.sql.functions import *
from mlflow.prophet import load_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Silver Training Data

# COMMAND ----------

# prophet takes in 2 columns which are ds - datetime and y - values
merge_df = spark.read.format("delta").load(f"{GROUP_DATA_PATH}silver/silver_merge")
display(merge_df)

# COMMAND ----------

df = merge_df.toPandas()
df = df[['started_at', 'start_yr', 'start_mnth', 'start_day', 'start_dow', 'start_hr', 'temp', 'feels_like', 'rain_1h', 'net_bike_change']]
cols = ['ds', 'year', 'month', 'day', 'day_of_week', 'hour', 'temp', 'feels_like', 'rain_1h', 'y']
df.columns = cols
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df = df.drop(['ds', 'year', 'month', 'day', 'day_of_week', 'hour'], axis = 1)
df = df.rename(columns={'datetime': 'ds'})
df.head()

# COMMAND ----------

df = df.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline model

# COMMAND ----------

from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
from sklearn.metrics import mean_squared_error

def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

client = mlflow.tracking.MlflowClient()
with mlflow.start_run(run_name="G03_prophet_model_baseline_1") as run:
    run_id = run.info.run_id
    baseline_model = Prophet()
    baseline_model.fit(df)

    df_cv = cross_validation(model=baseline_model, initial= "365 days", horizon="90 days", parallel="threads")
    df_p = performance_metrics(df_cv, rolling_window=1)
    metric_keys = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
    metrics = {k: df_p[k].mean() for k in metric_keys}
    param = extract_params(baseline_model)

    mlflow.prophet.log_model(baseline_model, artifact_path=GROUP_MODEL_NAME)
    mlflow.log_params(param)
    mlflow.log_metrics(metrics)
    model_uri = mlflow.get_artifact_uri(GROUP_MODEL_NAME)

    model_detail = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
    client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=model_detail.version, stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning model

# COMMAND ----------

from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import multiprocessing

ARTIFACT_PATH = "G03_model"

def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

client = mlflow.tracking.MlflowClient()

search_space = {
    'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.05, 0.5),
    'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.1, 50)
}

def objective_function(search_space):
    tuned_model = Prophet(yearly_seasonality=True, 
                             weekly_seasonality=True, 
                             daily_seasonality=True, 
                             changepoint_prior_scale=search_space["changepoint_prior_scale"], 
                             seasonality_prior_scale=search_space["seasonality_prior_scale"],
                             seasonality_mode="additive")

    for col in df.columns:
        if col not in ["ds", "y"]:
            tuned_model.add_regressor(col)

    tuned_model.fit(df)

    df_cv = cross_validation(model=tuned_model, initial = '90 days', period = '30 days',  horizon='30 days', parallel='threads')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmse = df_p["rmse"].mean()
    return {'loss': rmse, 'status': STATUS_OK}


with mlflow.start_run():
    best = fmin(
    fn=objective_function,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials= SparkTrials(parallelism = multiprocessing.cpu_count()))

    best_model = Prophet(growth='linear', changepoint_prior_scale = best['changepoint_prior_scale'], seasonality_prior_scale = best['seasonality_prior_scale'], yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

    for col in df.columns:
        if col not in ["ds", "y"]:
            best_model.add_regressor(col)
    
    best_model.fit(df)

    best_model_cv = cross_validation(model=best_model, initial = '90 days', period = '30 days',  horizon='30 days', parallel="threads")
    best_model_p = performance_metrics(best_model_cv, rolling_window=1)
    metric_keys = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
    metrics = {k: best_model_p[k].mean() for k in metric_keys}
    param = extract_params(best_model)

    mlflow.prophet.log_model(best_model, artifact_path=ARTIFACT_PATH)
    mlflow.log_params(param)
    mlflow.log_metrics(metrics)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

    model_detail = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
    client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=model_detail.version, stage="Staging")

# COMMAND ----------

# checking if everything worked properly
def print_registered_models_info(r_models):
    for rm in r_models:
        print("name: {}".format(rm.name))

model_version_details = client.get_model_version(
  name=GROUP_MODEL_NAME,
  version=model_detail.version 
)

print_registered_models_info(client.search_registered_models())
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------


