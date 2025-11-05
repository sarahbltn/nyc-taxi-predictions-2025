import os
import math
import optuna
import pathlib
import pickle
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task


# task leer datos
@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


# task agregar features
@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dv = DictVectorizer()

    X_train = dv.fit_transform(df_train[categorical + numerical].to_dict(orient="records"))
    X_val = dv.transform(df_val[categorical + numerical].to_dict(orient="records"))
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    return X_train, X_val, y_train, y_val, dv


# task optuna hyperparameter tuning
@task(name="Hyperparameter Tuning")
def hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv):
    mlflow.xgboost.autolog()

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    def objective(trial: optuna.trial.Trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-3), 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", math.exp(-5), math.exp(-1), log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", math.exp(-6), math.exp(-1), log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", math.exp(-1), math.exp(3), log=True),
            "objective": "reg:squarederror",
            "seed": 42,
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=10,
            )
            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            signature = infer_signature(X_val, y_pred)
            mlflow.xgboost.log_model(
                booster,
                name="model",
                input_example=X_val[:5],
                signature=signature,
            )
        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name="XGBoost Hyperparameter Optimization", nested=True):
        study.optimize(objective, n_trials=3)

    best_params = study.best_params
    best_params.update({"objective": "reg:squarederror", "seed": 42})
    return best_params


# entrenar mejor modelo
@task(name="Train Best Model")
def train_best_model(X_train, X_val, y_train, y_val, dv, best_params):
    with mlflow.start_run(run_name="Best model ever") as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        mlflow.log_params(best_params)

        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
        })

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=10,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        mlflow.xgboost.log_model(
            booster,
            name="model",
            input_example=input_example,
            signature=signature,
        )

        return run.info.run_id


# registrar modelo con mejor rmse automaticamente
@task(name="Register Best Model")
def register_best_model(experiment_name: str):
    """Encuentra el run con menor RMSE y registra el modelo automáticamente"""
    client = mlflow.MlflowClient()

    # Obtener ID del experimento
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Obtener los runs ordenados por RMSE ascendente
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise Exception("No se encontraron runs en el experimento.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    run_uri = f"runs:/{best_run_id}/model"

    # Registrar modelo y asignar alias
    model_name = "workspace.default.nyc-taxi-model-prefect"
    result = mlflow.register_model(model_uri=run_uri, name=model_name)
    client.set_registered_model_alias(model_name, "champion", result.version)

    print(f"Modelo registrado como '{model_name}' con alias @champion (versión {result.version})")


# flow principal
@flow(name="Main Flow")
def main_flow(year: int, month_train: str, month_val: str):
    """Main Prefect flow with MLflow integration"""
    load_dotenv(override=True)
    EXPERIMENT_NAME = "/Users/sarahbeltrang@gmail.com/nyc-taxi-experiment-prefect"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = read_data(f"../data/green_tripdata_{year}-{month_train}.parquet")
    df_val = read_data(f"../data/green_tripdata_{year}-{month_val}.parquet")

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    run_id = train_best_model(X_train, X_val, y_train, y_val, dv, best_params)
    register_best_model(EXPERIMENT_NAME)

if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02")

