import os
import math
import pickle
import mlflow
import pathlib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from prefect import flow, task
from mlflow.models.signature import infer_signature


# leer datos
@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Leer y limpiar los datos de NYC Taxi"""
    df = pd.read_parquet(file_path)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


# task crear features
@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """Crea features para entrenamiento, validación y prueba"""
    for df in [df_train, df_val, df_test]:
        df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()
    X_train = dv.fit_transform(df_train[categorical + numerical].to_dict(orient="records"))
    X_val = dv.transform(df_val[categorical + numerical].to_dict(orient="records"))
    X_test = dv.transform(df_test[categorical + numerical].to_dict(orient="records"))

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    y_test = df_test["duration"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, dv


# task entrenar y registrar modelos
@task(name="Train and Register Models")
def train_and_register_models(X_train, X_val, X_test, y_train, y_val, y_test, dv, experiment_name: str):
    """Entrena dos modelos, compara y registra al mejor"""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name)
    client = mlflow.MlflowClient()

    # Convertir a numpy para evitar error csr_matrix
    train_dataset = mlflow.data.from_numpy(X_train.toarray(), targets=y_train, name="green_tripdata_2025-01")
    validation_dataset = mlflow.data.from_numpy(X_val.toarray(), targets=y_val, name="green_tripdata_2025-02")
    test_dataset = mlflow.data.from_numpy(X_test.toarray(), targets=y_test, name="green_tripdata_2025-03")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    results = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_param("model_type", model_name)

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(validation_dataset, context="validation")

            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

            # Crear input_example y signature
            feature_names = dv.get_feature_names_out()
            input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
            signature = infer_signature(input_example, model.predict(X_val[:5]))

            # Log con firma obligatoria
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="workspace.default.nyc-taxi-model-prefect",
                input_example=input_example,
                signature=signature,
            )

            results[model_name] = {"rmse": rmse, "run_id": run.info.run_id}

    # Elegir mejor modelo (menor RMSE)
    best_model = min(results, key=lambda x: results[x]["rmse"])
    best_run_id = results[best_model]["run_id"]

    print(f"Mejor modelo: {best_model} (RMSE={results[best_model]['rmse']:.4f})")

    # Registrar como @challenger
    model_name = "workspace.default.nyc-taxi-model-prefect"
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_registered_model_alias(model_name, "challenger", mv.version)

    print(f"Modelo '{best_model}' registrado como @challenger (versión {mv.version})")

    return best_model, results, model_name, mv.version


# task comparar con champion
@task(name="Compare and Promote")
def compare_and_promote(results, model_name, challenger_version):
    """Compara el @challenger con el @champion y promueve si mejora"""
    client = mlflow.MlflowClient()

    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion").version
        champion_details = client.get_model_version(model_name, champion_version)
        champion_rmse = float(champion_details.description.split("RMSE=")[-1]) if "RMSE=" in str(champion_details.description) else None
    except Exception:
        print("No existe un @champion actual, se promoverá el @challenger automáticamente.")
        champion_version = None
        champion_rmse = None

    challenger_rmse = None
    for model_name_key, data in results.items():
        if data["rmse"] is not None:
            challenger_rmse = data["rmse"]

    if champion_rmse is None or challenger_rmse < champion_rmse:
        client.set_registered_model_alias(model_name, "champion", challenger_version)
        print(f"Nuevo @champion asignado (versión {challenger_version}, RMSE={challenger_rmse:.4f})")
    else:
        print(f"El @champion actual sigue siendo mejor (RMSE={champion_rmse:.4f} <= {challenger_rmse:.4f})")

# flow principal
@flow(name="nyc-taxi-tarea5-flow")
def tarea5_flow(year: int, month_train: str, month_val: str, month_test: str):
    """Flow que entrena 2 modelos, compara y promueve el mejor"""
    load_dotenv(override=True)

    EXPERIMENT_NAME = "/Users/sarahbeltrang@gmail.com/nyc-taxi-experiment-prefect"
    print(f"Iniciando experimento: {EXPERIMENT_NAME}")

    df_train = read_data(f"../data/green_tripdata_{year}-{month_train}.parquet")
    df_val = read_data(f"../data/green_tripdata_{year}-{month_val}.parquet")
    df_test = read_data(f"../data/green_tripdata_{year}-{month_test}.parquet")

    X_train, X_val, X_test, y_train, y_val, y_test, dv = add_features(df_train, df_val, df_test)
    best_model, results, model_name, challenger_version = train_and_register_models(
        X_train, X_val, X_test, y_train, y_val, y_test, dv, EXPERIMENT_NAME
    )
    compare_and_promote(results, model_name, challenger_version)


if __name__ == "__main__":
    tarea5_flow(year=2025, month_train="01", month_val="02", month_test="03")
