import os
from argparse import ArgumentParser

import mlflow
import pandas as pd
import psycopg2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
# docker-compose.yaml에서 설정한 MinIO에 접속하기 위한 아이디/비밀번호
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

def get_data(save_data=False):
    if save_data:
        print("get data from db")
        db_connect = psycopg2.connect(host="localhost", database="mydatabase", user="myuser", password="mypassword")
        df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connect)
        df.to_csv("data.csv", index=False)
    else:
        print("get data from saved data")
        df = pd.read_csv("data.csv") 
    X = df.drop(["id", "timestamp", "target"], axis="columns")
    y = df["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)       
    return X_train, X_valid, y_train, y_valid

def train(X_train, y_train):
    model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    print("training")
    model_pipeline.fit(X_train, y_train)

    return model_pipeline

def prediction(model, X_train, X_valid):
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    return train_pred, valid_pred

def metric(y_train, train_pred, y_valid, valid_pred):
    train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
    valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

    return train_acc, valid_acc

def save_to_mlflow(model_pipeline, model_name, X_train, train_pred, train_acc, valid_acc):
    print("saving model")
    mlflow.set_experiment("new-exp")

    signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
    input_sample = X_train.iloc[:10]

    with mlflow.start_run():
        mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path=model_name,
            signature=signature,
            input_example=input_sample,
        )

def load_from_mlflow(model_name, run_id):
    print("loading model")
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")

def save(model_name):
    X_train, X_valid, y_train, y_valid = get_data()
    model = train(X_train, y_train)
    train_pred, valid_pred = prediction(model, X_train, X_valid)
    train_acc, valid_acc = metric(y_train, train_pred, y_valid, valid_pred)
    print(f'train acc: {train_acc}, valid acc: {valid_acc}')
    save_to_mlflow(model, model_name, X_train, train_pred, train_acc, valid_acc)

def load(model_name, run_id):
    X_train, X_valid, y_train, y_valid = get_data()
    model = load_from_mlflow(model_name, run_id)
    train_pred, valid_pred = prediction(model, X_train, X_valid)
    train_acc, valid_acc = metric(y_train, train_pred, y_valid, valid_pred)
    print(f'train acc: {train_acc}, valid acc: {valid_acc}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', required=True, default='load', choices=['save', 'load'])
    parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
    parser.add_argument("--run-id", dest="run_id", type=str)

    args = parser.parse_args()
    if args.type == 'save':
        save(args.model_name)
    else:
        load(args.model_name, args.run_id)