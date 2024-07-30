import os
from argparse import ArgumentParser

import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
# docker-compose.yaml에서 설정한 MinIO에 접속하기 위한 아이디/비밀번호
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

def download_model(args):
    # Download model artifacts
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{args.run_id}/{args.model_name}", dst_path=".")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
    parser.add_argument("--run-id", dest="run_id", type=str)
    args = parser.parse_args()

    download_model(args)