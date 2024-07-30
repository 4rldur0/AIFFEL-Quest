### 1. 기본 모델 학습 `iris_train.py`
### 2. pipeline 모델 학습 `pipeline_train.py`
- scaler, classifier 따로 저장하는 1과 달리 pipeline 하나만 저장
### 3. db에서 데이터 가져오기 `db_trin.py`
- load_iris()로 데이터 불러오는 2와 달리 postgreSQL 연결하여 데이터 불러오기
### 4. mlflow를 활용하여 모델 저장 `mlflow_train.py`
- joblib으로 모델 저장하는 3과 달리 구축된 MLflow 서버에 저장
```
$python3 mlflow_train.py -t save --model-name "sk_model"
$python3 mlflow_train.py -t load --model-name "sk_model" --run-id "{Run ID}"
```

#### `docker-compose.yaml` 변경 사항
- 새로운 PostgreSQL DB를 Backend Store(정확도, 하이퍼파라미터 등의 MLflow 메타 데이터가 저장되는 DB)로 사용
- MinIO 서버를 Artifact Store(Model Registry)로 사용
    -  AWS credential을 통해 MinIO 대신 AWS S3를 사용해도 됨
- Backend Store 와 Artifact Store 에 접근 가능한 MLflow 서버를 생성