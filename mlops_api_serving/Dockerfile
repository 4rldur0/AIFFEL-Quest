FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install mlflow==1.30.0 pandas scikit-learn uvicorn streamlit "fastapi[all]"

COPY ./app/*.py .
COPY ./app/sk_model ./sk_model

#streamlit 사용하지 않을 시
#CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--reload"]

CMD ["python3", "main.py"]