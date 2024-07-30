from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib
import wandb

def set_model(config):
    model = None
    if config.model_name == "svc":
        model = SVC()
    elif config.model_name == "lr":
        model = LogisticRegression()
    elif config.model_name == "sgd":
        model = SGDClassifier()
    elif config.model_name == "dt":
        model = DecisionTreeClassifier()
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")

    model_pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model_pipeline

def wandb_train_save(run, config, X_train, X_valid, y_train, y_valid):
    model = set_model(config)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
    valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

    metrics = {'train_acc': train_acc, 'valid_acc': valid_acc}
    wandb.log(metrics)

    cm = confusion_matrix(y_valid, valid_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={'size': 30})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix({config.model_name})')
    wandb.log({"Confusion Matrix": wandb.Image(plt)})

    print("Saving model")
    artifact = wandb.Artifact(
        'classifier_model', type='model', description=config.model_name,
        metadata={"parameters": vars(config), "metrics": metrics}
    )

    model_path = f"models/wandb_{config.model_name}.joblib"
    joblib.dump(model, model_path)
    artifact.add_file(model_path)

    # Save the artifact
    run.log_artifact(artifact)

def train():
    print("Get data from saved data")
    df = pd.read_csv("data.csv")
    X = df.drop(["id", "timestamp", "target"], axis="columns")
    y = df["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

    run = wandb.init()
    config = wandb.config
    wandb_train_save(run, config, X_train, X_valid, y_train, y_valid)

    wandb.finish()

def wandb_tuning():
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'valid_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'model_name': {
                'values': ['svc', 'lr', 'sgd', 'dt']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config,
                           entity='4-rldur0',
                           project='iris_classifier')

    wandb.agent(sweep_id, function=train, count=4)


def load_from_wandb():
    run = wandb.init(project="iris_classifier", job_type="evaluate")

    # Download the model artifact
    artifact = run.use_artifact('classifier_model:latest', type='model')
    model_path = artifact.download()
    print(model_path)
    return joblib.load(model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', default='load', choices=['save', 'load'])

    args = parser.parse_args()
    if args.type == 'save':
        wandb_tuning()
    else:
        load_from_wandb()