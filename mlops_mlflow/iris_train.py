from argparse import ArgumentParser

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import joblib



def get_data():
    print("preparing data")
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)
    
    return X_train, X_valid, y_train, y_valid

def scale_data(X_train, X_valid, scaler=None):
    if scaler==None:
        scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_valid = scaler.transform(X_valid)

    return scaler, scaled_X_train, scaled_X_valid

def train(scaled_X_train, y_train):
    print("training")
    classifier = SVC()
    classifier.fit(scaled_X_train, y_train)

    return classifier

def prediction(classifier, scaled_X_train, scaled_X_valid):
    train_pred = classifier.predict(scaled_X_train)
    valid_pred = classifier.predict(scaled_X_valid)

    return train_pred, valid_pred

def metric(y_train, train_pred, y_valid, valid_pred):
    train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
    valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

    return train_acc, valid_acc

def save_model(scaler, classifier):
    print("saving model")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(classifier, "models/classifier.joblib")

def load_model():
    print("loading model")
    scaler_load = joblib.load("models/scaler.joblib")
    classifier_load = joblib.load("models/classifier.joblib")

    return scaler_load, classifier_load

def save():
    X_train, X_valid, y_train, y_valid = get_data()
    scaler, scaled_X_train, scaled_X_valid = scale_data(X_train, X_valid)
    classifier = train(scaled_X_train, y_train)
    train_pred, valid_pred = prediction(classifier, scaled_X_train, scaled_X_valid)
    train_acc, valid_acc = metric(y_train, train_pred, y_valid, valid_pred)
    print(f'train acc: {train_acc}, valid acc: {valid_acc}')
    save_model(scaler, classifier)

def load():
    X_train, X_valid, y_train, y_valid = get_data()
    scaler_load, classifier_load = load_model()
    scaler_load, scaled_X_train, scaled_X_valid = scale_data(X_train, X_valid, scaler_load)
    train_pred, valid_pred = prediction(classifier_load, scaled_X_train, scaled_X_valid)
    train_acc, valid_acc = metric(y_train, train_pred, y_valid, valid_pred)
    print(f'train acc: {train_acc}, valid acc: {valid_acc}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--type', default='load', choices=['save', 'load', ])

    args = parser.parse_args()
    if args.type == 'save':
        save()
    else:
        load()