import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.exception import CustomException
import dill

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_classification_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            yhat_train = model.predict(X_train)
            yhat_test = model.predict(X_test)

            acc_train = accuracy_score(y_train, yhat_train)
            acc_test = accuracy_score(y_test, yhat_test)

            prec = precision_score(y_test, yhat_test, average='weighted', zero_division=0)
            rec = recall_score(y_test, yhat_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, yhat_test, average='weighted', zero_division=0)

            report[model_name] = {
                "Train Accuracy": acc_train,
                "Test Accuracy": acc_test,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
