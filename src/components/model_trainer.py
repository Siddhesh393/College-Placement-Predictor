import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_classification_models

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Split training and testing data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Linear Regression":LogisticRegression(),
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "XGB Classifier":XGBClassifier(),
                "CatBoosting Classifier":CatBoostClassifier(verbose=False),
                "AdaBoost Classifier":AdaBoostClassifier()
            }

            model_report:dict = evaluate_classification_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            f1_scores = {model: metrics["F1 Score"] for model, metrics in model_report.items()}
            best_model_name = max(f1_scores, key=f1_scores.get)
            best_model_score = f1_scores[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            yhat_test=best_model.predict(X_test)
            f1 = f1_score(y_test, yhat_test, average='weighted', zero_division=0)

            return f1

        except Exception as e:
            raise CustomException(e,sys)
