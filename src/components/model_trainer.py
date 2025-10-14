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
                "Logistic Regression":LogisticRegression(),
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "XGB Classifier":XGBClassifier(),
                "CatBoosting Classifier":CatBoostClassifier(verbose=False),
                "AdaBoost Classifier":AdaBoostClassifier()
            }

            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10, 100],
                    # 'solver': ['liblinear', 'lbfgs'],
                    # 'penalty': ['l2', 'l1'],  # note: 'l1' only works with liblinear
                    'max_iter': [100, 200, 500]
                },

                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    # 'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'max_depth': [None, 10, 20, 30],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'bootstrap': [True, False]
                },

                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': [None, 5, 10, 20],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4]
                },

                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.8, 1.0],
                    # 'max_depth': [3, 5, 7]
                },

                "XGB Classifier": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [50, 100, 200],
                    # 'max_depth': [3, 5, 7],
                    # 'subsample': [0.6, 0.8, 1.0],
                    # 'colsample_bytree': [0.6, 0.8, 1.0]
                },

                "CatBoosting Classifier": {
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200],
                    # 'l2_leaf_reg': [1, 3, 5]
                },

                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.5, 1.0]
                }
            }


            model_report:dict = evaluate_classification_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

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
