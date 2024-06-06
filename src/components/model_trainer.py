# Basic imports
import os
import sys
from dataclasses import dataclass

# Modeling
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomerException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75],
                    'n_estimators': [50, 100, 200],
                    'criterion': ['squared_error', 'friedman_mse'],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'booster': ['gbtree', 'gblinear', 'dart'],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential'],
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # To get the best model score from the report
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_score'])
            best_model_score = model_report[best_model_name]['test_score']
            best_model = model_report[best_model_name]['model']  # Get the fitted model

            logging.info(f"Best model: {best_model_name} with test score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomerException("No best model found")
            else:
                logging.info("Best model found on both training and testing datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
            raise CustomerException(e, sys)
