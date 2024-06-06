import os
import sys
import numpy as np
import pandas as pd
import dill  # this will help in creating pkl file
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomerException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomerException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name} model")
            
            if model_name in params:
                grid_search = GridSearchCV(model, params[model_name], cv=5, scoring='r2')
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Predict on training data
            y_train_pred = best_model.predict(X_train)

            # Predict on test data
            y_test_pred = best_model.predict(X_test)

            # Evaluate train and test dataset
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'model': best_model  # Include the fitted model in the report
            }

        return report

    except Exception as e:
        logging.error(f"Exception occurred during model evaluation: {str(e)}")
        raise CustomerException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)      
    except Exception as e:
        raise CustomerException(e, sys)