import os
import sys
import numpy as np
import pandas as pd
import dill           #this will help in creating pkl file

from sklearn.metrics import r2_score
from src.exception import CustomerException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomerException(e, sys)
 

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predict on training data
            y_train_pred = model.predict(X_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Evaluate train and test dataset
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score
            }

        return report

    except Exception as e:
        raise CustomerException(e, sys)
