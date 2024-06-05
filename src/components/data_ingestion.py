import os
import sys
from src.exception import CustomerException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        
        try:
            # Print the current working directory
            current_dir = os.getcwd()
            logging.info(f"Current working directory: {current_dir}")
            
            # Use absolute path for the CSV file
            file_path = os.path.abspath(os.path.join( 
                'Lesion_Normal_Plantar_Classifier.egg-info', 
                'notebook', 
                'data', 
                'stud.csv')
                )
            logging.info(f"CSV file path: {file_path}")

            # Check if the file exists
            if not os.path.isfile(file_path):
                logging.error(f"File not found: {file_path}")
                # List files in the directory to debug
                logging.info(f"Files in the data directory: {os.listdir(os.path.dirname(file_path))}")
                raise FileNotFoundError(f"No such file or directory: '{file_path}'")
            
            df = pd.read_csv(file_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test Split is Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomerException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
