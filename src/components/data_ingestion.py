import os
import sys  # 'sys' library contains the exceptions.
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass       # Used for defining class variables

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Some inputs will be required during data ingestion such as file path for saving train, test or raw data. etc. So, these inputs will be created in this class. Better to use "@dataclass" when only the class variables has to be initialized. If the class also has functions, then better to use init method.
# Usually to define class variables, we will use init. But with the help of "@dataclass", we will be able to directly define class variables.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artificats', 'train.csv')    # training data will be saved in this path
    test_data_path: str = os.path.join('artificats', 'test.csv')    # training data will be saved in this path
    raw_data_path_data_path: str = os.path.join('artificats', 'data.csv')    # training data will be saved in this path


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()       # 'ingestion_config' will have paths to train, test and raw data.

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")    # This data can be from anywhere i.e., from APIs, database. etc.
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.train_data_path)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("Ingestion of the data is completed")

            # These path info will be required in my data transformation
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)