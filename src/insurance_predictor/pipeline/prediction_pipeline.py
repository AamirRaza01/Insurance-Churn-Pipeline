import os
import sys
import pandas as pd
from insurance_predictor.constants import SAVED_MODEL_DIR, MODEL_FILE_NAME
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
from insurance_predictor.utils.main_utils import load_object


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)

    def predict(self, dataframe: pd.DataFrame):
        try:
            logger.info("Loading model for prediction")
            model = load_object(self.model_path)
            prediction = model.predict(dataframe)
            return prediction
        except Exception as e:
            raise InsuranceException(e, sys)


class CustomData:
    def __init__(self,
                 Gender: str,
                 Age: int,
                 Driving_License: int,
                 Region_Code: float,
                 Previously_Insured: int,
                 Vehicle_Age: str,
                 Vehicle_Damage: str,
                 Annual_Premium: float,
                 Policy_Sales_Channel: float,
                 Vintage: int):

        self.Gender = Gender
        self.Age = Age
        self.Driving_License = Driving_License
        self.Region_Code = Region_Code
        self.Previously_Insured = Previously_Insured
        self.Vehicle_Age = Vehicle_Age
        self.Vehicle_Damage = Vehicle_Damage
        self.Annual_Premium = Annual_Premium
        self.Policy_Sales_Channel = Policy_Sales_Channel
        self.Vintage = Vintage

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage]
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise InsuranceException(e, sys)