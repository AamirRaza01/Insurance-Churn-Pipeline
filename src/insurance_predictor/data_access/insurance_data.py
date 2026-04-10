import sys
import pandas as pd
import numpy as np
from typing import Optional
from insurance_predictor.configuration.mongo_db_connection import MongoDBClient
from insurance_predictor.constants import DATABASE_NAME
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger


class InsuranceData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise InsuranceException(e, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            logger.info(f"Fetching data from collection: {collection_name}")

            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            logger.info(f"Data fetched successfully. Shape: {df.shape}")

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise InsuranceException(e, sys)