import os
import certifi
from pymongo import MongoClient
from insurance_predictor.constants import MONGODB_URL_KEY, DATABASE_NAME
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
import sys


ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.environ.get(MONGODB_URL_KEY)

                if mongo_db_url is None:
                    raise Exception(
                        f"Environment variable {MONGODB_URL_KEY} is not set"
                    )

                MongoDBClient.client = MongoClient(
                    mongo_db_url,
                    tlsCAFile=ca
                )
                logger.info("MongoDB connection established successfully")

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise InsuranceException(e, sys)