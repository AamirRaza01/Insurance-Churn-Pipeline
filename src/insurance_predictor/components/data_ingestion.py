import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from insurance_predictor.data_access.insurance_data import InsuranceData
from insurance_predictor.entity.config_entity import DataIngestionConfig
from insurance_predictor.entity.artifact_entity import DataIngestionArtifact
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logger.info("Exporting data from MongoDB to feature store")

            insurance_data = InsuranceData()
            dataframe = insurance_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logger.info(f"Dataset shape: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(f"Data saved to feature store: {feature_store_file_path}")

            return dataframe

        except Exception as e:
            raise InsuranceException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> DataIngestionArtifact:
        try:
            logger.info("Performing train test split")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe["Response"]
            )

            logger.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            train_file_path = self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.testing_file_path

            dir_path = os.path.dirname(train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)

            logger.info(f"Train data saved to: {train_file_path}")
            logger.info(f"Test data saved to: {test_file_path}")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=train_file_path,
                test_file_path=test_file_path
            )

            logger.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")
            dataframe = self.export_data_into_feature_store()
            data_ingestion_artifact = self.split_data_as_train_test(dataframe)
            logger.info("Data ingestion completed successfully")
            return data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(e, sys)