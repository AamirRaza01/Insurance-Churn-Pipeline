import os
import sys
import pandas as pd
from insurance_predictor.entity.config_entity import DataValidationConfig
from insurance_predictor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
from insurance_predictor.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig = DataValidationConfig()):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(self.data_validation_config.schema_file_path)
        except Exception as e:
            raise InsuranceException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise InsuranceException(e, sys)

    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self.schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            for column in self.schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_numerical_columns:
                logger.info(f"Missing numerical columns: {missing_numerical_columns}")
            if missing_categorical_columns:
                logger.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise InsuranceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")
            error_message = ""

            train_dataframe = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_dataframe = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train dataframe does not contain all columns\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test dataframe does not contain all columns\n"

            # Validate column existence
            status = self.is_column_exist(dataframe=train_dataframe)
            if not status:
                error_message += "Train dataframe is missing required columns\n"

            status = self.is_column_exist(dataframe=test_dataframe)
            if not status:
                error_message += "Test dataframe is missing required columns\n"

            validation_status = len(error_message) == 0

            validation_report = {
                "validation_status": validation_status,
                "message": error_message if error_message else "All validations passed",
                "train_shape": list(train_dataframe.shape),
                "test_shape": list(test_dataframe.shape)
            }

            write_yaml_file(
                file_path=self.data_validation_config.validation_report_file_path,
                content=validation_report
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=error_message if error_message else "All validations passed",
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logger.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)