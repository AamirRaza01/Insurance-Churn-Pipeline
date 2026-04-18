import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from insurance_predictor.constants import *
from insurance_predictor.entity.config_entity import DataTransformationConfig
from insurance_predictor.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
from insurance_predictor.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig = DataTransformationConfig()):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsuranceException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise InsuranceException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            numerical_columns = self.schema_config["numerical_columns"]
            categorical_columns = self.schema_config["categorical_columns"]

            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("encoder", OrdinalEncoder()),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")

            train_df = self.read_data(self.data_validation_artifact.validation_report_file_path
                                      .replace("data_validation/validation_report.yaml", "")
                                      .replace("data_validation", ""))

            # Read train and test data directly
            artifact_dir = os.path.dirname(
                os.path.dirname(self.data_transformation_config.data_transformation_dir)
            )
            train_file = os.path.join(artifact_dir, "data_ingestion", "ingested", "train.csv")
            test_file = os.path.join(artifact_dir, "data_ingestion", "ingested", "test.csv")

            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

            target_column = self.schema_config["target_column"]

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logger.info("Applying preprocessor on train and test data")
            preprocessor = self.get_data_transformer_object()

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Apply SMOTEENN to handle class imbalance
            logger.info("Applying SMOTEENN for class imbalance")
            smt = SMOTEENN(random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # Save transformed data and preprocessor
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logger.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)