from dataclasses import dataclass
from datetime import datetime
import os
from insurance_predictor.constants import *

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = "insurance_predictor"
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR
    )
    feature_store_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR,
        DATA_INGESTION_FEATURE_STORE_DIR,
        "insurance.csv"
    )
    training_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR,
        DATA_INGESTION_INGESTED_DIR,
        "train.csv"
    )
    testing_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR,
        DATA_INGESTION_INGESTED_DIR,
        "test.csv"
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR
    )
    validation_report_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR,
        DATA_VALIDATION_REPORT_FILE_NAME
    )
    schema_file_path: str = SCHEMA_FILE_PATH


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR
    )
    transformed_train_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        "train.npy"
    )
    transformed_test_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        "test.npy"
    )
    transformed_object_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSOR_OBJECT_FILE_NAME
    )


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR
    )
    trained_model_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_TRAINER_TRAINED_MODEL_NAME
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    overfitting_underfitting_threshold: float = MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD


@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_EVALUATION_DIR
    )
    report_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_EVALUATION_DIR,
        MODEL_EVALUATION_REPORT_NAME
    )
    change_threshold: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_PUSHER_DIR
    )
    saved_model_path: str = os.path.join(
        SAVED_MODEL_DIR,
        MODEL_FILE_NAME
    )