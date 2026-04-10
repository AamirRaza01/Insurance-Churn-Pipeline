import os
from datetime import datetime

# MongoDB
MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME = "insurance_db"
COLLECTION_NAME = "insurance_data"

# Pipeline artifacts root
ARTIFACT_DIR = "artifacts"
TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2
DATA_INGESTION_COLLECTION_NAME = "insurance_data"

# Data Validation
DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME = "validation_report.yaml"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# Data Transformation
DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
PREPROCESSOR_OBJECT_FILE_NAME = "preprocessor.pkl"

# Model Trainer
MODEL_TRAINER_DIR = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

# Model Evaluation
MODEL_EVALUATION_DIR = "model_evaluation"
MODEL_EVALUATION_REPORT_NAME = "evaluation_report.yaml"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02

# Model Pusher
MODEL_PUSHER_DIR = "model_pusher"
SAVED_MODEL_DIR = "saved_models"
MODEL_FILE_NAME = "model.pkl"

# DagsHub
DAGSHUB_USERNAME_KEY = "DAGSHUB_USERNAME"
DAGSHUB_TOKEN_KEY = "DAGSHUB_TOKEN"
DAGSHUB_REPO_NAME = "InsuranceLeadPredictor-MLOps"