import os
import sys
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from insurance_predictor.entity.config_entity import ModelTrainerConfig
from insurance_predictor.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
from insurance_predictor.utils.main_utils import (
    load_numpy_array_data,
    save_object,
    get_classification_score
)


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def train_and_select_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "XGBoost": XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42
                ),
                "LightGBM": LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
            }

            best_model = None
            best_model_name = None
            best_f1 = 0

            for name, model in models.items():
                logger.info(f"Training {name}...")
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                f1 = f1_score(y_test, y_pred)
                logger.info(f"{name} F1 Score: {f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_model_name = name

            logger.info(f"Best model selected: {best_model_name} with F1: {best_f1:.4f}")
            return best_model, best_model_name

        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_model, best_model_name = self.train_and_select_model(
                x_train, y_train, x_test, y_test
            )

            # Get metrics
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_metrics = get_classification_score(y_train, y_train_pred)
            test_metrics = get_classification_score(y_test, y_test_pred)

            # Check for overfitting
            diff = abs(train_metrics["f1_score"] - test_metrics["f1_score"])
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logger.info(f"Possible overfitting detected. Diff: {diff:.4f}")

            # Check minimum expected accuracy
            if test_metrics["f1_score"] < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model F1 score {test_metrics['f1_score']:.4f} is below "
                    f"expected {self.model_trainer_config.expected_accuracy}"
                )

            # Save model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logger.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            train_metric_artifact = ClassificationMetricArtifact(
                f1_score=train_metrics["f1_score"],
                precision_score=train_metrics["precision_score"],
                recall_score=train_metrics["recall_score"]
            )
            test_metric_artifact = ClassificationMetricArtifact(
                f1_score=test_metrics["f1_score"],
                precision_score=test_metrics["precision_score"],
                recall_score=test_metrics["recall_score"]
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )

            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise InsuranceException(e, sys)