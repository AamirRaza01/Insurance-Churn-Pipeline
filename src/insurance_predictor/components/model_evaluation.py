import os
import sys
import dill
import pandas as pd
import numpy as np
from insurance_predictor.entity.config_entity import ModelEvaluationConfig
from insurance_predictor.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricArtifact
)
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
from insurance_predictor.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    write_yaml_file,
    get_classification_score
)
from insurance_predictor.constants import SAVED_MODEL_DIR, MODEL_FILE_NAME


class ModelEvaluation:
    def __init__(self,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig()):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def get_best_model(self):
        try:
            best_model_path = os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)
            if not os.path.exists(best_model_path):
                return None
            model = load_object(best_model_path)
            return model
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation")

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Evaluate newly trained model
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            y_pred_trained = trained_model.predict(x_test)
            trained_metrics = get_classification_score(y_test, y_pred_trained)

            trained_metric_artifact = ClassificationMetricArtifact(
                f1_score=trained_metrics["f1_score"],
                precision_score=trained_metrics["precision_score"],
                recall_score=trained_metrics["recall_score"]
            )

            # Check if a production model exists
            best_model = self.get_best_model()

            if best_model is None:
                logger.info("No existing model found. New model will be accepted.")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=0.0,
                    best_model_path=self.model_trainer_artifact.trained_model_file_path,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                    train_model_metric_artifact=trained_metric_artifact,
                    best_model_metric_artifact=trained_metric_artifact
                )
                return model_evaluation_artifact

            # Compare with existing model
            y_pred_best = best_model.predict(x_test)
            best_metrics = get_classification_score(y_test, y_pred_best)

            best_metric_artifact = ClassificationMetricArtifact(
                f1_score=best_metrics["f1_score"],
                precision_score=best_metrics["precision_score"],
                recall_score=best_metrics["recall_score"]
            )

            improved_accuracy = trained_metrics["f1_score"] - best_metrics["f1_score"]
            is_model_accepted = improved_accuracy > self.model_evaluation_config.change_threshold

            logger.info(f"Trained model F1: {trained_metrics['f1_score']:.4f}")
            logger.info(f"Best model F1: {best_metrics['f1_score']:.4f}")
            logger.info(f"Improvement: {improved_accuracy:.4f}")
            logger.info(f"Model accepted: {is_model_accepted}")

            # Write evaluation report
            evaluation_report = {
                "is_model_accepted": is_model_accepted,
                "trained_model_f1": float(trained_metrics["f1_score"]),
                "best_model_f1": float(best_metrics["f1_score"]),
                "improved_accuracy": float(improved_accuracy)
            }
            write_yaml_file(self.model_evaluation_config.report_file_path, evaluation_report)

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME),
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=trained_metric_artifact,
                best_model_metric_artifact=best_metric_artifact
            )

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)