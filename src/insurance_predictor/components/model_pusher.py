import os
import sys
import shutil
from insurance_predictor.entity.config_entity import ModelPusherConfig
from insurance_predictor.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact
)
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger


class ModelPusher:
    def __init__(self,
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig = ModelPusherConfig()):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info("Starting model pusher")

            trained_model_path = self.model_evaluation_artifact.trained_model_path

            # Create saved_models directory
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)

            # Copy model to saved_models
            shutil.copy(src=trained_model_path, dst=saved_model_path)
            logger.info(f"Model pushed to: {saved_model_path}")

            # Also save in model pusher artifact dir
            model_pusher_dir = self.model_pusher_config.model_pusher_dir
            os.makedirs(model_pusher_dir, exist_ok=True)
            pusher_model_path = os.path.join(model_pusher_dir, "model.pkl")
            shutil.copy(src=trained_model_path, dst=pusher_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path,
                model_file_path=pusher_model_path
            )

            logger.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise InsuranceException(e, sys)