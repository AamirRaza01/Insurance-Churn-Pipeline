import os
import sys
from dotenv import load_dotenv

load_dotenv()

from insurance_predictor.pipeline.training_pipeline import TrainingPipeline
from insurance_predictor.logger import logger
from insurance_predictor.exception import InsuranceException


def main():
    try:
        logger.info("Starting training pipeline from demo.py")
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()

        if artifact:
            logger.info(f"Pipeline completed. Model saved at: {artifact.saved_model_path}")
            print(f"\n✅ Training complete! Model saved at: {artifact.saved_model_path}")
        else:
            logger.info("Pipeline completed but model was not pushed (existing model is better)")
            print("\n⚠️ Training complete but existing model was better — no update made")

    except InsuranceException as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()