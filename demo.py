import os
from dotenv import load_dotenv

load_dotenv()

from insurance_predictor.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()