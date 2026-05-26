import os
import sys
import dagshub
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger
from dotenv import load_dotenv

load_dotenv()


class DagsHubConnection:
    client = None

    def __init__(self):
        try:
            if DagsHubConnection.client is None:
                username = os.environ.get("DAGSHUB_USERNAME")
                token = os.environ.get("DAGSHUB_TOKEN")
                repo_name = "InsuranceLeadPredictor-MLOps"

                if not username or not token:
                    raise Exception(
                        "DAGSHUB_USERNAME or DAGSHUB_TOKEN "
                        "environment variables are not set"
                    )

                os.environ["MLFLOW_TRACKING_USERNAME"] = username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token

                dagshub.init(
                    repo_owner=username,
                    repo_name=repo_name,
                    mlflow=True
                )

                DagsHubConnection.client = dagshub
                logger.info("DagsHub connection established successfully")

            self.client = DagsHubConnection.client

        except Exception as e:
            raise InsuranceException(e, sys)