import os
import sys
import yaml
import dill
import numpy as np
from insurance_predictor.exception import InsuranceException
from insurance_predictor.logger import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise InsuranceException(e, sys)


def write_yaml_file(
    file_path: str,
    content: object,
    replace: bool = False
) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise InsuranceException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info(f"Saving object to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info("Object saved successfully")

    except Exception as e:
        raise InsuranceException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found at path: {file_path}")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise InsuranceException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise InsuranceException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise InsuranceException(e, sys)


def get_classification_score(y_true, y_pred) -> dict:
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        logger.info(
            f"Classification scores — F1: {f1:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        return {
            "f1_score": f1,
            "precision_score": precision,
            "recall_score": recall
        }

    except Exception as e:
        raise InsuranceException(e, sys)