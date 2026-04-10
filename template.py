import os
from pathlib import Path

project_name = "insurance_predictor"

list_of_files = [
    # Package init files
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/constants/__init__.py",

    # Entity
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",

    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",

    # Pipeline
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",

    # Configuration
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/mongo_db_connection.py",
    f"src/{project_name}/configuration/dagshub_connection.py",

    # Data access
    f"src/{project_name}/data_access/__init__.py",
    f"src/{project_name}/data_access/insurance_data.py",

    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",

    # Config files
    "config/schema.yaml",
    "config/config.yaml",

    # Streamlit app
    "app/streamlit_app.py",

    # Notebooks
    "notebooks/eda.ipynb",

    # CI/CD
    ".github/workflows/main.yaml",

    # Root files
    "setup.py",
    "pyproject.toml",
    ".env",
    "Dockerfile",
    "demo.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
        print(f"✅ Created: {filepath}")
    else:
        print(f"⏭️  Already exists: {filepath}")

print("\n🎉 Project structure created successfully!")