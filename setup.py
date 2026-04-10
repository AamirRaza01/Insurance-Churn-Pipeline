from setuptools import setup, find_packages

setup(
    name="insurance_predictor",
    version="0.1.0",
    author="Aamir Raza",
    # author_email="your@email.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)