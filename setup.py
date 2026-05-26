from setuptools import setup, find_packages

setup(
    name="insurance_predictor",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)