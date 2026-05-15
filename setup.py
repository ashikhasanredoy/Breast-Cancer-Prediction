from setuptools import find_packages, setup
from typing import List

DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path, "r") as file:
        requirements = file.readlines()

    requirements = [req.strip() for req in requirements]

    if DOT in requirements:
        requirements.remove(DOT)

    return requirements


setup(
    name="breast_cancer_prediction_project",
    version="0.0.1",
    author="Ashik Hasan Redoy",
    author_email="ashikhasanhredoy@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)