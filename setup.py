from setuptools import find_packages, setup

packages = find_packages(exclude=["tests"])

setup(
    name="classify",
    packages=packages
)
