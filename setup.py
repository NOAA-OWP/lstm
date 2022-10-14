#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="lstm",
    version="1.0",
    description="Hydrologic model based on Long Short-Term Memory algorithm",
    author="Jonathan Frame",
    author_email="jonathan.frame@noaa.gov",
    url="https://github.com/NOAA-OWP/lstm",
    include_package_data=True,  # (bmi_config_files, trained_neuralhydrology_models)
    # Can use the "package_data" keyword to list them.
    # package_data={},
    packages=find_packages(include=['lstm', 'lstm.*']),
    install_requires=["numpy", "pandas", "bmipy", "torch", "pyyml", "netCDF4", "xarray==0.16.0"]
)
