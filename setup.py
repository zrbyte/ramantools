from setuptools import setup, find_packages

setup(
    name="ramantools",
    version="0.2.3",
    packages=find_packages(),
    description="Tools for analyzing Raman spectroscopy data, measured by a Witec confocal Raman microscope.",
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    author="Peter Nemes-Incze",
    url="https://github.com/zrbyte/ramantools",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
