import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

install_requires = [
    'thomas-core',
    'diffprivlib',
    'dill',
    'lifelines',
    'dython',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'seaborn'
]


# This call to setup() does all the work
setup(
    name="synthetic-data-generation",
    version="0.1.1",
    description="Algorithms for generating synthetic data",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/daanknoors/synthetic_data_generation",
    author="Daan Knoors",
    author_email="d.knoors@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(include=["synthesis", 'synthesis.*']),
    include_package_data=True,
    install_requires=install_requires,
    keywords='synthetic-data synthesis synthetic-data-generation synthesizer differential-privacy privacy privbayes',
)