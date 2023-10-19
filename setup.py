from setuptools import setup, find_packages

setup(
    name='synthetic_data_generation',
    version='0.1.14',
    description='Algorithms for generating synthetic data',
    author='Daan Knoors',
    author_email='d.knoors@gmail.com',
    packages=find_packages(),
    install_requires=[
        'diffprivlib>=0.6.3',
        'dill>=0.3.4',
        'dython==0.6.8',
        'joblib>=1.0.1',
        'lifelines>=0.26.0',
        'matplotlib>=3.4.3',
        'numpy>=1.21.2',
        'pandas>=1.3.2',
        'pyjanitor>=0.21.2',
        'pandas_flavor==0.6.0',
        'scikit_learn>=1.0.2',
        'scipy>=1.7.1',
        'seaborn>=0.12.0',
        'thomas_core==0.1.3'
    ],
    extras_require={'interactive': ['jupyter']},
)