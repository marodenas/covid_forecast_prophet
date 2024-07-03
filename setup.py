from setuptools import setup, find_packages

setup(
    name='my_data_science_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'fastapi',
        'uvicorn',
        'joblib',
        'streamlit',
    ],
    test_suite='tests',
)
