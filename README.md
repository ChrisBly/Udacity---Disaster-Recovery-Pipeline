# Udacity---Disaster-Recovery-Pipeline
Udacity - Disaster Recovery Pipeline - Project

## Project Details
1: ETL Pipeline Preparation.ipynb
## libraries
- pandas
- sqlalchemy (create_engine)
## Data
- messages.csv
- categories.csv

References:

https://www.geeksforgeeks.org/python-remove-last-character-in-list-of-strings/
https://stackoverflow.com/questions/12850345/how-to-combine-two-data-frames-in-python-pandas
https://stackoverflow.com/questions/40095712/when-to-applypd-to-numeric-and-when-to-astypenp-float64-in-python
https://stackoverflow.com/questions/53105016/python-lambda-function-syntax-to-transform-a-pandas-groupby-dataframe

2: Machine Learning Pipeline

## libraries
- pandas as pd
- numpy as np
- import warnings
- warnings.filterwarnings('ignore')
- sqlalchemy (create_engine)
- re 
- nltk
- nltk.tokenize import word_tokenize
- nltk.tokenize import sent_tokenize
- nltk.stem import WordNetLemmatizer
- nltk.tokenize import word_tokenize
- sklearn.pipeline import Pipeline
- sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
- sklearn.multioutput import MultiOutputClassifier
- sklearn.ensemble import RandomForestClassifier
- sklearn.model_selection import train_test_split
- nltk.corpus import stopwords 
- sklearn.metrics import classification_report
-  sklearn.model_selection import GridSearchCV
-  pickle

## Data
combined.db

## Model
train_classifier.py

3: Flask App

run.py
