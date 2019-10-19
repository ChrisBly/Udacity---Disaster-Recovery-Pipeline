# Udacity---Disaster-Recovery-Pipeline
Udacity - Disaster Recovery Pipeline - Project

### Project Workspace - ETL
The first part of your data pipeline is the Extract, Transform, and Load process. Here, you will read the dataset, clean the data, and then store it in a SQLite database. We expect you to do the data cleaning with pandas. To load the data into an SQLite database, you can use the pandas dataframe .to_sql() method, which you can use with an SQLAlchemy engine.

Feel free to do some exploratory data analysis in order to figure out how you want to clean the data set. Though you do not need to submit this exploratory data analysis as part of your project, you'll need to include your cleaning code in the final ETL script, process_data.py.

### Project Workspace - Machine Learning Pipeline
For the machine learning portion, you will split the data into a training set and a test set. Then, you will create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, you will export your model to a pickle file. After completing the notebook, you'll need to include your final machine learning code in train_classifier.py.

### Data Pipelines: Python Scripts
After you complete the notebooks for the ETL and machine learning pipeline, you'll need to transfer your work into Python scripts, process_data.py and train_classifier.py. If someone in the future comes with a revised or new dataset of messages, they should be able to easily create a new model just by running your code. These Python scripts should be able to run with additional arguments specifying the files used for the data and model.


### F lask App
In the last step, you'll display your results in a Flask web app. We have provided a workspace for you with starter files. You will need to upload your database file and pkl file with your model.

This is the part of the project that allows for the most creativity. So if you are comfortable with html, css, and javascript, feel free to make the web app as elaborate as you would like.

In the starter files, you will see that the web app already works and displays a visualization. You'll just have to modify the file paths to your database and pickled model file as needed.

There is one other change that you are required to make. We've provided code for a simple data visualization. Your job will be to create two additional data visualizations in your web app based on data you extract from the SQLite database. You can modify and copy the code we provided in the starter files to make the visualizations.

## Below list the sections of this code repo:

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

## 1. Import libraries and load data from database
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

## 1.1 load data from database
combined.db

2. Write a tokenization function to process your text data

3. Build a machine learning pipeline

## 3.1 Create pipeline with Classifier

4 Train pipeline


## 4.1 Split data into train and test sets
## 4.2 Train pipeline

5. Test your model


## Model
train_classifier.py

#### Reference
- https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
- https://stackoverflow.com/questions/47481874/how-to-get-the-column-name-when-iterating-through-dataframe-pandas
- https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6

3: Flask App

## App folder:

### run.py
####  1 libraries
- pandas
- sqlalchemy (create_engine)
- numpy as np

### templates folder:

go.html
master.html


## Data Folder
- DisasterResponse.db
- disaster_categories.csv
- disaster_messages.csv

### process_data.py
#### ETL Functions:

- load_data
- clean_data
- save_data
- main

## Model Folder

model.pkl

### train_classifier.py


## 1. Import libraries and load data from database
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

#### Functions

- load_data
- Tokenize
- build_model
- Evaluate_model
- save_model
- main

### Reference: 
- https://knowledge.udacity.com/questions/43882
- https://knowledge.udacity.com/questions/57729
