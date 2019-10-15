import sys
# import libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine
import re 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Doc String: 
    Inputs:
    
    Input Files 1:
    Loading Database from ETL pipeline.
    
    Returns: ETL pipeline Database 
        
    """
       
    # Combining the sqlite:/// + database_filepath
    engine = create_engine('sqlite:///' + database_filepath)
    # Creating a Dataframefrom the SQL Table
    df = pd.read_sql_table('Disasters', engine)
    # Define the Inputs for the Disaster Recovery data
    X = df['message'] # Define feature 
    # Create category_names
    Y = pd.concat([df.iloc[:,4:40],df.iloc[:,-1:]],axis = 1)
    category_names = Y.columns
    
    return X,Y,category_names


def tokenize(text):
    """
    Doc String: 
    Inputs:
    
    Text Data gets tokenize text , normalize, lemmatize, 
    using the nltk package
    
    Returns: Text data that has been normalize, lemmatize, and tokenize 
        
    """
    # Normalize
    # Set text to lower case and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # Remove punctuation characters
    # Tokenize words 
    tokens = word_tokenize(text)
    # lemmatizer and remove stopwords
    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    # stopwords
    stop_words = set(stopwords.words('english'))
    # lemmatizer and remove stopwords
    output = [lemmatizer.lemmatize(w) for w in tokens if not w in stop_words]
    output = [] 
    for w in tokens: 
        if w not in stop_words: 
            output.append(w)
    return output

def build_model():
    '''
    Doc String:
    Build Model using GridSearchCV/pipeline
    
    Looking at the default hyper parameters:
    the n_estimators is set to 'warn' default
    min_samples_split is set to 2.
      
    Changing default hyper parameters to improve performanace.
    '''
    # Create pipeline with Classifier
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # Changing default hyper parameters to improve performanace.
    parameters = {'clf__estimator__n_estimators':[10,20],
             'clf__estimator__min_samples_split':[5,10]}
    # grid search
    cv = GridSearchCV(pipeline, parameters)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Doc String
    
    Test your model using sklearn's  classification_report
    Report the f1 score, precision and recall for each output category of the dataset. 
    
    '''
    Y_pred = pipeline.predict(X_test)
    Y_Predicton_df = pd.DataFrame(Y_pred, columns = Y_test.columns)
    # iterating through the columns 
    for column in Y_test:
        for i in Y_test[column].iteritems():
            print('Category: {}\n'.format(column))
            print(classification_report(Y_test[column],Y_Predicton_df[column]))
    pass
    
   
  


def save_model(model, model_filepath):
    '''
    Doc String:
    Saving Model
    '''
    pickle.dump(cv, open('model.pkl', 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()