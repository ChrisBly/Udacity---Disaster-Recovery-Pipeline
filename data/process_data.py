import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
# ETL Pipeline Preparation

def load_data(messages_filepath, categories_filepath):
    # 1. load datasets.
    # input paths:
    # messages_filepath
    # categories_filepath
    # Load messages.csv into a dataframe 
    messages = pd.read_csv(messages_filepath)
    # Load categories.csv into a dataframe 
    categories = pd.read_csv(categories_filepath)
    # Merge Data set
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df


def clean_data(df):
    # Section 2 Clean:
    # Input Dataframe: df from load_data
    # Split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = list(map(lambda i: i[ : -2], row)) 
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].astype('int')
        # Replace categories column in df with new category columns.
        df.drop(['categories'],axis=1,inplace=True)
        # concatenate the original dataframe with the new `categories` dataframe
        df = pd.concat([df, categories], axis=1)
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        return df


def save_data(df, database_filename):
    # inpurt files:
    # df - data frame
    # database_filename
    # Save the clean dataset into an sqlite database  
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()