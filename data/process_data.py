import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data -> reading the data from given file path of csv
    and store it into pandas
    arg -> message file path and categories file path
    return -> the dataframe after merging the message and categories
    '''
    all_messages = pd.read_csv(messages_filepath)
    message_categoty = pd.read_csv(categories_filepath)
#     print(all_messages.shape)
#     print(message_categoty.shape)
    result = pd.concat([all_messages, message_categoty], axis=1)
#     print(result.shape)
    return result
    
def clean_data(df):
    '''
    clean_data -> remove the unnecessary data from dataframe
    arg -> dataframe
    return -> cleaned dataset with message and categories
    '''
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[1].str.split('-').str.get(0)
    
    for col in categories:
        categories[col] = pd.to_numeric(categories[col].str.get(-1))
        
    df = pd.concat([df, categories], axis = 1)
    df = df.drop(['categories', 'child_alone'], axis=1)
    df['related'] = df['related'].replace(2, 1)
    return df.drop_duplicates()
    
    
def save_data(df, database_filename):
    '''
    save_data -> saving the dataframe to the database file
    arg -> cleaned dataframe and output file name
    '''
    sql_engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseTable', sql_engine, index = False, if_exists='replace')


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