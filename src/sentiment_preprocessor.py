import numpy as np
import string
import pandas as pd
import csv
import gzip
import shutil
from sklearn.feature_extraction.text import CountVectorizer

from constants import PATH_DATA_RAW
from constants import PATH_DATA_PROCESSED
from constants import PATH_DATA_BOW

labels = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}

#Preprocessor class for data
class SentimentPreprocessor:
    def __init__(self, filename):
        self.filename = filename
    
    def filter(self, comment_col, label_col):
        """Filters out characters like '.', ',' and converts to lower case"""

        #convert file to gzip to save storage
        #NOTE you need to manually delete the old csv file
        if not self.filename.endswith('.gz'):
            fh1_cv = open(PATH_DATA_RAW + self.filename, "rb")
            fh1_gz = gzip.open(PATH_DATA_RAW + self.filename + ".gz", "wb")
            shutil.copyfileobj(fh1_cv, fh1_gz)
            fh1_cv.close()
            fh1_gz.close()
            self.filename += ".gz"

        #open gzip file
        fh1 = gzip.open(PATH_DATA_RAW + self.filename, "rt", encoding='utf-8')
        fh2 = gzip.open(PATH_DATA_PROCESSED + self.filename, "wt", encoding='utf-8')
        # youtube_sentiment.csv:
        # ['', 'Video ID', 'Comment', 'Likes', 'Sentiment']

        #write headers so we can use the file with pandas
        fh2.write(f"comment,label\n")

        reader = csv.reader(fh1)

        #skip the first row so we can have our own defined header
        next(reader)

        for row in reader:
            comment, sentiment = row[comment_col], row[label_col]
            #Don't add entries with missing comments
            if not comment:
                continue
            # filter out punctuation characters from each comment then join each valid character on ''
            filtered = ''.join(filter(lambda c: c not in string.punctuation + "\n", comment.lower().strip()))
            fh2.write(f"{filtered},{sentiment}\n")

        fh1.close()
        fh2.close()

    def map_labels(self, negative, neutral, positive):
        """Converts labels to -1,0,1 for negaitve, neutral, and positive respectively"""

        df = self.__load_data()
        df['label'] = df['label'].replace(negative, labels['negative'])
        df['label'] = df['label'].replace(neutral, labels['neutral'])
        df['label'] = df['label'].replace(positive, labels['positive'])
        df.to_csv(PATH_DATA_PROCESSED + self.filename, index=False)

    def __load_data(self):
        """Loads data into a pandas dataframe
        NOTE need to load from filtered data"""
        try:
            df = pd.read_csv(PATH_DATA_PROCESSED + self.filename, compression='gzip')
            return df
        except:
            raise Exception(f"Could not load data from '{PATH_DATA_PROCESSED + self.filename}'\nDid you filter the data first?\nThe data should be compressed as gzip")

    def bag_of_words(self):
        """Turns the processed data into a bag of words representation"""
        df = self.__load_data()

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df["comment"].to_list(), df['label'].to_list())

        out_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        out_df['label'] = df['label']
        out_df.to_csv(PATH_DATA_BOW + self.filename, index=False, compression='gzip')




