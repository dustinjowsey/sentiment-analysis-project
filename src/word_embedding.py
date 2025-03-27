import numpy as np
import pandas as pd
import constants
import os
from pathlib import Path
from sklearn.model_selection import train_test_split 

PATH_TO_GLOVE = Path.home() / "Downloads" / "glove.twitter.27B.25d.txt"

class WordEmbedding:
    def __init__(self, file, path_to_glove=""):
        """
        Creates the word embedding data for your tests.

        Args:
        file (str): The PREPROCESSED file to convert to word embeddings
        path_to_glove (str): Path to the downloaded glove embeddings
                             NOTE These files are 25, 50, 100, or 200 dimensions as the names suggest
        """
        self.file = file
        self.path_to_glove = path_to_glove
        self.embeddings = {}

    def __convert_comment(self, comment, dim):
        vectors = []
        for word in comment:
            if word in self.embeddings:
                vectors.append(self.embeddings[word])
            else:
                vectors.append(np.zeros(dim))

        return np.mean(vectors)
    
    def generate_word_embedding(self):
        """
        Generates the word embeddings

        Returns:
            You can use these in test train split

            X: List of vector averages for each comment 
            
            y: List of labels
        """

        if self.path_to_glove == "":
            try:
                glove = open(PATH_TO_GLOVE, 'r', encoding='utf-8')
            except:
                print(f"ERROR! need a valid path_to_glove paramter, but got '{self.path_to_glove}'")
        else:
            try:
                glove = open(self.path_to_glove, 'r', encoding='utf-8')
            except:
                print(f"ERROR! need a valid path_to_glove paramter, but got '{self.path_to_glove}'")
                return

        #create the dictionary to store the word vector values
        for line in glove:
            splits = line.split(' ')
            word = splits[0]
            vector = np.asarray(splits[1:], dtype='float32')
            self.embeddings[word] = vector
        
        dim = len(splits) - 1
        #Convert file into a vector format based on the glove file
        _, ext = os.path.splitext(self.file)
        if ext == ".csv":
            df = pd.read_csv(self.file)
        elif ext == ".gz":
            df = pd.read_csv(self.file, compression='gzip')
        else:
            print(f"Error! The file must be either a csv or gz not {ext}")
            return
        
        comments = df["comment"]
        labels = df["label"]
        print(labels)
        vector = []
        X = np.array([self.__convert_comment(str(comment), dim) for comment in comments])
        return X, labels