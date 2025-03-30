import numpy as np
import pandas as pd
import constants
import os
from pathlib import Path
from sklearn.model_selection import train_test_split 
import csv
import torch
from transformers import BertTokenizer, BertModel
from tqdm.notebook import tqdm


PATH_TO_GLOVE = Path.home() / "Downloads" / "glove.twitter.27B.200d.txt"
PATH_TO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATH_TO_DATA = PATH_TO_ROOT + "/data"
PATH_TO_SAVE = PATH_TO_DATA + "/word_embedding/"
BATCH_SIZE = 8
SAMPLES_PER_LABEL = 9000

class WordEmbedding:
    def __init__(self, file, path_to_glove=""):
        """
        Creates the word embedding data for your tests. Use on processed data. WILL NOT WORK ON UNPROCESSED DATA (i.e filtered)

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
    
    def __save_file(self, out_file, header, comments, labels):
        if not os.path.exists(PATH_TO_SAVE):
            os.mkdir(PATH_TO_SAVE)

        _, ext = os.path.splitext(out_file)
        out_df = pd.DataFrame(comments)
        out_df[-1] = labels
        if ext == ".csv":
            out_df.to_csv(PATH_TO_SAVE + out_file, header=False, index=False)
        else:
            out_df.to_csv(PATH_TO_SAVE + out_file, compression="gzip", index=False)

    def __load_file(self):
        header = True

        _, ext = os.path.splitext(PATH_TO_DATA + self.file)
        if ext == ".csv":
            #check if the csv has a header or not before reading with pandas
            temp = open(PATH_TO_DATA + self.file, 'r', encoding='utf-8')
            sniffer = csv.Sniffer()
            header = sniffer.has_header(temp.read(-1))
            if header:
                df = pd.read_csv(PATH_TO_DATA + self.file)
            else:
                df = pd.read_csv(PATH_TO_DATA + self.file, header=None)
        elif ext == ".gz":
            df = pd.read_csv(PATH_TO_DATA + self.file, compression='gzip')
        else:
            print(f"Error! The file must be either a csv or gz not {ext}")
        
        return df, header
     
    def generate_bert(self, out_file, num_samples = SAMPLES_PER_LABEL):
        """
        Generates BERT word embeddings. This gives the vectors context

        Based on the code in 2.5 https://medium.com/@davidlfliang/intro-getting-started-with-text-embeddings-using-bert-9f8c3b98dee6
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        df, header = self.__load_file()
        embeddings = []

        if header:
            comments = df['comment'].tolist()
        else:
            comments = df[0].tolist()
        comments = [str(comment) for comment in comments]

        _, ext = os.path.splitext(out_file)
        if ext == ".csv":
            labels = df[1]
        else:
            labels = df["label"]

        if(num_samples > 0):
            #samples to pick per label
            samples_per_label = num_samples

            list_of_labels = [-1, 0, 1]
            indicies = {label: np.where(labels == label)[0] for label in list_of_labels}
            #grab the first occurances
            samples = {label: [i for i in range(0, samples_per_label)] for label in list_of_labels}
            #samples = {label: np.random.choice(indicies[label], samples_per_label, replace=False) for label in list_of_labels}

            all_samples = np.concatenate([samples[label] for label in list_of_labels])
            #want to mix up labels now since samples are seperated by label
            np.random.shuffle(all_samples)

            final_labels = []
            final_comments = []
            for sample in all_samples:
                final_labels.append(labels[sample])
                final_comments.append(comments[sample])
            
            labels = final_labels
            comments = final_comments
    
        #Need batches otherwise we will fill up memory
        batches = len(comments) // BATCH_SIZE + (1 if len(comments) % BATCH_SIZE != 0 else 0)
        
        #progress_bar = tqdm(total=batches, desc="Batches Completed ")
        print(f"Running {batches} batches")
        for i in range(batches):
            batch_comments = comments[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            inputs = tokenizer(batch_comments, padding=True, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state

            #average the vector for each comment
            mask = inputs['attention_mask']

            for j in range(len(batch_comments)):
                comment_embeddings = last_hidden_states[j]
                cur_mask = mask[j]
                embedding = comment_embeddings[cur_mask == 1].mean(dim=0)
                embeddings.append(embedding)
            #progress_bar.update()
        
        embeddings = torch.stack(embeddings)
        out_comments = embeddings.tolist()

        self.__save_file(out_file=out_file, header=header, comments=out_comments, labels=labels)

    def generate_word_embedding(self, out_file):
        """
        Generates the word embeddings into /data/word_emebddings
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

        dim = len(splits)
        df, header = self.__load_file()

        _, ext = os.path.splitext(out_file)
        if ext == ".csv":
            comments = np.array([self.__convert_comment(str(comment), dim) for comment in df[0]])
            labels = df[1]
        else:
            comments = np.array([self.__convert_comment(str(comment), dim) for comment in df['comment']])
            labels = df["label"]

        self.__save_file(out_file=out_file, header=header, comments=comments, labels=labels)
        #X = np.array([self.__convert_comment(str(comment), dim) for comment in comments])
        #return X, labels

wb = WordEmbedding("/../alt/data/Combined/test.csv")
wb.generate_bert("/../../alt/data/Combined/word_embedding/test.csv", num_samples=-1)
wb = WordEmbedding("/../alt/data/Combined/train.csv")
wb.generate_bert("/../../alt/data/Combined/word_embedding/train.csv", num_samples=9000)