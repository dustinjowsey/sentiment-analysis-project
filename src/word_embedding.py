import numpy as np
import pandas as pd
import constants
import os
from pathlib import Path
from sklearn.model_selection import train_test_split 
import csv
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import threading
import itertools


PATH_TO_GLOVE = Path.home() / "Downloads" / "glove.twitter.27B.200d.txt"
PATH_TO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PATH_TO_DATA = PATH_TO_ROOT + "/data"
PATH_TO_SAVE = PATH_TO_DATA + "/word_embedding/"
BATCH_SIZE = 8
SAMPLES_PER_LABEL = 9000
MAX_PROCESSES = 10

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

    def sample_labels(self, num_samples_per_label, X, y):
        list_of_labels = [-1, 0, 1]
        samples = {label: [] for label in list_of_labels}
        counts = {label: 0 for label in list_of_labels}
        for index in range(0, len(y), 1):
            label = y[index]
            if counts[label] >= num_samples_per_label:
                if sum(counts.values()) >= num_samples_per_label * len(list_of_labels):
                    break
                continue
            samples[label].append(index)
            counts[label] += 1 

        all_samples = np.concatenate([samples[label] for label in list_of_labels])
        np.random.seed(constants.SEED)
        np.random.shuffle(all_samples)

        return X.filter(items = all_samples, axis=0), y.filter(items = all_samples, axis=0)
    
    def __process_batch(self, comments, index, model, tokenizer, total_embeddings):
        embeddings = []
        inputs = tokenizer(comments, padding=True, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        
        #average the vector for each comment
        mask = inputs['attention_mask']

        for j in range(len(comments)):
            comment_embeddings = last_hidden_states[j]
            cur_mask = mask[j]
            embedding = comment_embeddings[cur_mask == 1].mean(dim=0)
            embeddings.append(embedding)
        total_embeddings.append((index, embeddings))
     
    def generate_bert(self, out_file, num_samples = SAMPLES_PER_LABEL):
        """
        Generates BERT word embeddings. This gives the vectors context

        Based on the code in 2.5 https://medium.com/@davidlfliang/intro-getting-started-with-text-embeddings-using-bert-9f8c3b98dee6
        """
        filename = self.file.split('/')[-1]
        print(f"Creating Word Embeddings For '{filename}'")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        df, header = self.__load_file()
        embeddings = []

        if header:
            comments = df['comment']
        else:
            comments = df[0]

        _, ext = os.path.splitext(out_file)
        if ext == ".csv":
            labels = df[1]
        else:
            labels = df["label"]

        if num_samples > 0:
            comments, labels = self.sample_labels(num_samples_per_label=num_samples, X=comments, y=labels)
            labels.astype(int)
    
        comments = [str(comment) for comment in comments]
        #Need batches otherwise we will fill up memory
        batches = len(comments) // BATCH_SIZE + (1 if len(comments) % BATCH_SIZE != 0 else 0)
        
        #multithreading support
        #threads = []
        embeddings = []

        progress_bar = tqdm(total=batches, desc="Batches Completed ", leave=True)
        for i in range(batches):
            batch_comments = comments[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            self.__process_batch(batch_comments, i, model, tokenizer, embeddings)
            #t = threading.Thread(target=self.__process_batch, args=(batch_comments, i, model, tokenizer, embeddings))
            #threads.append(t)
            #t.start()
            progress_bar.update(1)
        progress_bar.close()
        #for thread in threads:
        #    thread.join()

        embeddings.sort(key=lambda x: x[0])
        #embeddings = [embedding for _, batch_embeddings in embeddings for embedding in batch_embeddings]
        embeddings = list(itertools.chain(*[batch_embeddings for _, batch_embeddings in embeddings]))

        embeddings = torch.stack(embeddings)
        out_comments = embeddings.tolist()

        labels = labels.reset_index(drop=True)
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
wb.generate_bert("/../../alt/data/Combined/word_embedding/test.csv", num_samples=2000)
wb = WordEmbedding("/../alt/data/Combined/train.csv")
wb.generate_bert("/../../alt/data/Combined/word_embedding/train.csv", num_samples=9000)