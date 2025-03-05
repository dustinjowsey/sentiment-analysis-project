import csv
import string

# In order for this to work, import the csv into google docs then download it again.
# That fixes some odd encoding.
PATH_DATA_RAW = "../data/raw/"
PATH_DATA_FILTERED = "../data/filtered/"

#Filters out characters like '.', ','
def filter_sentiment(file_name, comment_col, row_col):
    fh1 = open(PATH_DATA_RAW + file_name, "r", encoding='utf-8')
    fh2 = open(PATH_DATA_FILTERED + file_name, "w", encoding='utf-8')
    # youtube_sentiment.csv:
    # ['', 'Video ID', 'Comment', 'Likes', 'Sentiment']

    reader = csv.reader(fh1)

    for row in reader:
        comment, sentiment = row[comment_col], row[row_col]

        # filter out punctuation characters from each comment then join each valid character on ''
        filtered = ''.join(filter(lambda c: c not in string.punctuation + "\n", comment.lower().strip()))
        fh2.write(f"{filtered},{sentiment}\n")

    fh1.close()
    fh2.close()