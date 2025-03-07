import csv
import string

# In order for this to work, import the csv into google docs then download it again.
# That fixes some odd encoding.
PATH_YT_I       = "../data/YouTube/raw.csv"
PATH_YT_O       = "../data/YouTube/preprocessed.csv"

# comments.csv:
# ['', 'Video ID', 'Comment', 'Likes', 'Sentiment']

fh1 = open(PATH_YT_I, "r", encoding='utf-8')
fh2 = open(PATH_YT_O, "w", encoding='utf-8')
fh1.readline()
reader = csv.reader(fh1)

for row in reader:
    comment, sentiment = row[2], row[4]

    # filter out punctuation characters from each comment then join each valid character on ''
    filtered = ''.join(filter(lambda c: c not in string.punctuation + "\n", comment.lower().replace('\n', ' ').replace('.', ' ').strip()))
    fh2.write(f"{filtered},{int(sentiment)-1}\n") # -1 abuses the labels being 1 offset from what we want. Only good for the YouTube set perhaps.

fh1.close()
fh2.close()