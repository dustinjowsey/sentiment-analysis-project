import csv
import string
from constants import *

# In order for this to work, import the csv into google docs then download it again.
# That fixes some odd encoding.

# YouTube labels need only be reduced by 1
def f1(L):
    return int(L) - 1

# Reddit labels are what they should be
def f2(L):
    return int(L)

# Twitter labels are in natural language so they will be converted to -1, 0, or 1
def f3(L):
    if L == "positive": return 1
    if L == "negative": return -1
    return 0

# comment row, label row, and label transformation function
formats         = [(2, 4, f1), (0, 1, f2), (1, 2, f3)]

for i in range(len(PATHS)):
    fh1 = open(PATHS[i] + RAW_NAME, 'r', encoding='utf-8')
    fh2 = open(PATHS[i] + CLN_NAME, 'w', encoding='utf-8')

    fh1.readline()
    reader = csv.reader(fh1)

    for row in reader:
        comment, sentiment = row[formats[i][0]], row[formats[i][1]]

        # massive filter line to clean the text
        filtered = ''.join(filter(lambda c: c not in string.punctuation and c not in string.digits + "\n", comment.lower().replace('\n', ' ').replace('.', ' ').strip()))
        filtered = filtered.split(' ')
        filtered = ' '.join([w for w in filtered if w != ''])
        fh2.write(f"{filtered},{formats[i][2](sentiment)}\n") # Write the cleaned comment and adjusted label to the file

    fh1.close()
    fh2.close()
    print(PATHS[i] + RAW_NAME, "==>", (PATHS[i] + CLN_NAME))