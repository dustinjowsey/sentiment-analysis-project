import csv
from topn import TOP_N
PATH_YT         = "../data/YouTube/preprocessed.csv"
PATH_TOPN       = f"../data/YouTube/top{TOP_N}.txt"
PATH_RESULT     = f"../data/YouTube/top{TOP_N}_wf.csv"


# Get list of top n words
fh          = open(PATH_TOPN, 'r', encoding='utf8')
top_n       = [line.strip() for line in fh]
fh.close()

fhi         = open(PATH_YT, 'r', encoding='utf8')
fho         = open(PATH_RESULT, 'w', encoding='utf8')
reader      = csv.reader(fhi)
for row in reader:
    comment, sentiment = row[0], row[1]

    # Initialize new dict to represent word-frequency
    wf      = {w: 0 for w in top_n}

    # For each occurence of a word in top_n, increment its count by 1
    for w in comment.split(' '):
        w = w.strip()
        if w in wf.keys(): wf[w] += 1

    for v in wf.values():
        fho.write(f"{v},")
    fho.write(f"{sentiment}\n")