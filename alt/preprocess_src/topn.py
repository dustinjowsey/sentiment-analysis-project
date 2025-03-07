import csv

TOP_N           = 1000
PATH_YT         = "../data/YouTube/preprocessed.csv"
PATH_RESULT     = f"../data/YouTube/top{TOP_N}.txt"

fh = open(PATH_YT, "r", encoding='utf-8')
reader = csv.reader(fh)

# Create a .txt file containing the TOP_N most frequent words found in the provided 
# .csv separated by \n, and ordered most frequent to least.
word_count = {}
for row in reader:
    comment = row[0]
    for w in comment.split(' '):
        w = w.strip()
        if w == '': continue
        if w not in word_count:
            word_count[w] = 1
            continue
        word_count[w] += 1

fh.close()

wcs = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
keys = list(wcs.keys())

fh = open(PATH_RESULT, "w", encoding='utf-8')
for i in range(TOP_N):
    fh.write(f"{keys[i]}\n")
fh.close()