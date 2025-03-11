import os
from constants import *

fh = open(CMB_PATH + TRN_NAME, "r", encoding='utf-8')

word_count = {}
for line in fh:
    line = line.split(',')[0]
    for w in line.split(' '):
        if w not in word_count:
            word_count[w] = 1
            continue
        word_count[w] += 1

fh.close()

wcs = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}
keys = list(wcs.keys())

try: os.mkdir(BOW_PATH)
except: pass

fh = open(BOW_PATH + BOW_TXT, "w", encoding='utf-8')
for i in range(BOW_QTY):
    fh.write(f"{keys[i]}\n")
fh.close()