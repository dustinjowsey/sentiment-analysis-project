from constants import *

# Get list of top n words
fh          = open(BOW_PATH + BOW_TXT, 'r', encoding='utf8')
top_n       = [line.strip() for line in fh]
fh.close()

fh          = open(CMB_PATH + TRN_NAME, 'r', encoding='utf8')
fho         = open(BOW_PATH + str(BOW_QTY) + TRN_NAME, 'w', encoding='utf8')
for line in fh:
    comment, sentiment = line.strip().split(',')

    # Initialize new dict to represent word-frequency
    wf      = {w: 0 for w in top_n}

    # For each occurence of a word in top_n, increment its count by 1
    for w in comment.split(' '):
        w = w.strip()
        if w in wf.keys(): wf[w] += 1

    for v in wf.values():
        fho.write(f"{v},")
    fho.write(f"{sentiment}\n")
fh.close()
fho.close()

# Repeat for testing set
fh          = open(CMB_PATH + TST_NAME, 'r', encoding='utf8')
fho         = open(BOW_PATH + str(BOW_QTY) + TST_NAME, 'w', encoding='utf8')
for line in fh:
    comment, sentiment = line.strip().split(',')

    # Initialize new dict to represent word-frequency
    wf      = {w: 0 for w in top_n}

    # For each occurence of a word in top_n, increment its count by 1
    for w in comment.split(' '):
        w = w.strip()
        if w in wf.keys(): wf[w] += 1

    for v in wf.values():
        fho.write(f"{v},")
    fho.write(f"{sentiment}\n")
fh.close()
fho.close()