import os
from constants import *

N_GRAM_VAL      = 2
# Return a list of the n-grams in the sequence
def n_grammify(example, n):
    exsplit = [e.strip() for e in example.split(' ') if e.strip() != '']
    return [' '.join(exsplit[i-2:i]) for i in range(n, len(exsplit) + 1)]

# Find all n-grams and their frequency
fh      = open(CMB_PATH + TRN_NAME, 'r', encoding='utf-8')
all_ngrams = {}
for line in fh:
    n_grams = n_grammify(line.split(',')[0], N_GRAM_VAL)
    for g in n_grams: 
        if g not in all_ngrams:
            all_ngrams[g] = 1
            continue
        all_ngrams[g] += 1
fh.close()

# Save most common n-grams
ngs = {k: v for k, v in sorted(all_ngrams.items(), key=lambda item: item[1], reverse=True)}
keys = list(ngs.keys())


print(f"{len(keys)} unique {N_GRAM_VAL}-grams")
try: os.mkdir(NGM_PATH)
except: pass

fh = open(NGM_PATH + NGM_TXT, "w", encoding='utf-8')
for i in range(NGM_QTY):
    fh.write(f"{keys[i]}\n")
fh.close()

# Turn training set into feature-label vectors of top n-grams
fhi         = open(CMB_PATH + TRN_NAME, 'r', encoding='utf-8')
fho         = open(NGM_PATH + str(NGM_QTY) + TRN_NAME, 'w', encoding='utf-8')
for line in fhi:
    comment, sentiment = line.strip().split(',')
    ngf     = {ng: 0 for ng in keys[:NGM_QTY]}
    
    for ng in n_grammify(comment, N_GRAM_VAL):
        if ng in ngf: ngf[ng] += 1
    
    for v in ngf.values():
        fho.write(f"{int(v)},")
    fho.write(f"{int(sentiment)}\n")
fhi.close()
fho.close()

# Do the same for the test set
fhi         = open(CMB_PATH + TST_NAME, 'r', encoding='utf-8')
fho         = open(NGM_PATH + str(NGM_QTY) + TST_NAME, 'w', encoding='utf-8')
for line in fhi:
    comment, sentiment = line.strip().split(',')
    ngf     = {ng: 0 for ng in keys[:NGM_QTY]}
    
    for ng in n_grammify(comment, N_GRAM_VAL):
        if ng in ngf: ngf[ng] += 1
    
    for v in ngf.values():
        fho.write(f"{int(v)},")
    fho.write(f"{int(sentiment)}\n")
fhi.close()
fho.close()