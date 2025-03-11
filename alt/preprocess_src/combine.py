import os
import random
from constants import *


# Combine training sets
examples = []
for p in PATHS:
    fh = open(p + TRN_NAME, 'r', encoding='utf-8')
    examples += [line for line in fh]
    fh.close()

try: os.mkdir(CMB_PATH)
except: pass

fho = open(CMB_PATH + TRN_NAME, 'w', encoding='utf-8')
random.seed(RND_SEED)
random.shuffle(examples)
for e in examples:
    fho.write(e)
fho.close()

# Combine testing sets
examples = []
for p in PATHS:
    fh = open(p + TST_NAME, 'r', encoding='utf-8')
    examples += [line for line in fh]
    fh.close()

fho = open(CMB_PATH + TST_NAME, 'w', encoding='utf-8')
random.seed(RND_SEED)
random.shuffle(examples)
for e in examples:
    fho.write(e)
fho.close()