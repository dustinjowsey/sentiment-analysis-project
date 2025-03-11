import math
import random
from constants import *

for p in PATHS:
    fh1 = open(p + CLN_NAME, 'r', encoding='utf-8')
    fh2 = open(p + TRN_NAME, 'w', encoding='utf-8')
    fh3 = open(p + TST_NAME, 'w', encoding='utf-8')

    # Load all examples and shuffle randomly according to a fixed seed
    examples = [line for line in fh1]
    random.seed(RND_SEED)
    random.shuffle(examples)

    trn_end    = math.ceil(TRN_RATIO*len(examples))
    for e in examples[:trn_end]:
        fh2.write(e)
    for e in examples[trn_end:]:
        fh3.write(e)

    fh1.close()
    fh2.close()
    fh3.close()