RAW_NAME        = "raw.csv"
CLN_NAME        = "cleaned.csv"
TRN_NAME        = "train.csv"
TST_NAME        = "test.csv"
TRN_RATIO       = 0.8
RND_SEED        = 474

CMB_PATH        = "../data/Combined/"
PATHS           = ["../data/Youtube/",
                   "../data/Reddit/",
                   "../data/Twitter/"]


BOW_QTY         = 5000
BOW_TXT         = f"{BOW_QTY}BoW.txt"
BOW_PATH        = f"{CMB_PATH}BoW/"

NGM_QTY         = 5000
NGM_TXT         = f"{NGM_QTY}NGRAM.txt"
NGM_PATH        = f"{CMB_PATH}NGram/"
