This part of the github is intended to provide a minimal basis for processing our data.

Provided that you have the raw.csv files in each of data/YouTube, data/Reddit, and data/Twitter, 
the following sequence of commands will pre-process the data:

python3 filter_sentiment.py
python3 partition.py
python3 combine.py

For our BoW NLP, run the following:

python3 topn.py
python3 wf.py



