This part of the github is intended to provide a minimal basis for processing our data.

I had this written out before checkin the github for the implementations using pandas,
so I am adding it in case it is helpful to others. I find it to be simpler though maybe
not as robust.

filter.py will take the "raw" csv and remove irrelevant characters to allow us to parse each word better.

topn.py will provide a .txt containing the top n most frequently used words across all comments in the provided .csv
The value TOP_N can be changed within that file and wf will inherit it.

wf.py will create a .csv with n + 1 columns, where the first n are the frequencies of each top nth word in each comment. 
The last one is the label. Each row represents a comment.

The resulting csv of wf.py can be used in experimentation.


