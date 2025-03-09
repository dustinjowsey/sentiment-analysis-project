from sklearn.base import clone
import threading
import math
# Generate splits for k-fold validation given the whole training set and its labels
def create_splits(k, X, y):
    remainder       = len(X) % k
    size            = math.floor(len(X) / k)
    # The remainder is an excess from an even division. These samples need to be divided into each group.
    prev            = 0
    end             = 0
    Xsplits, ysplits    = [], []
    for i in range(1, k+1):
        # The first (remainder) splits get 1 extra.
        end         += size if i > remainder else size + 1
        Xsplits.append(X[prev:end])
        ysplits.append(y[prev:end])
        prev        = end
    return Xsplits, ysplits

# Function to run a single experiment and return the accuracy
def run_exp(X_train, y_train, X_test, y_test, mlalgorithm, resultlist):
    mlalgorithm.fit(X_train, y_train)
    score = mlalgorithm.score(X_test, y_test)
    resultlist.append(score)


def K_Fold_Threaded(k, X, y, mlalgorithm):
    Xsplits, ysplits = create_splits(k, X, y)
    results = []
    threads = []
    for i in range(k):

        # create training and validation sets for this fold
        cur_X_train, cur_y_train    = [], []
        cur_X_test, cur_y_test      = Xsplits[i], ysplits[i]
        for j in range(k):
            if i == j: continue
            cur_X_train += Xsplits[j]
            cur_y_train += ysplits[j]
        
        # create a thread to run this instance of k-fold validation
        t = threading.Thread(target=run_exp, 
                            args=(cur_X_train, cur_y_train, cur_X_test, cur_y_test, clone(mlalgorithm), results))
        threads.append(t)
        t.start()
    
    # wait for all threads to finish
    for t in threads:
        t.join()
    
    return sum(results)/k

