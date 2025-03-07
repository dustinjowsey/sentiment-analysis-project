import math
import random
import sklearn
import matplotlib.pyplot as plt


MAX_OF_LABEL = 2250


fh = open("../data/YouTube/top1000_wf.csv")

######################
#  Data Preparation  #
######################

x, y = [], []
i = 0 
labels = {-1: 0, 0: 0, 1: 0}

for line in fh:
    ls      = line.strip().split(',')
    ls      = [int(entry) for entry in ls]
    label   = ls[-1]

    # Get an equal number of examples of each label
    if labels[label] >= MAX_OF_LABEL: continue
    labels[label] += 1

    x.append(ls[:-1])
    y.append(label)

# Show distribution of labels
print(f"Distribution of labels\n"
      f" Negative: {labels[-1]}\n"
      f" Neutral : {labels[0]}\n"
      f" Positive: {labels[1]}")

# Shuffle each list the same way using the same seed
random.seed(474)
random.shuffle(x)
random.seed(474)
random.shuffle(y)

# Create training and test sets
div    = math.floor(0.8 * len(x))
X_train, X_test = x[:div], x[div:]
y_train, y_test = y[:div], y[div:]

print("Training set size\n"
      f" {len(X_train)}")
print("Testing set size\n"
      f" {len(X_test)}")


######################
#   Experimentation  #
######################

# Train on the neural network and score it for accuracy
basic_mlp = sklearn.neural_network.MLPClassifier(max_iter=400, random_state=0)
basic_mlp.fit(X_train, y_train)
tscore = basic_mlp.score(X_train, y_train)
vscore = basic_mlp.score(X_test, y_test)

# Print training and test accuracies
print(f"Accuracies\n"
      f" Training: {tscore}\n"
      f" Testing : {vscore}")
# 0.9927777777777778
# 0.6103703703703703
# The training accuracy is really high. We are overfitting. Adjusting the alpha helps with this.
# The test accuracy is almost twice as good as random guessing. Nice. This will improve with tuning.

"""
TO DO:
    Tune the model on (two?) parameters.
    Perhaps the alpha on a logarithmic range from 10^-4 up to 10^2
    and another. 
    
    Don't do K-Fold yet. Just use a validation set for these configurations.

    Find an optimal configuration like we did for A2, then try slightly varying 
    the values in the optimal configuration. For these slight variations, do K-Fold
"""