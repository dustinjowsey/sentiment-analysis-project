import math
import random
import sklearn
import matplotlib.pyplot as plt
import sklearn.neural_network
import helpers
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
import joblib

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

"""
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

# Here's an example of using the threaded K-Fold cross validation helper I made :)
some_mlp = sklearn.neural_network.MLPClassifier(random_state=0)
avg_acc = helpers.K_Fold_Threaded(2, X_train, y_train, some_mlp)
print(avg_acc)



"""
TO DO:
    Gridsearch K-Fold validation on alpha and another parameter.
"""

"""
a_range = [i/2 for i in range(1, 11)]

exp1_tscores      = []
exp1_vscores      = []
for a in a_range:
      cur_mlp = sklearn.neural_network.MLPClassifier(max_iter=400, random_state=0, alpha = a)
      cur_mlp.fit(X_train, y_train)
      exp1_tscores.append(cur_mlp.score(X_train, y_train))
      exp1_vscores.append(cur_mlp.score(X_test, y_test))

print(exp1_tscores)
print(exp1_vscores)
"""

# Best alpha so far is alpha = 1.5 with acc = 0.676296
# Maybe gridsearch on alphas ranging from 0.5 to 2.5 in increments of 0.25

"""
pseudo

initalize best configuration to (p1[0], p2[0])
best acc = 0
for p1 in first parameter range
      for p2 in second paramter range
            cur_acc = result of k fold validation 
            if cur_acc > best_acc:
                  best config becomes current config
                  best acc becomes current acc
save best configuration

"""




# Load dataset
dataset_path = "top1000_wf.csv"  # Ensure the file is in the same directory
df = pd.read_csv(dataset_path)

# Extract features and labels (assuming the last column is the label)
X = df.iloc[:, :-1].values  # All columns except the last
y = df.iloc[:, -1].values   # The last column as labels

# Define parameter grid
param_grid = {
    "alpha": np.arange(0.5, 2.5, 0.25),
    "hidden_layer_sizes": [(150,), (200,)]  # medium, large hidden layers
}

# Initialize MLP Classifier
mlp = MLPClassifier(max_iter=400, random_state=0)

# Define K-Fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=0)

# Perform GridSearch with K-Fold validation
grid_search = GridSearchCV(mlp, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Save results 
best_params = grid_search.best_params_
best_score = grid_search.best_score_
joblib.dump(grid_search, "grid_search_results.pkl")

# Print results
print("Best Parameters:", best_params)
print("Best Score:", best_score)


#best score is average K-Fold validation accuracy

#param_grid = {
    #"alpha": np.arange(0.5, 2.5, 0.25),
    #"hidden_layer_sizes": [(50,), (100,)]  # Small, medium hidden layers

#Best Parameters: {'alpha': np.float64(1.5), 'hidden_layer_sizes': (100,)}
#Best Score: 0.736690569317688


#param_grid = {
    #"alpha": np.arange(0.5, 2.5, 0.25),
    #"hidden_layer_sizes": [(100,), (150,)]  # medium, large hidden layers

#Best Parameters: {'alpha': np.float64(1.5), 'hidden_layer_sizes': (100,)}
#Best Score: 0.736690569317688


#param_grid = {
    #"alpha": np.arange(0.5, 2.5, 0.25),
    #"hidden_layer_sizes": [(100,), (200,)]  # medium, large hidden layers

#Best Parameters: {'alpha': np.float64(1.5), 'hidden_layer_sizes': (100,)}
#Best Score: 0.736690569317688


#param_grid = {
    #"alpha": np.arange(0.5, 2.5, 0.25),
    #"hidden_layer_sizes": [(150,), (200,)]  # medium, large hidden layers


#Best Parameters: {'alpha': np.float64(1.0), 'hidden_layer_sizes': (200,)}
#Best Score: 0.7359843546284224


##5 FOLD##


#"alpha": np.arange(0.5, 2.5, 0.25),
#"hidden_layer_sizes": [(50,), (100,)] 

##Best Parameters: {'alpha': np.float64(1.25), 'hidden_layer_sizes': (50,)}
#Best Score: 0.7395696532290469


# "alpha": np.arange(0.5, 2.5, 0.25),
# "hidden_layer_sizes": [(100,), (150,)] 

#Best Parameters: {'alpha': np.float64(0.5), 'hidden_layer_sizes': (150,)}
#Best Score: 0.7402761453511219


# "alpha": np.arange(0.5, 2.5, 0.25),
# "hidden_layer_sizes": [(150,), (200,)] 

#Best Parameters: {'alpha': np.float64(0.5), 'hidden_layer_sizes': (150,)}
#Best Score: 0.7402761453511219


#to find the test accuracy with after choosing the best parameters to use

final_model = MLPClassifier(alpha=0.5, hidden_layer_sizes=(150,), max_iter=400, random_state=0)
final_model.fit(X_train, y_train)

test_accuracy = final_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

##########Test Accuracy: 0.6503703703703704########
