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

MAX_OF_LABEL = 9000

TRAIN_PATH  = "../data/combined/BoW/5000train.csv"
TEST_PATH   = "../data/combined/BoW/5000test.csv"
fh1 = open(TRAIN_PATH)
fh2 = open(TEST_PATH)
######################
#  Data Preparation  #
######################

# Load training data
X, y = [], []
labels = {-1: 0, 0: 0, 1: 0}

for line in fh1:
    ls      = line.strip().split(',')
    ls      = [int(entry) for entry in ls]
    label   = ls[-1]

    # Get an equal number of examples of each label
    if labels[label] >= MAX_OF_LABEL: continue
    labels[label] += 1

    X.append(ls[:5000]) # Only first 1000 features, as there are 5000 now
    y.append(label)

# Show distribution of labels
print(f"Distribution of labels\n"
      f" Negative: {labels[-1]}\n"
      f" Neutral : {labels[0]}\n"
      f" Positive: {labels[1]}")

# Shuffle each list the same way using the same seed
random.seed(474)
random.shuffle(X)
random.seed(474)
random.shuffle(y)

# Create training and validation sets
div    = math.floor(1 * len(X))
X_train, X_valid = X[:div], X[div:]
y_train, y_valid = y[:div], y[div:]

print("Training set size\n"
      f" {len(X_train)}")
print("Validation set size\n"
      f" {len(X_valid)}")



# Load test data
X_test, y_test = [], []
labels = {-1: 0, 0: 0, 1: 0}
for line in fh2:
    ls      = line.strip().split(',')
    ls      = [int(entry) for entry in ls]
    label   = ls[-1]

    # Get an equal number of examples of each label
    if labels[label] >= 2000: continue
    labels[label] += 1

    X_test.append(ls[:5000]) # Only first 1000 features, as there are 5000 now
    y_test.append(label)

print(f"Distribution of labels\n"
      f" Negative: {labels[-1]}\n"
      f" Neutral : {labels[0]}\n"
      f" Positive: {labels[1]}")

######################
#   Experimentation  #
######################

"""
print(len(X_train[0]))
# Train on the neural network and score it for accuracy
basic_mlp = sklearn.neural_network.MLPClassifier(alpha=0.01, hidden_layer_sizes=(150,), max_iter=200, random_state=0, learning_rate_init=0.000075)
basic_mlp.fit(X_train, y_train)
tscore = basic_mlp.score(X_train, y_train)
vscore = basic_mlp.score(X_valid, y_valid)

# Print training and test accuracies
print(f"Accuracies\n"
      f" Training    : {tscore}\n"
      f" Vallidation : {vscore}")

exit()
"""
"""
# 0.9927777777777778
# 0.6103703703703703
# The training accuracy is really high. We are overfitting. Adjusting the alpha helps with this.
# The test accuracy is almost twice as good as random guessing. Nice. This will improve with tuning.
"""
"""
# Here's an example of using the threaded K-Fold cross validation helper I made :)
some_mlp = sklearn.neural_network.MLPClassifier(random_state=0, learning_rate_init=0.001)
avg_acc = helpers.K_Fold_Threaded(3, X_train, y_train, some_mlp)
print(avg_acc)
some_mlp = sklearn.neural_network.MLPClassifier(random_state=0, learning_rate_init=0.0001)
avg_acc = helpers.K_Fold_Threaded(3, X_train, y_train, some_mlp)
print(avg_acc)

exit()
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


# Load dataset
#dataset_path = TRAIN_PATH  # Ensure the file is in the same directory
#df = pd.read_csv(dataset_path)

# Extract features and labels (assuming the last column is the label)
#X = df.iloc[:, :1000].values  # First 1000 columns
#y = df.iloc[:, -1].values   # The last column as labels

# Define parameter grid
param_grid = {
    "alpha": np.arange(0.01, 0.1, 0.02),
    "hidden_layer_sizes": [(150,), (200,)], # medium, large hidden layers
    "learning_rate_init": [0.000075]
}

# Initialize MLP Classifier
mlp = MLPClassifier(max_iter=200, random_state=0)

# Define K-Fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=0)

# Perform GridSearch with K-Fold validation
grid_search = GridSearchCV(mlp, param_grid, cv=cv, scoring='accuracy', n_jobs=6, verbose=1)
grid_search.fit(X_train, y_train)

# Save results 
best_params = grid_search.best_params_
best_score = grid_search.best_score_
joblib.dump(grid_search, "grid_search_results.pkl")

# Print results
print("Best Parameters:", best_params)
print("Best Score:", best_score)

exit()

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
final_model = MLPClassifier(alpha=0.09, hidden_layer_sizes=(200,), max_iter=200, random_state=0, learning_rate_init=0.000075)
final_model.fit(X_train, y_train)

test_accuracy = final_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

##########Test Accuracy: 0.6503703703703704######## need to run this again