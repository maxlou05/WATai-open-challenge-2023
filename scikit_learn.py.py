import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

label_map = {
    0: "bike",
    1: "cabinet"
}

with np.load("beginner_data_cml.npz") as data:
    X_train = data["train_images"]  # Shape should be (8981, 64, 64, 3) (8981 picutres of 64x64 values x3 colours)
    y_train = data["train_labels"]  # Shape should be (8981, )
    X_test = data["test_images"]    # Shape should be (1800, 64, 64, 3)

# Make predictions with the model on an unshuffled test dataset
def generate_csv(model, test_data, file_name):
    predictions = model.predict(test_data)

    df = pd.DataFrame({
        "Index": np.arange(test_data.shape[0]),  # This creates a list of integers from 0 to the lenght of test_data (number of pics)
        "Label": predictions
    })
    df.to_csv(file_name, index=False)

def test_acc_50(model, test_data):
    predictions = model.predict(test_data)

    truth = pd.read_csv("first_50.csv")
    counter = 0
    for i in range(50):
        if(predictions[i] == truth["Label"][i]):
            counter += 1
    
    return counter/50

def test_acc_50_2(data_file):
    df = pd.read_csv(data_file)

    truth = pd.read_csv("first_50.csv")
    counter = 0
    for i in range(50):
        if(df["Label"][i] == truth["Label"][i]):
            counter += 1
    
    return counter/50

# fig, axs = plt.subplots(2, 4, figsize=(7, 5))
# # Loops through label 0 (bike) and 1 (cabinet)
# for label in range(2):
#     for col in range(4):
#         i_train = np.argwhere(y_train == label)[col][0]  # Take the first 4 elements that have label bike (i assume [0] gives the index)
#         axs[label, col].imshow(X_train[i_train])
#         axs[label, col].set_title(label_map[label])

# # Disables the axis
# # for ax in axs.flatten():
# #     ax.axis("off")
# plt.tight_layout()
# plt.show()
print(test_acc_50_2("hist-gradient-tree.csv"))
fig, axs = plt.subplots(5, 10, figsize=(15, 10))
# Loops through label 0 (bike) and 1 (cabinet)
for label in range(5):
    for col in range(10):
        axs[label, col].imshow(X_test[label*10+col])

# Disables the axis
# for ax in axs.flatten():
#     ax.axis("off")
plt.tight_layout()
plt.show()

# Reshape data (flatten the picture)
X_train2 = np.reshape(X_train, (8981, 12288))  # 64 * 64 * 3 = 12288
X_test2 = np.reshape(X_test, (1800, 12288))

print("x train shape: ", X_train.shape)
print("x train2 shape: ", X_train2.shape)
print("y train shape: ", y_train.shape)
print("x test shape ", X_test.shape)
# print("a sample pic data", X_test[0])
print()

# # Using logistic regression
# # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
# from sklearn.linear_model import LogisticRegression  # Uses one vs rest mode?

# model = LogisticRegression(max_iter=2000)
# start = time.time()
# model.fit(X_train2, y_train)
# end = time.time()
# accuracy = model.score(X_train2, y_train)
# print("logsitic regression - time taken:", end-start, "s, accuracy:", accuracy)
# generate_csv(model, X_test2, "logistic-regression-1000.csv")

# Notes:
# max_iter=100: 89%
# max_iter=1000: acc=100%

# # Using the decision tree method
# # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.score
# from sklearn import tree

# # clf is the classifier, it will learn the rules based on the decision tree method
# clf = tree.DecisionTreeClassifier(max_depth=15, random_state=0)  # This takes a long time lol
# start = time.time()
# clf = clf.fit(X_train2, y_train)  # This just tells it to 'learn'
# end = time.time()
# accuracy = clf.score(X_train2, y_train)
# print("decision tree - time taken:", end-start, "s, accuracy:", accuracy)
# print("test accuracy: ", test_acc_50(clf, X_test2))
# generate_csv(clf, X_test2, "decision-tree-15-0.csv")

# Notes
# max_depth=10: acc=91%


# # Using N nearest neighbours
# from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier(n_neighbors=100, leaf_size=100, p=10)
# start = time.time()
# model = model.fit(X_train2, y_train)
# end = time.time()
# accuracy = model.score(X_train2, y_train)
# print("k_neighbors - time taken:", end-start, "s, accuracy:", accuracy)
# generate_csv(model, X_test2, "k_neighbors.csv")

# # Using SVC 
# from sklearn.svm import SVC

# model = SVC(kernel='sigmoid', max_iter=20)
# start = time.time()
# model = model.fit(X_train2, y_train)
# end = time.time()
# accuracy = model.score(X_train2, y_train)
# print("SVC_sigmoid - time taken:", end-start, "s, accuracy:", accuracy)
# generate_csv(model, X_test2, "svc_sigmoid.csv")

# model = SVC(kernel='rbf', max_iter=20)
# start = time.time()
# model = model.fit(X_train2, y_train)
# end = time.time()
# accuracy = model.score(X_train2, y_train)
# print("SVC_rbf - time taken:", end-start, "s, accuracy:", accuracy)
# generate_csv(model, X_test2, "svc_rbf.csv")

# # Naive Bayes
# from sklearn.naive_bayes import GaussianNB

# model = GaussianNB(var_smoothing=1e-12)
# start = time.time()
# model = model.fit(X_train2, y_train)
# end = time.time()
# accuracy = model.score(X_train2, y_train)
# print("gaussian_bayes - time taken:", end-start, "s, accuracy:", accuracy)
# generate_csv(model, X_test2, "gaussian_bayes.csv")

# # Gaussian Process
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.gaussian_process import GaussianProcessClassifier

# model = GaussianProcessClassifier(1.0*RBF(1.0), random_state=42, max_iter_predict=1)
# start = time.time()
# model = model.fit(X_train2, y_train)
# end = time.time()
# accuracy = model.score(X_train2, y_train)
# print("gaussian - time taken:", end-start, "s, accuracy:", accuracy)
# generate_csv(model, X_test2, "gaussian.csv")


# Using boosted tree methods
# https://scikit-learn.org/stable/modules/ensemble.html

# Gradient boosted tree
from sklearn.ensemble import GradientBoostingClassifier

# n_estimators/max_iter is number of trees to use
clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, max_depth=3)
start = time.time()
clf = clf.fit(X_train2, y_train)  # This just tells it to 'learn'
end = time.time()
accuracy = clf.score(X_train2, y_train)
print("gradient tree - time taken:", end-start, "s, accuracy:", accuracy)
print("test accuracy: ", test_acc_50(clf, X_test2))
generate_csv(clf, X_test2, "gradient-tree.csv")

from sklearn.ensemble import HistGradientBoostingClassifier  # A faster version of the above

clf = HistGradientBoostingClassifier(learning_rate=0.01, max_iter=1000, max_depth=None)
start = time.time()
clf = clf.fit(X_train2, y_train)  # This just tells it to 'learn'
end = time.time()
accuracy = clf.score(X_train2, y_train)
print("histogram gradient tree - time taken:", end-start, "s, accuracy:", accuracy)
print("test accuracy: ", test_acc_50(clf, X_test2))
generate_csv(clf, X_test2, "hist-gradient-tree.csv")

# Random Forest (randomized decision tree, combo of random forest and extra tree)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500, max_depth=None)
start = time.time()
clf = clf.fit(X_train2, y_train)  # This just tells it to 'learn'
end = time.time()
accuracy = clf.score(X_train2, y_train)
print("random forest - time taken:", end-start, "s, accuracy:", accuracy)
print("test accuracy: ", test_acc_50(clf, X_test2))
generate_csv(clf, X_test2, "random-forest.csv")