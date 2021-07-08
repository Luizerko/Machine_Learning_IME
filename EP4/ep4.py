name = "Luis Vitor Pedreira Iten Zerkowski & Ígor de Andrade Barberino"  # write YOUR NAME

honorPledge = "I affirm that I have not given or received any unauthorized " \
              "help on this assignment, and that this work is my own.\n"

print("\nName: ", name)
print("\nHonor pledge: ", honorPledge)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train_ori, y_train_ori), (X_test_ori, y_test_ori) = mnist.load_data()

print(X_train_ori.shape, y_train_ori.shape)
print(X_test_ori.shape, y_test_ori.shape)

labels = ["%s"%i for i in range(10)]

unique, counts = np.unique(y_train_ori, return_counts=True)
uniquet, countst = np.unique(y_test_ori, return_counts=True)

fig, ax = plt.subplots()
rects1 = ax.bar(unique - 0.2, counts, 0.25, label='Train')
rects2 = ax.bar(unique + 0.2, countst, 0.25, label='Test')
ax.legend()
ax.set_xticks(unique)
ax.set_xticklabels(labels)

plt.title('MNIST classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

fig, ax = plt.subplots(2, 3, figsize = (9, 6))

for i in range(6):
    ax[i//3, i%3].imshow(X_train_ori[i], cmap='gray')
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_title("Class: %d"%y_train_ori[i])
    
plt.show()

# Reduce the image size to its half 
X_train = np.array([image[::2, 1::2] for image in X_train_ori])
X_test  = np.array([image[::2, 1::2] for image in X_test_ori])

y_train = y_train_ori
y_test = y_test_ori

fig, ax = plt.subplots(2, 3, figsize = (9, 6))

for i in range(6):
    ax[i//3, i%3].imshow(X_train[i], cmap='gray')
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_title("Class: %d"%y_train_ori[i])
    
plt.show()

X_train = (X_train/255.0).astype('float32').reshape((60000,14*14))
X_test = (X_test/255.0).astype('float32').reshape((10000,14*14))

print(X_train.dtype)
print(X_test.dtype)

print("\nShape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)

print("\nMinimum value in X_train:", np.amin(X_train))
print("Maximum value in X_train:", np.amax(X_train))

print("\nMinimum value in X_test:", np.amin(X_test))
print("Maximum value in X_test:", np.amax(X_test))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import time

#train and validation split 70/30
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.3, random_state=10)

unique, counts = np.unique(y_train, return_counts=True)
uniquev, countsv = np.unique(y_val, return_counts=True)

counts = counts/counts.sum()
countsv = countsv/countsv.sum()

fig, ax = plt.subplots()
rects1 = ax.bar(unique - 0.2, counts, 0.25, label='Train')
rects2 = ax.bar(uniquev + 0.2, countsv, 0.25, label='Validation')
ax.legend()
ax.set_xticks(unique)
ax.set_xticklabels(labels)

plt.title('MNIST classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train,
                                                                          test_size=0.15, random_state=10)

#Logistic regression classifier
start_time = time.time()
best_accuracy = 0
best_f1 = 0
best_log_loss = 100000
best_logistic_regression_clf = None
opt_iter = -1
opt_C = -1
arg_iter = -1
arg_C = -1
accuracy_hist = []
f1_score_hist = []
log_loss_hist = []

#Fitting the model while selecting hyperparameters
cont_i = 0
cont_j = 0
for i in np.linspace(500, 700, 3):
    for j in range(1, 4):
        logistic_clf = LogisticRegression(random_state=10, max_iter=i,
                                        tol=0.0001, C=j).fit(X_train_train, y_train_train)
        
        predictions = logistic_clf.predict(X_train_val)
        predictions_prob = logistic_clf.predict_proba(X_train_val)

        accuracy_val = logistic_clf.score(X_train_val, y_train_val)
        f1_score_val = f1_score(y_true=y_train_val, y_pred=predictions, labels=np.unique(y_train_val), average='macro')
        log_loss_val = log_loss(y_true=y_train_val, y_pred=predictions_prob, labels=np.unique(y_train_val))

        accuracy_hist.append((i, j, accuracy_val))
        f1_score_hist.append((i, j, f1_score_val))
        log_loss_hist.append((i, j, log_loss_val))

        if accuracy_val+0.001 > best_accuracy and f1_score_val+0.001 > best_f1 and log_loss_val-0.001 < best_log_loss:
            best_accuracy = accuracy_val
            best_f1 = f1_score_val
            best_log_loss = log_loss_val
            logistic_regression_best_clf = logistic_clf
            
            opt_iter = i
            opt_C = j

            arg_iter = cont_i
            arg_C = cont_j
        
        cont_j += 1
    cont_j = 0
    cont_i += 1

print("Best parameters:")
print("Number of iterations: {}".format(opt_iter))
print("C value: {}".format(opt_C))
print("Accuracy in the training set: {}".format(best_accuracy))
print("F1 score in the training set: {}".format(best_f1))
print("Log loss in the training set: {}".format(best_log_loss))

print("Confusion matrix:")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_confusion_matrix(logistic_clf, X_train_val, y_train_val, labels=np.unique(y_train_val), ax=ax)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 12))
ax[0].plot([x[0] for x in accuracy_hist[arg_C:arg_C+7:3]],
            [x[2] for x in accuracy_hist[arg_C:arg_C+7:3]])
ax[0].axis('on')
ax[0].set_title("Number of iterations x Accuracy (best C fixed):")
ax[0].set_xlabel("Number of iterations")
ax[0].set_ylabel("Accuracy")

ax[1].plot([x[1] for x in accuracy_hist[arg_iter*3:(arg_iter*3)+3]],
            [x[2] for x in accuracy_hist[arg_iter*3:(arg_iter*3)+3]])
ax[1].axis('on')
ax[1].set_title("Accuracy x C value (best number of iterations fixed):")
ax[1].set_xlabel("C value")
ax[1].set_ylabel("Accuracy")

plt.show()

time_logistic = (time.time() - start_time)
print("Execution time: {}".format(time_logistic))

#Neural network classifier
start_time = time.time()
best_accuracy = 0
best_f1 = 0
neural_net_best_clf = None
opt_iter = -1
opt_batch = -1
opt_lr = -1
accuracy_hist = []
f1_score_hist = []

#Fitting the model while selecting hyperparameters
for i in np.linspace(50, 60, 2):
    for j in np.linspace(64, 128, 2):
        for z in np.geomspace(1e-3, 1e-2, 2):
            neural_net_clf = MLPClassifier(random_state=10, max_iter=i, tol=0.0001,
                                        hidden_layer_sizes=(7, 256), learning_rate_init=z,
                                        learning_rate='invscaling', validation_fraction=0.1,
                                        batch_size=j, verbose=True,
                                        early_stopping=True).fit(X_train_train, y_train_train)

            predictions = neural_net_clf.predict(X_train_val)

            accuracy_val = neural_net_clf.score(X_train_val, y_train_val)
            f1_score_val = f1_score(y_true=y_train_val, y_pred=predictions, labels=np.unique(y_train_val), average='macro')

            accuracy_hist.append((i, j, accuracy_val))
            f1_score_hist.append((i, j, f1_score_val))

            if accuracy_val+0.001 > best_accuracy and f1_score_val+0.001 > best_f1:
                best_accuracy = accuracy_val
                best_f1 = f1_score_val
                neural_net_best_clf = neural_net_clf
                
                opt_iter = i
                opt_batch = j
                opt_lr = z

print("Best parameters:")
print("Number of iterations: {}".format(opt_iter))
print("Batch size: {}".format(opt_batch))
print("Learnin rate: {}".format(opt_lr))
print("Accuracy in the training set: {}".format(best_accuracy))
print("F1 score in the training set: {}".format(best_f1))

print("Confusion matrix:")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_confusion_matrix(logistic_clf, X_train_val, y_train_val, labels=np.unique(y_train_val), ax=ax)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].plot([x[0] for x in accuracy_hist], [x for x in range(len(accuracy_hist))])
ax[0].axis('off')
ax[0].set_title("Accuracy x Number of iterations: ")

ax[1].plot([x[1] for x in accuracy_hist], [x for x in range(len(accuracy_hist))])
ax[1].axis('off')
ax[1].set_title("Accuracy x C value: ")

plt.show()

time_nn = (time.time() - start_time)
print("Execution time: {}".format(time_nn))

#SVM classifier
start_time = time.time()
best_accuracy = 0
best_f1 = 0
svm_best_clf = None
opt_iter = -1
opt_C = -1
accuracy_hist = []
f1_score_hist = []

start_time = time.time()
for i in np.linspace(500, 600, 2):
    for j in range(1, 3):
        svm_clf = SVC(random_state=10, max_iter=i, tol=0.001,
                    C=j, kernel='rbf', verbose=False,
                    decision_function_shape='ovr').fit(X_train_train, y_train_train)

        predictions = svm_clf.predict(X_train_val)

        accuracy_val = svm_clf.score(X_train_val, y_train_val)
        f1_score_val = f1_score(y_true=y_train_val, y_pred=predictions, labels=np.unique(y_train_val), average='macro')

        accuracy_hist.append((i, j, accuracy_val))
        f1_score_hist.append((i, j, f1_score_val))

        if accuracy_val+0.001 > best_accuracy and f1_score_val+0.001 > best_f1:
            best_accuracy = accuracy_val
            best_f1 = f1_score_val
            svm_best_clf = svm_clf

            opt_iter = i
            opt_C = j
            
print("Best parameters:")
print("Number of iterations: {}".format(opt_iter))
print("C value: {}".format(opt_C))
print("Accuracy in the training set: {}".format(best_accuracy))
print("F1 score in the training set: {}".format(best_f1))

print("Confusion matrix:")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_confusion_matrix(logistic_clf, X_train_val, y_train_val, labels=np.unique(y_train_val), ax=ax)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].plot([x[0] for x in accuracy_hist], [x for x in range(len(accuracy_hist))])
ax[0].axis('off')
ax[0].set_title("Accuracy x Number of iterations: ")

ax[1].plot([x[1] for x in accuracy_hist], [x for x in range(len(accuracy_hist))])
ax[1].axis('off')
ax[1].set_title("Accuracy x C value: ")

plt.show()
time_svm = (time.time() - start_time)
print("Execution time: {}".format(time_svm))

#Tempo de execução modelos logistic_reg/neural_net/SVM no Google Colab:
#Processador -> 39/98
#Processador + GPU -> 32/79
#Processador + TPU -> 39/95

#Choosing the best model based on validation set
best_accuracy = -1
best_f1 = -1

predictions_logistic = logistic_regression_best_clf.predict(X_val)
accuracy_logistic = logistic_regression_best_clf.score(X_val, y_val)
f1_score_logistic = f1_score(y_true=y_val, y_pred=predictions_logistic, labels=np.unique(y_val), average='macro')

predictions_neural_net = neural_net_best_clf.predict(X_val)
accuracy_neural_net = neural_net_best_clf.score(X_val, y_val)
f1_score_neural_net = f1_score(y_true=y_val, y_pred=predictions_neural_net, labels=np.unique(y_val), average='macro')

predictions_svm = svm_best_clf.predict(X_val)
accuracy_svm = svm_best_clf.score(X_val, y_val)
f1_score_svm = f1_score(y_true=y_val, y_pred=predictions_svm, labels=np.unique(y_val), average='macro')

print("Logistic regression accuracy: {}".format(accuracy_logistic))
print("Logistic regression f1 score: {}".format(f1_score_logistic))

print("Neural network accuracy: {}".format(accuracy_neural_net))
print("Neural network f1 score: {}".format(f1_score_neural_net))

print("SVM accuracy: {}".format(accuracy_svm))
print("SVM f1 score: {}".format(f1_score_svm))

#Retraining model including validation set and computing E_out estimate on test set 