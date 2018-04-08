from sklearn import preprocessing,metrics
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np

def collection_values_to_array(dataset):
    dataset = np.array(dataset)
    new_dataset = []
    for row in dataset:
        row_array = np.array(eval(row[0]))
        new_dataset.append(row_array)

    return np.array(new_dataset)

def print_accuracy(title, datasetY, predictions):
    print(title)

    print("accuracy: ", metrics.accuracy_score(datasetY, predictions))
    print("precision: ", metrics.precision_score(datasetY, predictions))
    print("recall: ", metrics.recall_score(datasetY, predictions))
    print("f1: ", metrics.f1_score(datasetY, predictions))
    print("area under curve (auc): ", metrics.roc_auc_score(datasetY, predictions))

def applySVM(dftrain,dftest):
    # print ("dftrain:",dftrain.head(5))
    # print ("dftest: ",dftest.head(5))
    data_train = dftrain.drop(["eventTime"], axis=1)
    data_test = dftest.drop(["eventTime"], axis=1)
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled_train = min_max_scaler.fit_transform(data_train)
    np_scaled_test = min_max_scaler.fit_transform(data_test)
    clf = OneClassSVM(nu=0.95,kernel='rbf',verbose=True,gamma=0.1,random_state=False)  # nu=0.95 * outliers_fraction  + 0.05
    clf.fit(np_scaled_train)
    y_pred_train = clf.predict(np_scaled_train)
    y_pred_test = clf.predict(np_scaled_test)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    for index,rows in data_train.iterrows():
        if y_pred_train[index] == -1:
            print ("Errorneous Anamoly Detected in train set: ",rows)
    for index, rows in data_test.iterrows():
        if y_pred_test[index] == -1:
            print("Errorneous Anamoly Detected in test set: ", rows)
    # Visualize
    plt.title("Novelty Detection")
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np_scaled_train, 'ro', np_scaled_test, 'g^')

    plt.subplot(212)
    plt.plot(y_pred_train, 'ro', y_pred_test, 'g^')
    plt.xlabel(
      "Anomalies in training set: %d/%d; Anomalies in test set: %d/%d;"
        % (n_error_train, data_train.shape[0], n_error_test, data_test.shape[0]))
    plt.show()


    # Display accuracy on validation set
    # print_accuracy("Validation", np_scaled_test, y_pred_test)