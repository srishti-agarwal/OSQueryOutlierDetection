from sklearn import preprocessing,metrics
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import custom_function as cf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from win_preprocess import processData
from lin_preprocess import processLData

def collection_values_to_array(dataset):
    dataset = np.array(dataset)
    new_dataset = []
    for row in dataset:
        row_array = np.array(eval(row[0]))
        new_dataset.append(row_array)

    return np.array(new_dataset)

def print_accuracy(title, datasetY, predictions):
    print(title)
    print datasetY.shape
    print predictions.shape
    print("accuracy: ", metrics.accuracy_score(datasetY, predictions))
    print("precision: ", metrics.precision_score(datasetY, predictions))
    print("recall: ", metrics.recall_score(datasetY, predictions))
    print("f1: ", metrics.f1_score(datasetY, predictions))
    print("area under curve (auc): ", metrics.roc_auc_score(datasetY, predictions))

def applySVM():
    dftrain, dftest = load_features()
    dftrain = dftrain.fillna(0)
    dftest = dftest.fillna(0)

    data_train = dftrain.drop(["eventTime"], axis=1)
    data_test = dftest.drop(["eventTime"], axis=1)
    data_train = data_train[(data_train.T != 0).any()]
    data_test = data_test[(data_test.T != 0).any()]

    min_max_scaler = preprocessing.StandardScaler()
    np_scaled_train = min_max_scaler.fit_transform(data_train)
    np_scaled_test = min_max_scaler.fit_transform(data_test)

    clf = OneClassSVM(nu=0.01,kernel='rbf',verbose=True,gamma=0.1,random_state=False)
    clf.fit(np_scaled_train)
    y_pred_train = clf.predict(np_scaled_train)
    y_pred_test = clf.predict(np_scaled_test)
    n_error_test = y_pred_test[y_pred_test == -1].size

    #collect all the anomalous points
    for index, rows in data_test.iterrows():
        if y_pred_test[index] == -1:
            data_test.iloc[index].to_csv("data/anomalous.csv")
    # Visualize
    plt.title("Novelty Detection")
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np_scaled_train, 'bo', np_scaled_test, 'y^')
    plt.subplot(212)
    plt.plot(y_pred_train, 'go', y_pred_test, 'r^')
    plt.ylabel("Labels")
    plt.xlabel(
      "Anomalies in test set: %s;"
        % (str(n_error_test*100/data_test.shape[0])+"%"))
    plt.show()

def applyPCAModel():
    dftrain, dftest = load_features()
    dftrain = dftrain.fillna(0)

    data_train = dftrain.drop(["eventTime"], axis=1)
    # print ("X_train: ",data.head(5))
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data_train)
    data_train = pd.DataFrame(np_scaled)
    # reduce to 2 importants features
    pca = PCA(n_components=2)
    data_train = pca.fit_transform(data_train)
    # standardize these 2 new features
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data_train)
    data_train = pd.DataFrame(np_scaled)

    #After processing the data fit the model for our testing logs
    print ("find the best value of clusters using elbow loss curve")
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(data_train) for i in n_cluster]
    scores = [kmeans[i].score(data_train) for i in range(len(kmeans))]

    fig, ax = plt.subplots()
    ax.plot(n_cluster, scores)
    plt.show()

    dftrain['cluster'] = kmeans[10].predict(data_train)
    dftrain['principal_feature1'] = data_train[0]
    dftrain['principal_feature2'] = data_train[1]
    print (dftrain['cluster'].value_counts())
    fig, ax = plt.subplots()
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'pink', 4: 'black', 5: 'orange', 6: 'cyan', 7: 'yellow', 8: 'brown',
              9: 'purple', 10: 'white', 11: 'grey', 12: 'lightblue', 13: 'lightgreen', 14: 'darkgrey'}
    ax.scatter(dftrain['principal_feature1'], dftrain['principal_feature2'], c=dftrain["cluster"].apply(lambda x: colors[x]))
    plt.show()
    distance = cf.getDistanceByPoint(data_train, kmeans[10])
    outlier_percentage = 0.01
    number_of_outliers = int(outlier_percentage*len(distance))
    threshold = distance.nlargest(number_of_outliers).min()
    dftrain['anomaly_PCA'] = (distance >= threshold).astype(int)
    print(dftrain['anomaly_PCA'].value_counts())
    fig, ax = plt.subplots()
    colors = {0: 'blue', 1: 'red'}
    print ("the outliers in red")
    ax.scatter(dftrain['principal_feature1'], dftrain['principal_feature2'], c=dftrain["anomaly_PCA"].apply(lambda x: colors[x]))
    plt.show()

def create_and_save_win():
    #creating the train feature vector
    train_data = processData()
    train_data.processLogs('train.log')
    train_vec = train_data.dataframe

    #creating the test feature vector
    test_data = processData()
    test_data.processLogs('test.log')
    test_vec = test_data.dataframe

    #save the test and train features
    print ("Saving train and test feature vectors")
    train_vec.to_csv('data/train_vec.csv', index=False)
    test_vec.to_csv('data/test_vec.csv', index=False)

def create_and_save_lin():
    #creating the train feature vector
    train_data = processLData()
    train_data.processLogs('train_lin.log')
    train_vec = train_data.dataframe

    #creating the test feature vector
    test_data = processLData()
    test_data.processLogs('test_lin.log')
    test_vec = test_data.dataframe

    #save the test and train features
    print ("Saving train and test feature vectors")
    train_vec.to_csv('data/train_vec.csv', index=False)
    test_vec.to_csv('data/test_vec.csv', index=False)

def load_features():
    print ("Loading train features")
    train_features = pd.read_csv('data/train_vec.csv')
    print ("Loaded train features")

    print ("Loading test features")
    test_features = pd.read_csv('data/test_vec.csv')
    print ("Loaded test features")
    return train_features, test_features

def predict_anomalies(algo):
    algo = int(algo)
    options = {
        1 : applySVM,
        2 : applyPCAModel,
    }
    options[algo]()
