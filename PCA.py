from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import custom_function as cf
import pandas as pd


def applyPCAModel(dftrain,dftest):
    data_train = dftrain.drop(["eventTime"], axis=1)
    data_test = dftest.drop(["eventTime"], axis=1)
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

    np_scaled_test = min_max_scaler.fit_transform(data_test)
    data_test = pd.DataFrame(np_scaled_test)
    data_test = pca.fit_transform(data_test)
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled_test = min_max_scaler.fit_transform(data_test)
    data_test = pd.DataFrame(np_scaled_test)

    #After processing the data fit the model for our testing logs
    #find the best value of clusters using elbow loss curve
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(data_train) for i in n_cluster]
    scores = [kmeans[i].score(data_train) for i in range(len(kmeans))]

    fig, ax = plt.subplots()
    ax.plot(n_cluster, scores)
    plt.show()

    dftest['cluster'] = kmeans[10].predict(data_test)
    dftest['principal_feature1'] = data_test[0]
    dftest['principal_feature2'] = data_test[1]
    print (dftest['cluster'].value_counts())
    fig, ax = plt.subplots()
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'pink', 4: 'black', 5: 'orange', 6: 'cyan', 7: 'yellow', 8: 'brown',
              9: 'purple', 10: 'white', 11: 'grey', 12: 'lightblue', 13: 'lightgreen', 14: 'darkgrey'}
    ax.scatter(dftest['principal_feature1'], dftest['principal_feature2'], c=dftest["cluster"].apply(lambda x: colors[x]))
    plt.show()
    distance = cf.getDistanceByPoint(data_test, kmeans[10])
    number_of_outliers = int(len(distance))
    threshold = distance.nlargest(number_of_outliers).min()
    dftest['anomaly_PCA'] = (distance >= threshold).astype(int)
    print(dftest['anomaly_PCA'].value_counts())
    fig, ax = plt.subplots()
    colors = {0: 'blue', 1: 'red'}
    ax.scatter(dftest['principal_feature1'], dftest['principal_feature2'], c=dftest["anomaly_PCA"].apply(lambda x: colors[x]))
    plt.show()


# def applyMultivariate(df):
#     tr_data = df.drop(["eventTime"], axis=1)
#     mu, sigma = cf.estimateGaussian(tr_data)
#     p = cf.multivariateGaussian(tr_data,mu,sigma)
#     # p_cv = cf.multivariateGaussian(cv_data,mu,sigma)
#     # fscore, ep = cf.selectThresholdByCV(p_cv, gt_data)
#     # print(fscore, ep)
#     # outliers = np.asarray(np.where(p < ep))