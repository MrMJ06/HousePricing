import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold


def preprocess_data(dataframe, corr_thr, max_dummies_num, max_nan_ratio):

    for header in list(dataframe):
        if dataframe[header].isnull().sum() > len(dataframe[header])*max_nan_ratio:
            dataframe.__delitem__(header)
        else:
            new_values = []
            if dataframe[header].dtype == np.object_:
                new_values = pd.get_dummies(dataframe[header], prefix='is')

                if len(set(new_values)) > max_dummies_num:
                    dataframe.__delitem__(header)
                else:
                    dataframe[header] = new_values

            if header in dataframe.keys() and 'SalePrice' in dataframe.keys() and np.abs(dataframe.corr()['SalePrice'][header]) < corr_thr:
                dataframe.__delitem__(header)
            else:
                if header in dataframe.keys() and len(new_values) > 0:
                    for new_class in new_values:
                        dataframe[new_class] = new_values[new_class]
                        if 'SalePrice' in dataframe.keys() and np.abs(dataframe.corr()['SalePrice'][new_class]) < corr_thr:
                            dataframe.__delitem__(new_class)
                    dataframe.__delitem__(header)

    if 'SalePrice' in dataframe.keys():
        dataframe.dropna(inplace=True)
        #dataframe.loc[:, dataframe.columns != 'SalePrice'] = (dataframe.loc[:, dataframe.columns != 'SalePrice'] - dataframe.loc[:,dataframe.columns != 'SalePrice'].mean()) / dataframe.loc[:, dataframe.columns != 'SalePrice'].std()
    else:
        dataframe.fillna(dataframe.mean(), inplace=True)
        #dataframe = (dataframe - dataframe.mean())/dataframe.std()


    return dataframe

def train(data_norm, data_label, it):
    best_accuracy = 0
    best_classifier = None

    pd.set_option("display.max_rows", 500)
    for i in range(0, it):
        train_data_input = data_norm

        # data_train, data_test, labels_train, labels_test = train_test_split(train_data_input, train_data_label, test_size=0.20, stratify=train_data_label)
        classifier = MLPRegressor(hidden_layer_sizes=[15], max_iter=1000, learning_rate_init=0.001)

        kf = KFold(n_splits=10, shuffle=True)
        mean_accuracy = 0
        partitions = kf.split(train_data_input, data_label)
        for train_index, test_index in partitions:
            X_train, X_test = np.array(train_data_input)[train_index], np.array(train_data_input)[test_index]
            y_train, y_test = np.array(data_label)[train_index], np.array(data_label)[test_index]

            classifier.fit(X_train, y_train)
            accuracy = classifier.score(X_test, y_test)
            mean_accuracy += accuracy / 10

        if best_accuracy < mean_accuracy:
            best_accuracy = mean_accuracy
            best_classifier = classifier
            print('Iteration: ' + str(i) + ' ' + str(best_accuracy))
    return best_classifier

pd.set_option('max_columns', 50)
pd.set_option('max_rows', 500)

train_data = pd.read_csv('resources/train.csv', na_values='NA')
test_data = pd.read_csv('resources/test.csv', na_values='NA')
labels = test_data["Id"]
train_data.__delitem__('Id')
test_data.__delitem__('Id')

train_data = preprocess_data(train_data, 0.3, 15, 1 / 5)
test_data = preprocess_data(test_data, 0.3, 15, 1 / 5)

sales_price = train_data['SalePrice']

train_data.__delitem__('SalePrice')
test_data = test_data[list(train_data.keys())]

cls = train(train_data, sales_price, 25)

predictions = cls.predict(test_data)
dictionary = {'Id': labels, 'SalePrice': predictions}
result = pd.DataFrame(data=dictionary)

result.to_csv(path_or_buf="results.csv", index=False)
