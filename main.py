import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(corr_thr, all_data):

    print(all_data.info())
    all_data.__delitem__("GarageArea")
    all_data.__delitem__("TotRmsAbvGrd")

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType','MSSubClass', 'PoolQC'):
        all_data[col].fillna('None', inplace=True)

    for col in ('GarageYrBlt', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):
        all_data[col] = all_data[col].fillna(0)

    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

    all_data = all_data.drop(['Utilities'], axis=1)

    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    # MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

    # Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)

    # Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    all_data.__delitem__("1stFlrSF")

    all_data.fillna(all_data.median(), inplace=True)

    numeric_feats = all_data.dtypes[np.logical_and(all_data.dtypes != "object", all_data.dtypes != "str")].index
    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})

    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)

    for header in list(all_data):

        new_values = []
        if all_data[header].dtype == np.object_ or all_data[header].dtype == np.str_:
            new_values = pd.get_dummies(all_data[header], prefix='is')
            all_data[header] = new_values

        if header in all_data.keys() and len(new_values) > 0:
            for new_class in new_values:
                all_data[new_class] = new_values[new_class][0:len(train_data)]

    all_data.reset_index(drop=True, inplace=True)

    all_data.__delitem__("GrLivArea")
    # k = 15  # number of variables for heatmap
    # cols = all_data.corr().nlargest(k, 'SalePrice')['SalePrice'].index
    # cm = np.corrcoef(all_data[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
    #                  yticklabels=cols.values, xticklabels=cols.values)
    return all_data


def rmsle_cv(model, data, labels):
    kf = KFold(10, shuffle=True, random_state=42).get_n_splits(data)
    rmse= np.sqrt(-cross_val_score(model, data, labels, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


def train(data_norm, data_label, it, model):
    best_accuracy = 0
    best_classifier = None

    pd.set_option("display.max_rows", 500)
    for i in range(0, it):
        train_data_input = data_norm

        # data_train, data_test, labels_train, labels_test = train_test_split(train_data_input, train_data_label, test_size=0.20, stratify=train_data_label)
        classifier = model

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

all_data = pd.concat((train_data, test_data), sort=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data = preprocess_data(0, all_data)

sales_price = np.log(train_data['SalePrice'])

train_data.__delitem__('SalePrice')
lco = LocalOutlierFactor()

train_data = all_data[:train_data.shape[0]]
test_data = all_data[train_data.shape[0]:]

is_inlier = lco.fit_predict(train_data, sales_price)

# all_data = (all_data - all_data.mean())/all_data.std()
for n, inlier in enumerate(is_inlier):
    if inlier < 0:
        train_data.drop(n, inplace=True)
        sales_price.drop(n, inplace=True)
train_data.reset_index(drop=True, inplace=True)

model_lgb = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11))

clsLG = train(train_data, sales_price, 1, model_lgb)
predictionsLG = clsLG.predict(test_data)

predictions = np.exp(predictionsLG)

dictionary = {'Id': labels, 'SalePrice': predictions}
result = pd.DataFrame(data=dictionary)
result.to_csv(path_or_buf="results.csv", index=False)