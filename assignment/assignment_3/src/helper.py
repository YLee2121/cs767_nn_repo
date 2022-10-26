import pandas as pd   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import six


# ML algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from id3.id3 import Id3Estimator # https://svaante.github.io/decision-tree-id3/
from xgboost import XGBRFClassifier # https://www.datacamp.com/tutorial/xgboost-in-python
from catboost import CatBoostClassifier # https://catboost.ai/en/docs/concepts/python-quickstart
from lightgbm import LGBMClassifier # https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/

class id3_(Id3Estimator):
    def score(self, x_test, y_test):
        yhat = self.predict(x_test)
        return sum(yhat == y_test) / len(y_test)

cart_tree = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
svm = SVC()
nb = GaussianNB()
id3 = id3_()


# CV
from sklearn.model_selection import cross_val_score




def foo1(df):

    # split data
    x = df.loc[:, df.columns != 'target']
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(x.iloc[:, :], y, train_size=0.7, random_state=0)



    # five algorithm
    cart_tree.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    svm.fit(x_train, y_train)
    nb.fit(x_train, y_train)
    id3.fit(x_train, y_train)


    print(f'tree cart accracy {cart_tree.score(x_test, y_test):.3f}')
    print(f'knn accuracy {knn.score(x_test, y_test):.3f}')
    print(f'svm accuracy {svm.score(x_test, y_test):.3f}')
    print(f'NB accuracy {nb.score(x_test, y_test):.3f}')
    print(f'tree ID3 accuracy {id3.score(x_test, y_test):.3f}')
    print()




    # 10 fold CV on five algorithm
    print('10 FOLD')
    print('tree cart accracy {:.3f}'.format(sum(cross_val_score(cart_tree, x, y, cv=10)) / 10))
    print('knn accuracy {:.3f}'.format(sum(cross_val_score(knn, x, y, cv=10)) / 10))
    print('svm accuracy {:.3f}'.format(sum(cross_val_score(svm, x, y, cv=10)) / 10))
    print('NB accuracy {:.3f}'.format(sum(cross_val_score(nb, x, y, cv=10)) / 10))
    print('tree ID3 accuracy {:.3f}'.format(sum(cross_val_score(id3, x, y, cv=10)) / 10 ))


def aug_data_for_iris(target):

    source = {
        0: [(5, 0.35), (3.42, 0.37), (1.46, 0.17), (0.24, 0.10)], 
        1: [(5.93, 0.51), (2.77, 0.31), (4.26, 0.47), (1.32, 0.19)], 
        2: [(6.58, 0.63), (2.97, 0.32), (5.55, 0.55), (2.03, 0.27)]
    }
    res = [np.random.normal(loc=mu, scale=sd) for mu, sd in source[target]]
    res.append(target)

    c = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'target']
    return pd.DataFrame( np.array(res).reshape((1, -1)), columns=c)

def sample_df_for_iris(n):
    
    c = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'target']
    new_df = pd.DataFrame(columns=c)

    for _ in range(n):
        new0 = aug_data_for_iris(0)
        new1 = aug_data_for_iris(1)
        new2 = aug_data_for_iris(2)
        new_df = pd.concat([new_df, new0, new1, new2])

    return new_df


def accuracy(yhat, y_test):
    return sum(yhat == y_test) / len(y_test)

def xgboost_accuracy(df):
    clf = XGBRFClassifier()
    x = df.loc[:, df.columns != 'target']
    y = df['target']   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf.fit(x_train, y_train)
    yhat = clf.predict(x_test)
    return accuracy(yhat, y_test)

def catboost_accuracy(df):
    clf = CatBoostClassifier(verbose=False)
    x = df.loc[:, df.columns != 'target']
    y = df['target']   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf.fit(x_train, y_train)
    yhat = clf.predict(x_test)
    yhat = yhat.reshape((1, -1))[0]
    return accuracy(yhat, y_test)

def lgbm_accuracy(df):
    clf = LGBMClassifier()
    x = df.loc[:, df.columns != 'target']
    y = df['target']   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    clf.fit(x_train, y_train, eval_set=[(x_test,y_test),(x_train,y_train)], verbose=False, eval_metric='logloss')
    yhat = clf.predict(x_test)
    return accuracy(yhat, y_test)