import pandas as pd
import numpy as np
import scipy.stats as st
import math
from sklearn import linear_model, metrics
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import resample
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle as pkl

# these are the sentiment analysis models
model1 = pkl.load(open("models/model1.p", "rb"))
model2 = pkl.load(open("models/model2.p", "rb"))
model3 = pkl.load(open("models/model3.p", "rb"))
model4 = pkl.load(open("models/model4.p", "rb"))
model5 = pkl.load(open("models/model5.p", "rb"))
model6 = pkl.load(open("models/model6.p", "rb"))
model7 = pkl.load(open("models/model7.p", "rb"))

# returns vader polarity scores
vader = SentimentIntensityAnalyzer()
def vader_pos(text):
    return vader.polarity_scores(text)['pos']
def vader_neu(text):
    return vader.polarity_scores(text)['neu']
def vader_neg(text):
    return vader.polarity_scores(text)['neg']

# use latent dirchlet allocation to classify topics (5 categories), returns matrix
def classify_lda(text):
    tf_vectorizer = TfidfVectorizer(max_features=25)
    temp = tf_vectorizer.fit_transform(text)
    lda = LatentDirichletAllocation(n_topics=5)
    lda.fit(temp)
    return np.matrix(lda.transform(temp))

# runs linear regression and k-fold (k=10)
def ols(X, y):
    lm = linear_model.LinearRegression()
    kf = KFold(n_splits=10, shuffle=True)
    r2 = []
    mse = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2.append(metrics.r2_score(y_test, y_pred))
        mse.append(metrics.mean_squared_error(y_test, y_pred))
    return mse, r2

# runs ridge regression for given alpha and k-fold (k=10)
def ridge(X, y, a):
    lm = linear_model.Ridge(alpha=a)
    kf = KFold(n_splits=10, shuffle=True)
    r2 = []
    mse = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2.append(metrics.r2_score(y_test, y_pred))
        mse.append(metrics.mean_squared_error(y_test, y_pred))
    return mse, r2

# runs lasso regression for given alpha and k-fold (k=10)
def lasso(X, y, a):
    lm = linear_model.Lasso(alpha=a)
    kf = KFold(n_splits=10, shuffle=True)
    r2 = []
    mse = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = lm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2.append(metrics.r2_score(y_test, y_pred))
        mse.append(metrics.mean_squared_error(y_test, y_pred))
    return mse, r2

# given alphas, find most optimal for Ridge regression
def findRidgeAlpha(X,y,alphas):
    model = linear_model.Ridge()
    search = GridSearchCV(estimator=model, param_grid = dict(alpha=alphas))
    search.fit(X,y)
    return search.best_estimator_.alpha

# given alphas, find most optimal for Lasso regression
def findLassoAlpha(X,y,alphas):
    model = linear_model.Lasso()
    search = GridSearchCV(estimator=model, param_grid = dict(alpha=alphas))
    search.fit(X,y)
    return search.best_estimator_.alpha

# performs support vector regression
def svreg(X,y):
    clf = SVR()
    kf = KFold(n_splits=10, shuffle=True)
    r2 = []
    mse = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2.append(metrics.r2_score(y_test, y_pred))
        mse.append(metrics.mean_squared_error(y_test, y_pred))
    return mse, r2

# import data
df = pd.read_csv('realdonaldtrump.csv')

# feature extraction
df['word_count'] = df['text'].str.split().apply(len) # word count
df['char_length'] = df['text'].apply(len) # character length
df['vader_pos'] = df['text'].apply(vader_pos) # vader positivity score
df['vader_neu'] = df['text'].apply(vader_neu) # vader neutrality score
df['vader_neg'] = df['text'].apply(vader_neg) # vader negativity score

# classify using latent dirchlet allocation
ld_matrix = classify_lda(df['text'])
df['ldacat1'] = ld_matrix[:,0]
df['ldacat2'] = ld_matrix[:,1]
df['ldacat3'] = ld_matrix[:,2]
df['ldacat4'] = ld_matrix[:,3]
df['ldacat5'] = ld_matrix[:,4]


X = df[['word_count', 'char_length', 'vader_pos', 'vader_neu', 'vader_neg',
        'ldacat1', 'ldacat2', 'ldacat3', 'ldacat4', 'ldacat5']] # regressors
y = df['favorite_count'] # target = favorites
# y = df['retweet_count'] # taerget = retweets

# linear regression
ols_mse, ols_r2 = ols(X, y)

# regularized regression
alphas = [.001,.01,.1,1,10,100,1000,10000,100000]
ar = findRidgeAlpha(X,y,alphas)
al = findLassoAlpha(X,y,alphas)
rreg_mse, rreg_r2 = ridge(X, y, ar)
lreg_mse, lreg_r2 = lasso(X, y, al)

# SVR
svreg_mse, svreg_r2 = svreg(X,y)

# print results
print("OLS MSE: ", ols_mse)
print("OLS R2: ", ols_r2)
print("Ridge MSE: ", rreg_mse)
print("Ridge R2: ", rreg_r2)
print("Lasso MSE: ", lreg_mse)
print("Lasso R2: ", lreg_r2)
print("SVR MSE: ", svreg_mse)
print("SVR R2: ", svreg_r2)
