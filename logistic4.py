import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_size = 4

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoid_prime(self, t):
        return self.sigmoid(t) * (1 - self.sigmoid(t))

    def predict_proba(self, row, coef_):
        t = (1, *row) @ coef_
        return self.sigmoid(t)

    def zscore(self, x):
        m = np.average(x)
        s = np.std(x)
        return (x - m) / s
        
    def fit_mse(self, X_train, y_train):
#        self.coef_ = np.random.randn(self.coef_size)   # initialized weights
        self.coef_ = np.zeros(self.coef_size)
        y_hat_a = np.zeros(len(X_train))

        for e in range(self.n_epoch):
#            print(e)
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                y_hat_a[i] = y_hat
                self.coef_[0] -= self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                self.coef_[1] -= self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * row[0]
                self.coef_[2] -= self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * row[1]
                self.coef_[3] -= self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * row[2]
            if e == 0 or e == len(X_train) - 1:
                errors = np.zeros(len(X_train))
                for k in range(len(X_train)):
                    sum = 0
                    for j in range(k + 1):
                        sum += (y_hat_a[j] - y_train[j]) ** 2
                    errors[k] = sum / (k + 1)    
                if e == 0:
                    self.mse_error_first = errors
                if e == len(X_train) - 1:
                    self.mse_error_last = errors

    def predict(self, X_test, cut_off=0.5):
        predictions = np.zeros(len(X_test))
        for i, row in enumerate(X_test):
            y_hat = self.predict_proba(row, self.coef_)
            predictions[i] = 1 if y_hat >= cut_off else 0
        return predictions # predictions are binary values - 0 or 1

    def evaluate(self, predictions, Y_test):
        counter = 0
        for i in range(len(predictions)):
            counter += 1 if predictions[i] == Y_test[i] else 0
        return counter / len(predictions)

    def fit_log_loss(self, X_train, y_train):
#        self.coef_ = np.random.randn(self.coef_size)   # initialized weights
        self.coef_ = np.zeros(self.coef_size)
        n = len(X_train)
        y_hat_a = np.zeros(len(X_train))

        for e in range(self.n_epoch):
#            print(e)
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                y_hat_a[i] = y_hat
                self.coef_[0] -= self.l_rate * (y_hat - y_train[i]) / n
                self.coef_[1] -= self.l_rate * (y_hat - y_train[i]) * row[0] / n
                self.coef_[2] -= self.l_rate * (y_hat - y_train[i]) * row[1] / n
                self.coef_[3] -= self.l_rate * (y_hat - y_train[i]) * row[2] / n
            if e == 0 or e == len(X_train) - 1:
                errors = np.zeros(len(X_train))
                for k in range(len(X_train)):
                    sum = 0
                    for j in range(k + 1):
                        sum -= y_train[j] * np.log(y_hat_a[j]) + (1 - y_train[j]) * (1 - np.log(y_hat_a[j]))
                    errors[k] = sum / (k + 1)    
                if e == 0: 
                    self.logloss_error_first = errors
                if e == len(X_train) - 1:
                    self.logloss_error_last = errors

if __name__ == "__main__":

    lr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)

    breast_cancer = load_breast_cancer()

    feature_names = list(breast_cancer['feature_names'])
    f1 = feature_names.index('worst concave points')
    f2 = feature_names.index('worst perimeter')
    f3 = feature_names.index('worst radius')

    data = breast_cancer['data']
    target = breast_cancer['target']
    l1 = [d[f1] for d in data]
    l2 = [d[f2] for d in data]
    l3 = [d[f3] for d in data]

    z1 = lr.zscore(l1)
    z2 = lr.zscore(l2)
    z3 = lr.zscore(l3)

    x = list(zip(z1, z2, z3))
    y = target
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=43)

    answer_dict = {}

    lr.fit_mse(x_train, y_train)
    predictions = lr.predict(x_test)
    accuracy_rate = sklearn.metrics.accuracy_score(y_test, predictions)
    answer_dict['mse_accuracy'] = accuracy_rate

    lr.fit_log_loss(x_train, y_train)
    predictions = lr.predict(x_test)
    accuracy_rate = sklearn.metrics.accuracy_score(y_test, predictions)
    answer_dict['logloss_accuracy'] = accuracy_rate

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
    preictions = clf.predict(x_test)
    accuracy_rate = sklearn.metrics.accuracy_score(y_test, predictions)
    answer_dict['sklearn_accuracy'] = accuracy_rate

    answer_dict['mse_error_first'] = list(lr.mse_error_first)
    answer_dict['mse_error_last'] = list(lr.mse_error_last)
    answer_dict['logloss_error_first'] = list(lr.logloss_error_first)
    answer_dict['logloss_error_last'] = list(lr.logloss_error_last)

    print(answer_dict)

    print("""Answers to the questions:
1) 0.00005
2) 0.00000
3) 0.00153
4) 0.00550
5) expanded
6) expanded
""")
    pass