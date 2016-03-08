# solution for exercise 1

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


class linear_regression(object):
    def __init__(
                    self,
                    method='newton',
                    ephocs=10
                ):
        self.method = method
        self.ephocs = ephocs
        self.cost_ = list()

    def fit(self, X, y):
        self.n_feature = X.shape[1]
        self.weights_ = np.zeros(self.n_feature + 1)
        # self.weights_ = np.random.uniform(-1.0, 1.0, size=self.n_feature+1)
        train_X = X.copy()
        # add bias to train X data
        m = X.shape[0]
        ones = np.ones((m, 1))
        train_X = np.concatenate((ones, train_X), axis=1)
        self._train(train_X, y)

    def _train(self, X, y):
        m = y.shape[0]
        # constract the dignozal matrix of gradient_of_sigmoid
        D = np.zeros((m, m))
        # print 'Hessian'
        # print hessian
        # the hessian in first iteration is non-singular, as it's
        # values are zeros as initial weights set to zero
        for i in range(self.ephocs):
            z = self.predict(X)
            cost = np.dot(y.T, np.log(z)) + np.dot((1 - y).T, np.log(1 - z))
            cost = -1.0 * cost
            self.cost_.append(cost)
            gradient = np.dot(X.T, (z - y))
            gradient_of_sigmoid = z * (1 - z)
            for i in range(m):
                D[i, i] = gradient_of_sigmoid[i]
            hessian = np.dot(X.T, np.dot(D, X))
            invert_hessian = np.linalg.pinv(hessian)
            self.weights_ -= np.dot(gradient, invert_hessian)

    def predict(self, X):
        if X.shape[1] != self.weights_.shape[0]:
            m = X.shape[0]
            ones = np.ones((m, 1))
            predict_X = np.concatenate((ones, X), axis=1)
        else:
            predict_X = X

        return expit(np.dot(predict_X, self.weights_))


def ex1_1():
    X = np.loadtxt('q1x.dat')
    y = np.loadtxt('q1y.dat')
    # draw scatter seperately
    fit_draw_linear_boundary(X, y)


def get_y_line_with_x(a, x, b, intercept, boundary=0.5):
    """
    x and y will fit a line as follow:
    a*x + b*y + intercept = boundary
    x and y fit a line in 2D subspace
    Parameter:
        x 1d array: (m,)
        a number, a parameter associate with x
        b number, a parameter associate with y
        intercept number
        boundary number, default : 0.5
    Return:
        y 1d array: (m,)
    """
    y = (boundary - intercept - a * x) / b
    return y


def fit_draw_linear_boundary(X, y):
    # draw scatter seperately
    positive_points = np.where(y == 1)
    negetive_points = np.where(y == 0)
    plt.plot(X[positive_points, 0], X[positive_points, 1], 'ro')
    plt.plot(X[negetive_points, 0], X[negetive_points, 1], 'b+')
    # fit a line in space
    lr = linear_regression(ephocs=50)
    lr.fit(X, y)
    intercept, a, b = lr.weights_[0], lr.weights_[1], lr.weights_[2]
    x1_min = X[:, 0].min()
    x1_max = X[:, 0].max()
    x1 = np.linspace(x1_min, x1_max, 50)
    x2 = get_y_line_with_x(a, x1, b, intercept)
    plt.plot(x1, x2, 'y-')
    plt.show()


def ex1_2():
    # load q2x.dat and q2y.dat
    # fit a line with non-weighted linear regression
    X = np.loadtxt('q2x.dat')
    y = np.loadtxt('q2y.dat')
    x = np.linspace(X.min(), X.max(), 50)
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.concatenate((ones, X.reshape((-1, 1))), axis=1)
    # get the weights(theta) with normal equation
    var_X = np.dot(X.T, X)
    inv_var_X = np.linalg.pinv(var_X)
    theta = np.dot(np.dot(inv_var_X, X.T), y)
    y_line = x * theta[1] + x * theta[0]
    # scatter the trainning data
    plt.plot(X[:, 1], y, 'b+')
    # draw fitted line
    plt.plot(x, y_line, 'r-')
    plt.show()


class WeightedLinearRegression(object):
    def __init__(self, kernal='guassian', tau=0.1):
        self.kernal = kernal
        self.tau = tau

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        """
        X is a 1d array
        """
        eudin_distance = np.power(np.sum((X - self.train_X), axis=1), 2)
        w = np.exp(
            -1.0 * eudin_distance / (2.0 * self.tau**2)
            )
        W = np.diag(w) * 0.5
        # print W.shape
        # print self.train_X.shape
        inv = np.linalg.pinv(np.dot(np.dot(self.train_X.T, W), self.train_X))
        theta = np.dot(np.dot(np.dot(inv, self.train_X.T), W), self.train_y)
        predict_y = np.dot(theta.T, X)
        return predict_y


def ex1_3():
    # load q2x.dat and q2y.dat
    # fit a line with non-weighted linear regression
    X = np.loadtxt('q2x.dat')
    y = np.loadtxt('q2y.dat')
    x = np.linspace(X.min(), X.max(), 200)
    m = X.shape[0]
    ones = np.ones((m, 1))
    train_X = np.concatenate((ones, X.reshape((-1, 1))), axis=1)
    m = x.shape[0]
    ones = np.ones((m, 1))
    predict_X = np.concatenate((ones, x.reshape((-1, 1))), axis=1)
    predict_y = list()
    # ----------------------------------
    wlr = WeightedLinearRegression()
    wlr.fit(train_X, y)
    for v in predict_X:
        py = wlr.predict(v)
        predict_y.append(py)
    predict_y_tau1 = np.array(predict_y)
    # ----------------------------------
    wlr = WeightedLinearRegression(tau=0.3)
    wlr.fit(train_X, y)
    predict_y = list()
    for v in predict_X:
        py = wlr.predict(v)
        predict_y.append(py)
    predict_y_tau2 = np.array(predict_y)
    # ----------------------------------
    wlr = WeightedLinearRegression(tau=2)
    wlr.fit(train_X, y)
    predict_y = list()
    for v in predict_X:
        py = wlr.predict(v)
        predict_y.append(py)
    predict_y_tau3 = np.array(predict_y)
    # scatter the trainning data
    # scatter the trainning data
    plt.plot(train_X[:, 1], y, 'b+', label='train data')
    plt.plot(predict_X[:, 1], predict_y_tau1, 'r-', label='tau=0.1')
    plt.plot(predict_X[:, 1], predict_y_tau2, 'g-', label='tau=0.3')
    plt.plot(predict_X[:, 1], predict_y_tau3, 'c-', label='tau=2')
    plt.legend()
    plt.show()
