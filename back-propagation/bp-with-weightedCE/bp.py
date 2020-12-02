import numpy as np
import sklearn.datasets as sd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

'''
dataset:
X to be input with two features
y to be k = 2
'''
X, y = sd.make_moons(500, noise=0.4)

def plot(X1, X2, y):
    color = {0: 'blue', 1: 'red'}
    plt.title('dataset')
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i, j, c in zip(X1, X2, y):
        plt.scatter(i, j, c = color[c], marker='o', s = 50, edgecolors='k', cmap = plt.cm.Spectral)
        
plot(X[:, 0], X[:, 1], y)
plt.show()

def nn_plot(X, y, pfunc):
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    steps = 0.01
    xx, yy = np.meshgrid(np.arange(xmin, xmax, steps), np.arange(ymin, ymax, steps))
    labels = pfunc(np.c_[xx.ravel(), yy.ravel()])
    zz = labels.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap = plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap = plt.cm.Spectral)
    plt.show()

class NN(object):
    def __init__(self):
        self.input= 2
        self.output = 1
        self.hidden_units = 2
        
        # weight for false negative and false positive
        # false negative is more serious than false positive
        self.fn_wt = 50
        self.fp_wt = 10
        
        np.random.seed(1)
        # w1 matrix from input to hidden layer 2*2
        self.w1 = np.random.randn(self.input, self.hidden_units)
        # w2 matrix from hidden layer 2*1
        self.w2 = np.random.randn(self.hidden_units, self.output)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    # derivation for sigmoid
    def deriv_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    def ff(self, x):
        # hidden layer
        self.hu_sum = np.dot(self.w1.T, x.T)
        self.hu_output = self.sigmoid(self.hu_sum)
        # output layer
        self.output_sum = np.dot(self.w2.T, self.hu_output)
        self.output = self.sigmoid(self.output_sum)
        #print("w1: ", self.w1)
        #print("w2: ", self.w2)
        return self.output
    
    def loss(self, pred, y):
        m = y.shape[0]
        logprob = self.fn_wt * np.multiply(np.log(pred), y) + self.fp_wt * np.multiply(1-y, np.log(1 - pred))
        loss = -np.sum(logprob) / m
        return loss
    
    def bp(self, X, y):
        pred = self.ff(X)
        m = X.shape[0]
        # update weight in w2
        # chain rule: 
        #output_delta = pred - y
        #output_delta = 100*pred + 1900* np.multiply(y, pred) - 2000*y
        output_delta = -50 * y + 40 * np.multiply(pred, y) + 10 * pred 
        delta_dz = np.multiply(output_delta, self.deriv_sigmoid(self.output_sum))
        self.dw2 = (1/m) * np.sum(np.multiply(self.hu_output, delta_dz), axis = 1).reshape(self.w2.shape)
        
        # update weight in w1
        hidden_delta = output_delta * self.w2 * self.deriv_sigmoid(self.hu_sum)
        self.dw1 = (1/m) * np.dot(X.T, hidden_delta.T)
        
    def update(self, lr = 1.2):
        self.w1 = self.w1 - lr * self.dw1
        self.w2 = self.w2 - lr * self.dw2
        
    def train(self, X, y, it = 100):
        for i in range(it):
            y_hat = self.ff(X)
            los = self.loss(y_hat, y)
            self.bp(X, y)
            # update weight w1, w2
            self.update()
            if i % 10 == 0:
                print("loss: ", los)
                
    def pred(self, X):
        y_hat = self.ff(X)
        y_hat = [1 if x_[0] >= 0.5 else 0 for x_ in y_hat.T]
        return np.array(y_hat)
    
    def score(self, pred, y):
        # accuracy
        corect_cnt = np.sum(pred == y)
        correct0 = 0
        correct1 = 0
        for i in range(len(y)):
            if pred[i] == y[i]:
                if pred[i] == 0:
                    correct0 += 1
                else:
                    correct1 += 1
        return corect_cnt / len(y), correct0, correct1
        
    
if __name__ == '__main__':
    # 4 fold validation
    tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size = 0.25)
    # compare linear model perceptron with neural network
    clf = Perceptron(tol = 1e-3, random_state= 0)
    clf.fit(X, y)
    nn_plot(X, y, lambda x : clf.predict(x))
    print("Perceptron's score: ", clf.score(X, y))
    model = NN()
    model.train(tr_X, tr_y)
    pred_y = model.pred(te_X)
    print("after train")
    score, score0, score1 = model.score(pred_y, te_y)
    nn_plot(X, y, lambda x : model.pred(x))
    print("predict: ", pred_y)
    print("label:   ", te_y)
    print("NN's score: ", score, score0, score1)