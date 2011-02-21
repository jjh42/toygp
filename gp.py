# http://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf
# http://videolectures.net/gpip06_mackay_gpb/
from pylab import *

class GP(object):
    """A simple naive implentation of a Gaussian process purely for learning. Expects all input
    as numpy matrixes."""
    sigma_f = 1.27
    l = 1.
    sigma_n = 0.09
    def __init__(self):
        pass
    
    def train(self, x,y):
        """Train the data on a setup of vector in column form x with the paired y as a column vector."""
        assert(x.shape[1] == y.shape[0])
        self.K = self.covfn(x, x)
        self.invK = inv(self.K)
        self.y = y
        self.x = x
    
    def predict(self, xstar):
        """Given a new entry x (or column of x entries) return the mean and the variance of the
        prediction on y."""
        kstar = self.covfn(self.x, xstar)
        kstarstar = self.covfn(xstar, xstar)
        ystar_mean = kstar*self.invK*self.y
        ystar_var = diag(kstarstar - kstar*self.invK*kstar.transpose())
        return (ystar_mean, ystar_var)

    def covfn(self,x, xprime):
        # Calculate the covariance between and x and xprime (in vector form each column is a seperate vector).
        # If multiple entries are provided then returns the covariance matrix.
        k = lambda xi,xj: asscalar((self.sigma_f**2*exp(-dot((xi - xj).T,xi - xj)/(2*self.l**2)) +
             all(xi == xj)*self.sigma_n))
        C = zeros((xprime.shape[1], x.shape[1]))
        for i in range(xprime.shape[1]):
             for j in range(x.shape[1]):
                 C[i,j] = k(x[:,j], xprime[:,i])
        return asmatrix(C)

class GPDemo2D(object):
    x = matrix([-1.5, -1, -0.75, -0.4, -0.25, 0])
    y = matrix([-1.8, -1, -0.1, 0.5, 1., 1.2]).T
    
    def plot(self):
        self.gp = GP(); self.gp.train(self.x, self.y)
        xpred = asmatrix(linspace(np.min(self.x), np.max(self.x)))
        # Plot data
        clf()
        plot(self.x.T, self.y, 'r^', markersize=10)
        # Plot predict
        ypred_mean, ypred_var = self.gp.predict(xpred)
        errorbar(asarray(xpred).flatten(), asarray(ypred_mean).flatten(), yerr=np.sqrt(asarray(ypred_var).flatten())*1.96)

