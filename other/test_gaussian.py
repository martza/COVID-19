import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from scipy import optimize


def r2(y,y_new):
    r = 1-np.sum((y-y_new)**2)/np.sum((y-mse(y,y_new))**2)
    return r
def mse(y,y1):
    m = np.sum((y-y1)**2/len(y))
    return m

x=np.arange(-1,3,0.01)
y=np.exp(-(x-1.0)**2/2)

y1=np.log(y)

def gauss(x, a, b, c, d):
    f = c*np.exp(-(x-a)**2/b)+d
    return f

mean = sum(y)/len(x)
sum((y-mean)**2)/len(y)

popt, pcov = optimize.curve_fit(gauss, x, y, p0= [1,0.5,1,0])
y_pred = gauss(x,*popt)
r2(y,y_pred)
r2_score(y,y_pred)

plt.plot(x,y, label = 'exact')
plt.plot(x,gauss(x,*popt), label = 'transform')
plt.legend()
plt.show()

plt.plot(x,y1, label = 'log')
plt.plot(x,-(x-1.0)**2/2.0, label ='exact')
