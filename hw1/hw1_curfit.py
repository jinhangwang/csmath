#! /usr/bin/env python

# csmath_hw1_curfit
# Jinhang Wang
# jhw@zju.edu.cn

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


m1 = 3
m2 = 9
n1 = 9
n2 = 100  
regularization = np.exp(-18)	

def real_func(x):
    return np.sin(2*np.pi*x) #sin(2 pi x)

def fake_func(p, x):
    f = np.poly1d(p)
    return f(x)

def residuals(p, y, x):
    return y - fake_func(p, x)

#regulization
def regresiduals(p,y,x):
	ret = y - fake_func(p, x)
	ret = np.append(ret, np.sqrt(regularization)*p) 
	# add lambda^(1/2)p up to array
	return ret

def plotcurving(M, N, text, reg = 0):
	#choose N point randomly as x
	x = np.linspace(0, 1, N)
	#plot with the point
	x_show = np.linspace(0, 1, 1000)

	y0 = real_func(x)
	#G-distribution y with noise	
	y1 = [np.random.normal(0, 0.1) + y for y in y0]

	p0 = np.random.randn(M)

	if reg == 0:
		plsq = leastsq(regresiduals, p0, args=(y1, x))
	else:
		plsq = leastsq(residuals, p0, args=(y1, x))

	print ('Fitting Parameters: ', plsq[0])

	font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }

	plt.title(text, fontdict=font)
	plt.plot(x_show, real_func(x_show), label='real')
	plt.plot(x_show, fake_func(plsq[0], x_show), label='fitted curve')
	plt.plot(x, y1, 'bo', label='with noise')
	plt.legend()
	plt.show()

# for testing

plotcurving(3,9,'M=3,N=9') #para = 3 with 9 random nosie point
plotcurving(9,9,'M=9,N=9') #para = 9 with 9 random nosie point
plotcurving(9,15,'M=9,N=15') #para = 9 with 15 random nosie point
plotcurving(9,100,'M=9,N=100') #para = 9 with 100 random nosie point
plotcurving(9,10,'M=9,N=10,ln(lambda)=-18',1) 
#para = 9 with 10 random nosie point where ln(lambda) = -18



