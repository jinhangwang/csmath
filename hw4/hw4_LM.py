#! /usr/bin/env python

# csmath_hw4_LM
# Jinhang Wang
# jhw@zju.edu.cn

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

def Equation(eq_name, var, x):
    if eq_name == 'quadratic':
       value = 0.5 * np.dot(np.dot(x.T,var['H']),x)\
       + np.dot(var['C'].T,x) + var['B']
    return value


def Derivatives(eq_name, var, x):
    if eq_name == 'quadratic':
        data = np.dot(var['H'],x) + var['C']

    return data


def Jacobian(eq_name, var, x):
    if eq_name == 'quadratic':
        data = var['H']

    return data


def CalMu(data, mu):
    I = np.eye(data.shape[0])
    while(not np.all(np.linalg.eigvals(data+I*mu)>0)):
        mu = mu * 4

    return mu


def LevenbergMarquardt(eq_name ,var, max_iter = 400, tol = 1e-8):
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    #X,Y = np.meshgrid(X, Y)
    Z = np.ones(shape = (X.size, Y.size))
    for i in range(X.size):
        for j in range(Y.size):
            Z[i][j] = Equation('quadratic',var, np.array([X[i],Y[j]]))
    X,Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=6, cstride=6, alpha=0.3,edgecolor = 'black',color = 'yellow')
    cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
    print('LevenbergMarquardt for quardic optimal')

    i = 0
    g = np.random.random((3,1))
    x = np.array([[4],[4]])
    mu = 0.1
    f = Equation(eq_name, var, x)
    q = None

    while(i <= max_iter and np.dot(g.T,g) > tol):
        ax.scatter(x[0],x[1],f,zdir='z',edgecolor = 'black',color = 'red',s = 25)
        ax.scatter(x[0],x[1],[0],zdir='z',edgecolor = 'black',color = 'none',s = 25)
        g = Derivatives(eq_name, var, x)
        G = Jacobian(eq_name, var, x)
        mu = CalMu(G,mu)
        s = np.linalg.solve(G+np.eye(G.shape[0])*mu, -1 * g)
        f_new = Equation(eq_name, var, x + s)
        q_new = f_new + 0.5*np.dot(g.T, s) + 0.5*np.dot(np.dot(s.T,G),s)
        if  q == None:
            f = f_new
            q = q_new
            x = x + s
        else:
            r = (f_new-f)/(q_new-q)
            f = f_new
            q = q_new
            if r < 0.25:
                mu = mu*4
            if r > 0.75:
                mu = mu/2
            if r >=0.25 and r<=0.75:
                mu = mu
            if r <= 0:
                x = x
            else:
                x = x + s
        i = i + 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')



def main():
    '''
    Main function
    '''
    print('main.')
    var = {}
    var['H'] = np.ones(shape =(2,2))
    var['H'][0][1] = 0
    var['H'][1][0] = 0
    var['C'] = np.zeros(shape = (2,1))
    var['B'] = 1
    LevenbergMarquardt('quadratic',var)


    plt.show()

if __name__ == '__main__':
    main()
