#! /usr/bin/env python

# csmath_hw2_pca
# Jinhang Wang
# jhw@zju.edu.cn

import numpy as np
from numpy import genfromtxt
import matplotlib.pylab as py


def ReadData(filename):
    digit = genfromtxt('data.txt', dtype = str)

    # scan the file onece and find the size of digit "3"
    count_three = 0; #size of digit "3"
    count_sum = 0;#total lines in the file
    for data in digit:
        count_sum = count_sum + 1
        if(data == '3'):
            count_three = count_three + 1
    matrix_three = np.ones(shape = [count_three,32 * 32],dtype = int)

    # Read data vector of digit "3"
    count = 0;
    for i in range(0, count_sum):
        if(digit[(i//33 + 1) * 33 - 1] == '3'):
            if(i%33 == 32):
                count = count + 1
                continue
            else:
                for j in range(0,32):
                    matrix_three[count][(i%33)*32 + j] = digit[i][j]

    return matrix_three


def Pca(data):
    x, y = data.shape
    # centeralize the data
    c_data = (data - np.average(data,0)).T
    # calculate covariance matrix
    cov = np.dot(c_data, c_data.T)
    [u,s,v] = np.linalg.svd(cov)

    pcs = np.dot(u.T, c_data)

    return  pcs, c_data, s, cov, u


def ShowFig(data,font):
    x, y = data.shape
    fig, ax = py.subplots(nrows = x//2,ncols = 2) # get i*j subplots
    for i in range(0, x//2):
        for j in range (0, 2):
            ax[i][j].imshow(data[i*2+j].reshape(32, 32))
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            string = '$'+'Feature: '+ str(i*2 + j)+'$'
            ax[i][j].set_xlabel(string,fontdict=font)
    py.subplots_adjust(wspace = 0.1,hspace = 0.1)
    py.show()


def main():
    matrix = ReadData('data.txt')
    x,y = matrix.shape
    pcs, c_data, s, cov, u= Pca(matrix)
    colors = np.random.rand(x)
    area = np.pi * (15 * np.random.rand(x))**2
    a = pcs[0,:]
    b = pcs[1,:]


    # Figure the result
    font = {'family' : 'serif',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   :  20,}
    py.scatter(a,b,s = area,c = colors,alpha = 0.5,label = r"$Digit\_Three$")
    py.xlabel(r"$First$ $Principle$ $Component$", fontdict = font)
    py.ylabel(r"$Second$ $Principle$ $Component$", fontdict = font)
    py.legend()
    py.grid(True)
    py.show()

    ShowFig(u.T[0:4],font)


if __name__ == '__main__':
    main()
