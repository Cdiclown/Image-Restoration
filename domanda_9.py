'''Aprrosimazione funzione ai minimi quadrati'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LUdec

def p(alphas, x):
    y = np.zeros(x.size)
    for i in range(alphas.size):
        y += alphas[i]*x**i
        
    return y
    
def alphas(n, x, y):
    N = x.size
    
    A = np.zeros((N, n+1))
    
    for i in range(n+1):
        for j in range (N):
            A[j,i] = x[j]**i
    
    ATA = A.T@A
    ATy = A.T@y
    
    lu,piv = LUdec.lu_factor(ATA)
    
    alpha_normali = LUdec.lu_solve((lu,piv), ATy)
    
    # SVD
    
    U, s, Vh = np.linalg.svd(A)
    
    alpha_svd = np.zeros(n+1)
    
    for i in range(n+1):
        ui = U[:,i]
        vi = Vh.T[:,i]
        
        alpha_svd = alpha_svd + ((ui.T@y)*vi/s[i])
    
    return alpha_normali, alpha_svd

m = 20
orders = np.array([1, 2, 3, 5, 7])

def create_approximations(f, start, end, orders =orders):
    x = np.linspace(start,end,m)
    y = f(x)
    
    errors=np.zeros(orders.size)
    
    x_plot = np.linspace(start,end,100)
    
    j = 0
    for i in orders:
        alpha_normali, alpha_svd = alphas(i, x, y)
        y1 = p(alpha_svd, x)
        y2 = p(alpha_normali, x)
        
        y_svd = p(alpha_svd, x_plot)
        y_normali = p(alpha_normali, x_plot)
        
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.title(f"Equazioni normali grado={i}")
        plt.plot(x, y, 'o')
        plt.plot(x_plot, f(x_plot), 'blue')
        plt.plot(x_plot, y_normali, label=f"EQ normali grado={i}", color='green')
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.title(f"SVD grado={i}")
        plt.plot(x, y, 'o')
        plt.plot(x_plot, f(x_plot), 'blue')
        plt.plot(x_plot, y_svd, label=f"SVD grado={i}", color='green')
        plt.grid()
        
        plt.show()
        
        err1 = np.linalg.norm(y-y1, 2)
        err2 = np.linalg.norm(y-y2, 2)
        print (f'Errore di approssimazione con Eq. Normali (Grado {i:n}): ', err1)
        print (f'Errore di approssimazione con SVD (Grado {i:n}): ', err2, '\n')
        
        j=j+1
        
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0.0, shadow=True)

    
    return errors

f1 = lambda x: 1/(1+25*(x**2))
err = create_approximations(f1, -1, 1)