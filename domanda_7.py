'''
Approssimazione dati ai minimi quadrati (data set 2).
Discutere l’approssimazione all’ insieme di dati scaricati da:
https://www.kaggle.com/sakshamjn/heightvsweight-for-linear-polynomial-regression
Con polinomi di grado 1,….7.
Risolvere il problema dei minimi quadrati sia con equazioni normali che con SVD, visualizzando
i polinomi ottenuti.
'''


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


dataset = pd.read_csv("HeightVsWeight.csv")
data = dataset.to_numpy()
 
x1 = data[:,0]
y1 = data[:,1]

max_ = 7
for n in range(1, max_+1):
    alpha_normali1, alpha_svd1 = alphas(n,x1,y1)
    
    x_plot1 = np.linspace(10,80,1000)
    
    y_normali1 = p(alpha_normali1, x_plot1)
    y_svd1 = p(alpha_svd1, x_plot1)
    
    y_norm = p(alpha_normali1, x1)
    y_svd = p(alpha_svd1, x1)
    
    err1 = np.linalg.norm(y1-y_norm, 2)
    err2 = np.linalg.norm(y1-y_svd, 2)
    print (f'Errore di approssimazione con Eq. Normali (Grado {n:n}): ', err1)
    print (f'Errore di approssimazione con SVD (Grado {n:n}): ', err2, '\n')
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, 'o')
    plt.plot(x_plot1, y_normali1, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Approssimazione tramite Eq. Normali (Grado={n: n})')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(x1, y1, 'o')
    plt.plot(x_plot1, y_svd1, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Approssimazione tramite SVD (Grado={n: n})')
    plt.grid()
    plt.show()