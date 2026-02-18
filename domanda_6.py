'''
Approssimazione dati ai minimi quadrati (data set 1). Discutere l’approssimazione al seguente
insieme di dati:
(1.0, 1.18), (1.2, 1.26), (1.4, 1.23), (1.6, 1.37), (1.8, 1.37), (2.0, 1.45), (2.2, 1.42), (2.4, 1.46),
(2.6, 1.53), (2.8, 1.59), (3.0, 1.50)
Con polinomi di grado 1,….7
Risolvere il problema dei minimi quadrati sia con equazioni normali che con SVD, visualizzando
i polinomi ottenuti
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LUdec

x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])

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
    ATy = np.dot(A.T, y)
    
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

max_ = 7

for n in range(1, max_+1):
    
    alpha_normali, alpha_svd = alphas(n, x, y)
    
    y1 = p(alpha_normali, x)
    y2 = p(alpha_svd, x)
    
    err1 = np.linalg.norm(y-y1, 2)
    err2 = np.linalg.norm(y-y2, 2)
    print (f'Errore di approssimazione con Eq. Normali (Grado {n:n}): ', err1)
    print (f'Errore di approssimazione con SVD (Grado {n:n}): ', err2, '\n')
    
    x_plot = np.linspace(1,3,1000)
    
    y_normali = p(alpha_normali, x_plot)
    y_svd = p(alpha_svd, x_plot)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')
    plt.plot(x_plot, y_normali, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Approssimazione tramite Eq. Normali (Grado={n: n})')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'o')
    plt.plot(x_plot, y_svd, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Approssimazione tramite SVD (Grado={n: n})')
    plt.grid()
    plt.show()