'''
Risoluzione di sistemi lineari: confronto metodi. Dato un problema test di dimensione
variabile la cui soluzione esatta sia il vettore x di tutti elementi unitari e b il termine noto
ottenuto moltiplicando la matrice A per la soluzione x
• Confrontare la soluzione del sistema lineare Ax=b con i metodi diretti LU e Cholesky
e i metodi iterativi di Jacobi e gauss Sidel considerando i grafici dell’ errore e del
tempo dei 4 metodi al variare della dimensione N del sistema
• Testare sulla matrice tridiagonale simmetrica e definita positiva positiva avente
sulla diagonale elementi uguali a 9 e quelli sopra e sottodiagonali uguali a -4
(variare n).
'''


import time
import numpy as np
import scipy
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import matplotlib.pyplot as plt 

# Numero di condizione
def Condizione(A):
    condA = np.linalg.cond(A, 2)
    return condA

# definzione metodi diretti
def cholesky(A, x, print_data=False):
    b = np.dot(A, x)

    condA = Condizione(A)
    print('K(A) = ', condA)

    L = scipy.linalg.cholesky(A, True)
    y = scipy.linalg.solve(L, b)
    my_x = scipy.linalg.solve(L.T, y)

    err =  scipy.linalg.norm(x-my_x,2)

    if(print_data):
        print('K(A)=', condA, '\n')
        print('my_x = \n', my_x)
        print('norm =', err)
    
    return my_x, condA, err

def lu_dec(A, x, print_data=False):
    b = np.dot(A, x)
    condA = Condizione(A)
    print('K(A) = ', condA)
    lu, piv = LUdec.lu_factor(A)
    my_x = LUdec.lu_solve((lu,piv), b)
    err =  scipy.linalg.norm(x-my_x,2)
    if(print_data):
        print('my_x = \n ', my_x)
        print('K(A)=', condA, '\n')
        print('norm =', err)
    return my_x, condA, err

# Definizione dei metodi iterativi

def Jacobi(A, b, x0, max_iter, tol, x_true):
    dim = np.size(x0)
    iter = 0
    x = x0.copy()
    norma_it = 1 + tol  # Soglia di convergenza dell'errore relativo
    rel_err = np.zeros((max_iter, 1))
    iter_err = np.zeros((max_iter, 1))
    rel_err[0] = np.linalg.norm(x_true - x0) / np.linalg.norm(x_true)
    
    while norma_it > tol and iter < max_iter:
        x_old = np.copy(x)
        
        for i in range(dim):
            x[i] = (b[i] - np.dot(A[i, 0:i], x_old[0:i]) - np.dot(A[i, i+1:dim], x_old[i+1:dim])) / A[i, i]
        
        iter += 1
        norma_it = np.linalg.norm(x_old - x) / np.linalg.norm(x_old)
        rel_err[iter] = np.linalg.norm(x_true - x) / np.linalg.norm(x_true)
        iter_err[iter] = norma_it
    
    rel_err = rel_err[:iter]
    iter_err = iter_err[:iter]
    
    return x, iter, rel_err, iter_err

def gauss_seidel(A, b, x0, max_iter, tol, x_true):
    dim = np.size(x0)
    iter = 0
    x = x0.copy()
    norma_it = 1 + tol
    rel_err = np.zeros((max_iter, 1))
    iter_err = np.zeros((max_iter, 1))
    rel_err[0] = np.linalg.norm(x_true - x0) / np.linalg.norm(x_true)
    
    while norma_it > tol and iter < max_iter:
        x_old = np.copy(x)
        
        for i in range(dim):
            x[i] = (b[i] - np.dot(A[i, 0:i], x[0:i]) - np.dot(A[i, i+1:dim], x_old[i+1:dim])) / A[i, i]
        
        iter += 1
        norma_it = np.linalg.norm(x_old - x) / np.linalg.norm(x_old)
        rel_err[iter] = np.linalg.norm(x_true - x) / np.linalg.norm(x_true)
        iter_err[iter] = norma_it
    
    rel_err = rel_err[:iter]
    iter_err = iter_err[:iter]
    
    return x, iter, rel_err, iter_err


n = np.arange(2, 100)
errs_LU = np.zeros(np.size(n))
time_LU = np.zeros(np.size(n))
errs_cholesky = np.zeros(np.size(n))
time_cholesky = np.zeros(np.size(n))
iter_gauss = np.zeros(np.size(n))
errs_gauss = np.zeros(np.size(n))
time_gauss = np.zeros(np.size(n))
iter_jacobi = np.zeros(np.size(n))
errs_jacobi = np.zeros(np.size(n))
time_jacobi = np.zeros(np.size(n))

# test sulla matrice tridiagonale
j = 0
for i in n:
    c = np.eye(i) * 9
    s = np.diag(np.ones(i-1)*(-4),k=1)
    a = np.diag(np.ones(i-1)*(-4),k=-1)
    A = c + s + a
    xtrue = np.ones((i, 1))
    b = np.dot(A, xtrue)
    
    maxit = 500
    tol = 1.e-8
    x0 = np.zeros((i, 1))
    x0[0] = 1
    
    start = time.time()
    my_x, condA, err = cholesky(A, xtrue)
    end = time.time() - start
    
    time_cholesky[j] = end
    errs_cholesky[j] = err / scipy.linalg.norm(xtrue, 2)
    
    start = time.time()
    my_x, condA, err = lu_dec(A, xtrue)
    end = time.time() - start
    
    time_LU[j] = end
    errs_LU[j] = err / scipy.linalg.norm(xtrue, 2)
    
    start = time.time()
    (x, ite, relErr, errIter) = Jacobi(A, b, x0, maxit, tol, xtrue)
    end = time.time() - start
    
    iter_jacobi[j] = ite
    errs_jacobi[j] = relErr[-1]
    time_jacobi[j] = end
    
    start = time.time()
    (x1, ite1, relErr1, errIter) = gauss_seidel(A, b, x0, maxit, tol, xtrue)
    end = time.time() - start
    
    time_gauss[j] = end
    iter_gauss[j] = ite1
    errs_gauss[j] = relErr1[-1]
    
    j = j+1

plt.figure(figsize=(20, 10))
plt.title("Numero di iterazioni al variare della dimensione")
plt.plot(n, iter_gauss, label='Gauss-Seidel', color='blue', linewidth=1)
plt.plot(n, iter_jacobi, label='Jacobi', color='green', linewidth=1)
plt.legend()
plt.xlabel("dim(A)")
plt.ylabel("iterazioni")
plt.grid()
plt.show()

plt.figure(figsize=(20, 10))
plt.title("Errore relativo al variare della dimensione")
plt.plot(n, errs_gauss, color='blue', linewidth=1, marker='.')
plt.plot(n, errs_jacobi, color='green', linewidth=1, marker='.')
plt.plot(n, errs_LU, label='tempo', color='red', linewidth=1, marker='.'  )
plt.plot(n, errs_cholesky, label='tempo', color='orange', linewidth=1, marker='.'  )
plt.legend(['Gauss-Seidel', 'Jacobi', 'LU', 'Cholesky'])
plt.xlabel("dim(A)")
plt.ylabel("errore relativo")
plt.grid()
plt.show()

plt.figure(figsize=(20, 10))
plt.title('Tempo impiegato al variare della dimensione')
plt.plot(n, time_gauss, label='tempo', color='green', linewidth=1, marker='.'  )
plt.plot(n, time_jacobi, label='tempo', color='blue', linewidth=1, marker='.'  )
plt.plot(n, time_LU, label='tempo', color='red', linewidth=1, marker='.'  )
plt.plot(n, time_cholesky, label='tempo', color='orange', linewidth=1, marker='.'  )
plt.legend(['Gauss-Seidel', 'Jacobi', 'LU', 'Cholesky'])
plt.xlabel('dim(A)')
plt.ylabel('tempo')
plt.grid()
plt.show()

