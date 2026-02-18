'''

Risoluzione di sistemi lineari con metodo iterativo. Dato un problema test di dimensione
variabile la cui soluzione esatta sia il vettore x di tutti elementi unitari e b il termine noto
ottenuto moltiplicando la matrice A per la soluzione x discutere:
• la soluzione del sistema lineare Ax=b con i metodi iterativi di Jacobi e gauss Sidel al
variare del punto iniziale e della tolleranza per il criterio di arresto.
• Il numero di iterazioni effettuate al variare della dimensione n del sistema (grafico
del numero di iterazioni al variare di n).
Testare sulla matrice tridiagonale simmetrica e definita positiva positiva avente sulla diagonale
elementi uguali a 9 e quelli sopra e sottodiagonali uguali a -4 (variare n).
'''
import numpy as np
import matplotlib.pyplot as plt

# Definizione dei metodi iterativi

def Jacobi(A, b, x0, max_iter, tol, x_true):
    dim = np.size(x0)
    iter = 0
    x = x0.copy()
    norma_it = 1 + tol  # Soglia di convergenza dell'errore relativo
    rel_err = np.zeros((max_iter, 1)) # i valori dell'errore relativo nelle iterazioni
    iter_err = np.zeros((max_iter, 1)) #valori dell'errore assoluto  nelle iterazioni
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


# Creazione dati e problema test

dim = 20

c = np.eye(dim) * 9
s = np.diag(np.ones(dim-1)*(-4), k=1)
a = np.diag(np.ones(dim-1)*(-4), k=-1)
A = c + s + a

x_true = np.ones((dim, 1))
b = np.dot(A, x_true)

# Metodi iterativi
x0 = np.zeros((dim, 1))
x0[0] = 1

max_iter = 200
tol = 1.e-8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A, b, x0, max_iter, tol, x_true) 
(xGS, kGS, relErrGS, errIterGS) = gauss_seidel(A, b, x0, max_iter, tol, x_true)

print('Soluzione calcolata da Jacobi:')
for e in range(dim):
    print('%0.2f' % xJacobi[e])

print('Soluzione calcolata da Gauss-Seidel:')
for e in range(dim):
    print('%0.2f' % xGS[e])

# Grafici

# Confronto grafico degli errori relativi
rangeJabobi = range(0, kJacobi)
rangeGS = range(0, kGS)

plt.figure(figsize=(20, 10))
plt.plot(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='.')
plt.plot(rangeGS, relErrGS, label='Gauss-Seidel', color='red', linewidth=1, marker='.')
plt.legend(loc='upper right')
plt.xlabel('Iterazioni')
plt.ylabel('Errore relativo')
plt.grid()
plt.title('Errore relativo durante le iterazioni - Tridiagonale')
plt.show()

# Comportamento al variare di N

dims = np.arange(10, 200, 10)

ErrRelF_J = np.zeros(np.size(dims))
ErrRelF_GS = np.zeros(np.size(dims))

ite_J = np.zeros(np.size(dims))
ite_GS = np.zeros(np.size(dims))

for i, dim in enumerate(dims):
    c = np.eye(dim) * 9
    s = np.diag(np.ones(dim-1)*(-4), k=1)
    a = np.diag(np.ones(dim-1)*(-4), k=-1)
    A = c + s + a

    x_true = np.ones((dim, 1))
    b = np.dot(A, x_true)

    x0 = np.zeros((dim, 1))
    x0[0] = 1

    (xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A, b, x0, max_iter, tol, x_true)
    (xGS, kGS, relErrGS, errIterGS) = gauss_seidel(A, b, x0, max_iter, tol, x_true)

    ErrRelF_J[i] = relErrJacobi[-1]
    ErrRelF_GS[i] = relErrGS[-1]

    ite_J[i] = kJacobi
    ite_GS[i] = kGS

plt.figure(figsize=(20, 10))
plt.plot(dims, ErrRelF_J, color='blue', linewidth=1)
plt.plot(dims, ErrRelF_GS, color='red', linewidth=1)
plt.legend(['Jacobi', 'Gauss-Seidel'])
plt.xlabel('Dim(A)')
plt.ylabel('Errore relativo')
plt.grid()
plt.title('Errore relativo finale al variare di Dim(A) - Tridiagonale')
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(dims, ite_J, color='blue', linewidth=1)
plt.plot(dims, ite_GS, color='red', linewidth=1)
plt.legend(['Jacobi', 'Gauss-Seidel'])
plt.xlabel('Dim(A)')
plt.ylabel('Iterazioni')
plt.grid()
plt.title('Numero di iterazioni al variare di Dim(A) - Tridiagonale')
plt.show()
