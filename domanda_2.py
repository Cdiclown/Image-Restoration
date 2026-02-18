'''
Risoluzione di sistemi lineari con fattorizzazione di Cholesky (I). Dato un problema test di
dimensione variabile la cui soluzione esatta sia il vettore x di tutti elementi unitari e b il
termine noto ottenuto moltiplicando la matrice A per la soluzione x discutere:
• il numero di condizione (o una stima di esso)
• la soluzione del sistema lineare Ax=b con la fattorizzazione di Cholesky.
Testare sulla matrice di Hilbert, (n variabile fra 2 e 15).
'''



import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt 


# Numero di condizione
def Condizione(A):
    condA = np.linalg.cond(A, 2)
    return condA

# Fattorizzazione di cholesky
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


# Creazione dati e problema test
n = np.random.randint(2, 15)
A = scipy.linalg.hilbert(n)
x = np.ones((n, 1))
b = np.matmul(A, x)

# Grafici
cholesky(A, x, True)

n = np.arange(2, 15)
conds = np.zeros(np.size(n))
errs = np.zeros(np.size(n))
j = 0

for i in n:
    A = scipy.linalg.hilbert(i)
    x = np.ones((i, 1))
    my_x, condA, err = cholesky(A, x)
    conds[j] = condA
    errs[j] = err / scipy.linalg.norm(x, 2)
    j = j + 1   

# Grafico del numero di condizione vs dim
plt.plot(n, conds)
plt.title('Condizionamento di A al variare di dim(A) - Hilbert')
plt.xlabel('dim(A)')
plt.ylabel('K(A)')
plt.grid()
plt.show()

# Grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(n, errs)
plt.title('Errore relativo al variare di dim(A) - Hilbert') 
plt.xlabel('dim(A)')
plt.ylabel('Err = ||my_x-x|| / ||x||')
plt.grid()
plt.show()
