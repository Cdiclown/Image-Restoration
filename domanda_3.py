'''
Risoluzione di sistemi lineari con fattorizzazione di Cholesky (II). Dato un problema test di
dimensione variabile la cui soluzione esatta sia il vettore x di tutti elementi unitari e b il
termine noto ottenuto moltiplicando la matrice A per la soluzione x discutere:
• il numero di condizione (o una stima di esso)
• la soluzione del sistema lineare Ax=b con la fattorizzazione di Cholesky.
Testare sulla matrice tridiagonale simmetrica e definita positiva avente sulla diagonale elementi
uguali a 9 e quelli sopra e sottodiagonali uguali a -4 (variare n)

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

# creazione dati e problema test

i = 4
c = np.eye(i) * 9
s = np.diag(np.ones(i-1)*(-4),k=1)
a = np.diag(np.ones(i-1)*(-4),k=-1)
A = c + s + a
x = np.ones((i, 1))

cholesky(A, x, print_data=True)

#Grafici 

n = np.arange(2,15, 1)
conds = np.zeros(np.size(n))
errs = np.zeros(np.size(n))

j = 0
for i in n:
    c = np.eye(i) * 9
    s = np.diag(np.ones(i-1)*(-4),k=1)
    a = np.diag(np.ones(i-1)*(-4),k=-1)
    A = c + s + a
    x = np.ones((i, 1))
    
    my_x, condA, err = cholesky(A, x)
    
    conds[j] = condA
    errs[j] = err / scipy.linalg.norm(x, 2)
    
    j = j+1   

# Grafico del numero di condizione vs dim
plt.plot(n, conds)
plt.title('Condizionamento di A al variare di dim(A) ')
plt.xlabel('dim(A)')
plt.ylabel('K(A)')
plt.grid()
plt.show()

# Grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(n, errs)
plt.title('Errore relativo al variare di dim(A) ') 
plt.xlabel('dim(A)')
plt.ylabel('Err = ||my_x-x|| / ||x||')
plt.grid()
plt.show()