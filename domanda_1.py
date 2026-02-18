"""
 Risoluzione di sistemi lineari con fattorizzazione LU. Dato un problema test di dimensione
variabile la cui soluzione esatta sia il vettore x di tutti elementi unitari e b il termine noto
ottenuto moltiplicando la matrice A per la soluzione x discutere:
• il numero di condizione (o una stima di esso)
• la soluzione del sistema lineare Ax=b con la fattorizzazione LU con pivoting.
Testare su una matrice di numeri casuali A generata con la funzione randn di Matlab, (n variabile
fra 10 e 1000)
"""

import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt


# funizione che fa la fattorizzazione LU con pivoting
def solve_lu(A, b):
    lu, piv = LA.lu_factor(A)
    x = LA.lu_solve((lu, piv), b)
    return x

# Creazione dati e problema test
n = np.random.randint(10, 1000)
A = np.random.randn(n, n)
x= np.ones((n, 1))
b = np.matmul(A, x)

# Numero di condizione
condA = np.linalg.cond(A, 2)
print('K(A) =', condA)

# Fattorizzazione LU con pivoting
x_solution = solve_lu(A, b)
print('my_x =\n', x_solution )
print('norm =', np.linalg.norm(x - x_solution, 'fro'))

# Grafici
K_A = np.zeros((20, 1))
Err = np.zeros((20, 1))

for i, n in enumerate(np.arange(10, 30)):
    A = np.random.randn(n, n)
    x = np.ones((n, 1))
    b = np.matmul(A, x)
    
    K_A[i] = np.linalg.cond(A, 2)
    
    x_solution = solve_lu(A, b)

    
    Err[i] = np.linalg.norm(x_solution  - x, 2) / np.linalg.norm(x, 2)

dim = np.arange(10, 30)

plt.plot(dim, K_A)
plt.grid()
plt.title('Condizionamento di A al variare di dim(A)')
plt.xlabel('dim(A)')
plt.ylabel('K(A)')
plt.show()

plt.plot(dim, Err)
plt.grid()
plt.title('Errore relativo al variare di dim(A)')
plt.xlabel('dim(A)')
plt.ylabel('Err = ||my_x - x|| / ||x||')
plt.show()
