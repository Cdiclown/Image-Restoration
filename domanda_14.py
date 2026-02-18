'''Calcolo degli zeri di una funzione'''

import numpy as np
import matplotlib.pyplot as plt

def bisezione(xtrue, f, a, b, tol):
    eps = np.finfo(float).eps
        
    k = int(np.log2((b-a)/tol))
    
    ak = a
    bk = b
    err = np.zeros((k+1))
    err[0] = tol+1
    iterations = 0
    ck = 0
    
    if (np.sign(f(a)*f(b)) != -1):
        return ("Error", err, iterations)
    
    while (np.abs(bk-ak)>tol+eps*np.max([np.abs(b),np.abs(a)]) and iterations < k):
        iterations += 1
        ck = ak + (bk-ak)/2
        
        err[iterations] = np.abs(ck-xtrue)
        
        if (np.sign(f(ck)) == 0):
            return (ck, err, iterations)
        elif (np.sign(f(ck)) == -1 ):
            ak = ck
        elif (np.sign(f(ck)) == 1 ):
            bk = ck
        
    return (ck, err, iterations)



def newton(f, df, tolf, tolx, maxit, xTrue, x0=0):
    err=np.zeros(maxit, dtype=float)
    vecErrore=np.zeros( (maxit,1), dtype=float)
    i=0
    err[0]=tolx+1
    vecErrore[0] = np.abs(x0-xTrue)
    x=x0
    xold = x0

    while (not(np.abs(f(x)) <= tolf and np.abs(x-xold) <= tolx) and i < maxit):
        xold = x
        # if(np.sign(df(xold)) == 0):
        #     return ("Error", i, err, vecErrore)
        x = xold - (f(xold)/df(xold))
        i += 1
        err[i] = err[i-1] + 1
        vecErrore[i] = np.abs(x-xTrue)
    
    return (x, i, err, vecErrore[:i])



def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
  
  err=np.zeros(maxit+1, dtype=np.float64)
  vecErrore=np.zeros(maxit+1, dtype=np.float64)
  
  i= 0
  err[0]=tolx+1
  vecErrore[0] = np.abs(x0-xTrue)
  x = x0

  while (not(np.abs(f(x)) <= tolf and np.abs(x-x0) <= tolx) and i < maxit):
    x0 = x
    x = g(x0)
    i += 1
    vecErrore[i] = np.abs(x-xTrue)
    err[i] = err[i-1] + 1
    
  err = err[:i]
  vecErrore = vecErrore[:i]
  return (x, i, err, vecErrore)


f = lambda x: x-x**(1/3)-2
df = lambda x: 1-(1/(3*x**(2/3)))

a = 3
b = 5
g1 = lambda x: x**(1/3)+2
xtrue = 3.52137970680457
tol = 10**(-6)

tolf = 10**(-6)
tolx = 10**(-10)
maxit = 100

(x, err, iterations) = bisezione(xtrue, f, a , b,tol)

print("Soluzione trovata per f con il metodo di bisezione: ", x)
print("Numero di iterazioni effettuate dal metodo di bisezione: ", iterations)

x_plot = np.linspace(a,b,100)

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.title("Funzione")
plt.plot(x_plot, f(x_plot))
plt.grid()
plt.plot(xtrue,0,'o')
plt.plot(x,f(x),'o')
plt.subplot(1,2,2)
plt.title("Errore con metodo di bisezione")
plt.plot(err)
plt.grid()
plt.show()


(x, iterations, err, vecErr) = newton(f, df, tolf,tolx, 100, xtrue, x0=1.0)

print("Soluzione trovata per f con il metodo di Newton: ", x)
print("Numero di iterazioni effettuate dal metodo di Newton: ", iterations)

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.title("Funzione")
plt.plot(x_plot, f(x_plot))
plt.grid()
plt.plot(xtrue,0,'o')
plt.plot(x,f(x),'o')
plt.subplot(1,2,2)
plt.title("Errore con metodo di Newton")
plt.plot(vecErr[:iterations])
plt.grid()
plt.show()
                                        
(x, i, err, vecErr) = succ_app(f, g1, tolx, tolf, maxit, xtrue)

print("Found value with succ. appr. method for f1 with g1: ", x, "\nIterations: ", i)

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.title("Funzione")
plt.plot(x_plot, f(x_plot))
plt.grid()
plt.plot(xtrue,0,'o')
plt.plot(x,f(x),'o')
plt.subplot(1,2,2)
plt.title("Errore con metodo delle appr. succ. (g1)")
plt.plot(vecErr)
plt.grid()
plt.show()

