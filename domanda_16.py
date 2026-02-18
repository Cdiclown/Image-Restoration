'''Metodo del gradiente'''

import numpy as np
import matplotlib.pyplot as plt

def next_step(f, x,grad): # backtracking procedure for the choice of the steplength
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  while((f(x+alpha*p) > f(x) + c1*alpha*grad.T@p) and j < jmax):
      alpha = rho*alpha
      j+=1
  if(j>jmax):
      return -1
  else:
      return alpha


def minimize(f, grad_f, x0, x_true, step, MAXITERATION, ABSOLUTE_STOP, constant = False):
    
    size = np.size(x0)
    
    x = np.zeros((size, MAXITERATION))
    norm_grad_list = np.zeros((1, MAXITERATION))
    function_eval_list = np.zeros((1, MAXITERATION))
    error_list = np.zeros((1, MAXITERATION))
    
    k = 0
    x_last = np.copy(x0)
    x[:, k] = x_last
    function_eval_list[:,k] = abs(f(x_last))
    error_list[:,k] = np.linalg.norm(x_last-x_true)
    norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last))
    
    while(np.linalg.norm(grad_f(x_last)) > ABSOLUTE_STOP and k < MAXITERATION):
        grad = grad_f(x_last)
        
        if (not constant):
            step = next_step(f, x_last, grad)
        
        if (step == -1):
            print ("Non convergente")
            return -1
        
        x_last = x_last - step*grad
        
        x[:,k] = x_last
        function_eval_list[:,k] = f(x_last)
        error_list[:,k] = np.linalg.norm(x_last-x_true)
        norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last))
        k=k+1
        
    k = k-1
    function_eval_list = function_eval_list[:,:k+1]
    error_list = error_list[:,:k+1]
    norm_grad_list = norm_grad_list[:,:k+1]
    
    print('iterations=',k)
    print('last guess: x= ', x[0, :size])
 
    return (x_last,norm_grad_list, function_eval_list, error_list, k, x)

#funzione

lamb = 0

def f(x):
    b = np.ones(x.size)
    return np.linalg.norm(x - b)**2+lamb*np.linalg.norm(x)**2
    
def grad_f(x):
    b = np.ones(x.size)
    return 2*(x - b)+2*lamb*x


step=0.03
MAXITERATIONS=2000
ABSOLUTE_STOP=1.e-5
x0 = np.array((3,-5, 2, 0))
x_true = np.zeros(np.size(x0))

(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(f, grad_f, x0,x_true,step,MAXITERATIONS, ABSOLUTE_STOP)

# Iterazioni vs Norma Gradiente
plt.figure()
plt.plot(norm_grad_list.T, '.-', color='red')
plt.title('Iterazioni vs Norma Gradiente')

#Iterazioni vs Funzione Obiettivo
plt.figure()
plt.plot(function_eval_list.T, '.-', color='green')
plt.title('Iterazioni vs Funzione Obiettivo')


(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(f, grad_f, x0,x_true,step,100, ABSOLUTE_STOP, constant=True)

# Iterazioni vs Norma Gradiente
plt.figure()
plt.plot(norm_grad_list.T, '.-', color='red')
plt.title('Iterazioni vs Norma Gradiente')

#Iterazioni vs Funzione Obiettivo
plt.figure()
plt.plot(function_eval_list.T, '.-', color='green')
plt.title('Iterazioni vs Funzione Obiettivo')









































