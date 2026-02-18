'''Metodo del gradiente'''

import numpy as np
import matplotlib.pyplot as plt

def next_step(f, x,grad): #La "procedura di backtracking"  per la scelta della lunghezza del passo 
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

def f(x):
    return 10*pow(x[0]-1,2)+pow(x[1]-2,2)

def grad_f(x):
    return np.array([20*(x[0]-1),2*(x[1]-2)])

step=0.1
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5
x0 = np.array([3,-5])
x_true=np.array([1,2])

(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(f, grad_f, x0,x_true,step,MAXITERATIONS, ABSOLUTE_STOP)

v_x0 = np.linspace(-5,5,500)
v_x1 = np.linspace(-5, 5, 500)
x0v,x1v = np.meshgrid(v_x0,v_x1)
z = f(np.array([x0v,x1v]))

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x0v, x1v, z, cmap='viridis')
ax.set_title('Surface plot')
plt.show()

contours = plt.contour(x0v, x1v, z, levels=100)
plt.plot(x[0,0:k], x[1,0:k], '.-')
plt.show()

# Iterazioni vs Norma Gradiente
plt.figure()
plt.plot(norm_grad_list.T, '.-', color='red')
plt.title('Iterazioni vs Norma Gradiente')

#Errore vs Iterazioni
plt.figure()
plt.plot(error_list.T, '.-', color='blue')
plt.title('Errore vs Iterazioni')

#Iterazioni vs Funzione Obiettivo
plt.figure()
plt.plot(function_eval_list.T, '.-', color='green')
plt.title('Iterazioni vs Funzione Obiettivo')

(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(f, grad_f, x0,x_true,step,100, ABSOLUTE_STOP, constant=True)

v_x0 = np.linspace(-5,5,500)
v_x1 = np.linspace(-5, 5, 500)
x0v,x1v = np.meshgrid(v_x0,v_x1)
z = f(np.array([x0v,x1v]))

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x0v, x1v, z, cmap='viridis')
ax.set_title('Surface plot')
plt.show()

contours = plt.contour(x0v, x1v, z, levels=100)
plt.plot(x[0,0:k], x[1,0:k], '.-')
plt.show()

# Iterazioni vs Norma Gradiente
plt.figure()
plt.plot(norm_grad_list.T, '.-', color='red')
plt.title('Iterazioni vs Norma Gradiente (alpha costante)')

#Errore vs Iterazioni
plt.figure()
plt.plot(error_list.T, '.-', color='blue')
plt.title('Errore vs Iterazioni (alpha costante')

#Iterazioni vs Funzione Obiettivo
plt.figure()
plt.plot(function_eval_list.T, '.-', color='green')
plt.title('Iterazioni vs Funzione Obiettivo (alpha costante)')








































