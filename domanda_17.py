'''
 Deblur di immagini
Discutere La ricostruzione di un’immagine corrotta da blur e rumore, mostrando la soluzione
naive ottenuta risolvendo un problema di minimi quadrati e la soluzione ottenuta con il
metodo di regolarizzazione di Tikhonov, mostrando sia le immagini ricostruite che i grafici dei
parametri PSNR e MSE al variare del numero di iterazioni . Discutere inoltre i risultati al variare
di Lambda, parametro di regolarizzazione.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy import signal
from numpy import fft


np.random.seed(0)

# Crea un kernel Gaussiano di dimensione kernlen e deviazione standard sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    # Kernel gaussiano unidmensionale
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Kernel gaussiano bidimensionale
    kern2d = np.outer(kern1d, kern1d)
    # Normalizzazione
    return kern2d / kern2d.sum()

# Esegui l'fft del kernel K di dimensione d agggiungendo gli zeri necessari 
# ad arrivare a dimensione shape
def psf_fft(K, d, shape):
    # Aggiungi zeri
    K_p = np.zeros(shape)
    K_p[:d, :d] = K

    # Sposta elementi
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

    # Esegui FFT
    K_otf = fft.fft2(K_pr)
    return K_otf

# Moltiplicazione per A
def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))

# Moltiplicazione per A trasposta
def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))


"""
PROBLEMA TEST
"""

def degradate(fexact, kernel_size, kernel_sigma, noise_sigma):
    m,n = fexact.shape

    K = psf_fft(gaussian_kernel(kernel_size, kernel_sigma), kernel_size, [m,n])

    fblur = A(fexact, K)

    noise = np.random.normal(0, noise_sigma, size = [m,n])

    f = fblur + noise
    PSNR = metrics.peak_signal_noise_ratio(fexact, f)

    plt.figure(figsize=(30,10))
    plt.subplot(1,2,1)
    plt.imshow(fexact, cmap='gray')
    plt.title("Immagine originale")
    plt.subplot(1,2,2)
    plt.imshow(f, cmap='gray')
    plt.title(f'Immagine degradata  (PSNR: {PSNR: .2f})', fontsize=20)
    plt.show()
    
    print("PSNR: ", PSNR)
    
    return (f, PSNR, (m,n), K)


def minimize(f, grad_f, x0, x_true, step, MAXITERATION, ABSOLUTE_STOP, args=(), constant = False):
    
    def next_step(f, x,grad, args=()):#La "procedura di backtracking"  per la scelta della lunghezza del passo 

        if not isinstance(args, tuple):
            args = (args,)
              
        alpha=1.1
        rho = 0.5
        c1 = 0.25
        p=-grad
        j=0
        jmax=10
        while((f(x+alpha*p, *args) > f(x, *args) + c1*alpha*grad.T@p) and j < jmax):
             alpha = rho*alpha
             j+=1
        if(j>jmax):
            return -1
        else:
            return alpha
    
    size = np.size(x0)
    x0 = np.reshape(x0, (1, size))
    
    if not isinstance(args, tuple):
        args = (args,)
    
    x = np.zeros((size, MAXITERATION))
    norm_grad_list = np.zeros((1, MAXITERATION))
    function_eval_list = np.zeros((1, MAXITERATION))
    error_list = np.zeros((1, MAXITERATION))
    
    k = 0
    x_last = np.copy(x0)
    x[:, k] = x_last
    function_eval_list[:,k] = abs(f(x_last, *args))
    error_list[:,k] = np.linalg.norm(x_last-x_true)
    norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last, *args))
    
    while(np.linalg.norm(grad_f(x_last, *args)) > ABSOLUTE_STOP and k < MAXITERATION):
        grad = grad_f(x_last, *args)
        
        if (not constant):
            step = next_step(f, x_last, grad, *args)
        
        if (step == -1):
            print ("Non convergente")
            return -1
        
        x_last = x_last - step*grad
        
        x[:,k] = x_last
        function_eval_list[:,k] = abs(f(x_last, *args))
        error_list[:,k] = np.linalg.norm(x_last-x_true)
        norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last, *args))
        k=k+1
        
    k = k-1
    function_eval_list = function_eval_list[:,:k+1]
    error_list = error_list[:,:k+1]
    norm_grad_list = norm_grad_list[:,:k+1]
 
    return (x_last,norm_grad_list, function_eval_list, error_list, k, x)



fexact = data.coins().astype(np.float64) / 255.0
image, PSNR_deg, (m,n), K = degradate(fexact, 9, 3, 0.02)

def f(x):
    x_r = np.reshape(x, (m,n))
    res = (0.5)*(np.sum(np.square(A(x_r, K)-image)))  #l'obiettivo di minimizzazione per la ricostruzione dell'immagine degradata
    return res

def grad(x):
    x_r = np.reshape(x, (m,n))
    res = AT(A(x_r, K), K)-AT(image, K)
    res_r = np.reshape(res, m*n)
    return res_r

lamb = 0.05

def f1(x, lamb):
    x_r = np.reshape(x, (m,n))
    res = (0.5)*(np.sum(np.square(A(x_r, K)-image))) + (lamb/2)*(np.sum(np.square(x_r)))
    return res

def grad1(x, lamb):
    x_r = np.reshape(x, (m,n))
    res = AT(A(x_r, K), K)-AT(image, K) + lamb*x_r
    res_r = np.reshape(res, m*n)
    return res_r

## NAIVE

x0 = image
maxit = 50
tol = 1e-20

#res = minimize(f, x0, method='CG', jac = grad,  options={'maxiter':maxit, 'return_all':True})
(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(f, grad, x0, np.array(np.size(x0)), 0.1, maxit, tol)

PSNRs = np.zeros(k)
MSEs = np.zeros(k)

for i in range(k):
    temp = np.reshape(x[:np.size(x0), i], (m,n))
    PSNRs[i] = metrics.peak_signal_noise_ratio(fexact, temp)
    MSEs[i] = metrics.mean_squared_error(fexact, temp)

x = x[:np.size(x0), -1]
x = np.reshape(x, (m,n))

plt.figure(figsize=(30,10))
plt.subplot(1,2,1)
plt.imshow(fexact, cmap='gray')
plt.title("Immagine originale")
plt.subplot(1,2,2)
plt.imshow(x, cmap='gray')
plt.title(f'Immagine Ricostruita con metodo naive \n (PSNR: {PSNRs[-1]: .2f}, MSE: {MSEs[-1]: .2f})', fontsize=20)
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Iterazione VS PSNR")
plt.plot(PSNRs)
plt.grid()
plt.subplot(1, 2, 2)
plt.title("Iterazione VS MSE")
plt.plot(MSEs)
plt.grid()
plt.show()

#%%
## REGOLARIZZATA

x0 = image
maxit = 50

lambdas = np.linspace(0.01,0.1, 10)
PSNRS = np.zeros(lambdas.size)
MSES = np.zeros(lambdas.size)
j = 0

for i in lambdas:
    (x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(f1, grad1, x0, np.array(np.size(x0)), 0.1, maxit, tol, args=(i))

    x = x[:np.size(x0), -1]
    x = np.reshape(x, (m,n))
    PSNRS[j] = metrics.peak_signal_noise_ratio(fexact, x)
    MSES[j] = metrics.mean_squared_error(fexact, x)
    
    plt.figure(figsize=(30,10))
    plt.subplot(1,2,1)
    plt.imshow(fexact, cmap='gray')
    plt.title("Immagine originale")
    plt.subplot(1,2,2)
    plt.imshow(x, cmap='gray')
    plt.title(f'Immagine Ricostruita  (PSNR: {PSNRS[j]: .2f}, MSE: {MSES[j]: .6f}, LAMBDA: {i: .2f})', fontsize=20)
    plt.show()
    j += 1

plt.figure(figsize=(20, 10))
plt.grid()
plt.title("PSNR al variare del parametro lambda")
plt.plot(lambdas, np.full(lambdas.size, PSNR_deg), 'orange', label='Corrotta')    
plt.plot(lambdas, PSNRS, 'red', label='Ricostruzione')
plt.show()

plt.figure(figsize=(20, 10))
plt.grid()
plt.title("MSE al variare del parametro lambda")
plt.plot(lambdas[1:], MSES[1:], 'green', label='Ricostruzione')
plt.show()





