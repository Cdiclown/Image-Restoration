'''Compressione di un'immagine tramite SVD'''

from skimage import data
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

def compress(A, p_max):
    U, s, Vh = np.linalg.svd(A, False)
    
    Ap = np.zeros(A.shape)
    
    err_rel = np.zeros((p_max))
    c = np.zeros((p_max)) #coefficenti approsimazioni 
    
    for i in range(0,p_max):
        
        
        Ap += np.outer(U[:, i], Vh[i, :]) * s[i]
                
        err_rel[i] = (np.linalg.norm(A-Ap,2))/(np.linalg.norm(A,2))
        if (i == 0):
            c[0] = 1000
        else:
            c[i] = (1/i)*np.min(A.shape)-1
    
    print('\n')
    print('L\'errore relativo della ricostruzione di A è', err_rel[-1])
    print('Il fattore di compressione è c=', c[-1])
    
    plt.figure(figsize=(20, 10))
    
    fig1 = plt.subplot(1, 2, 1)
    fig1.imshow(A, cmap='gray')
    plt.title('True image')
    
    fig2 = plt.subplot(1, 2, 2)
    fig2.imshow(Ap, cmap='gray')
    plt.title('Reconstructed image with p =' + str(p_max))
    
    plt.figure(figsize=(20, 10))
    fig1 = plt.subplot(1, 2, 1)
    fig1.plot(err_rel, 'o-')
    plt.title('Errore relativo al variare di p')
    plt.grid()
    fig2 = plt.subplot(1, 2, 2)
    fig2.plot(c, 'o-')
    plt.title('Fattore di compressione al variare di p')
    plt.grid()
    plt.show()

A = data.coins()
compress(A,10)

A1 = skimage.io.imread("monalisa.jpg", True)
compress(A1,10)

A2 = skimage.io.imread("moon.jpg", True)
compress(A2,10)