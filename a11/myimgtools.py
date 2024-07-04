import numpy as np
from scipy import misc
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import scipy.sparse as sp
import scipy.sparse.linalg


def make_derivatives2D(M, N):
    
    # y-derivatives
    row = np.zeros(2*M*N);
    col = np.zeros(2*M*N); 
    val = np.zeros(2*M*N); 
    ctr = 1;
    for x in range(0,N):
        for y in range(0,M-1):
            row[ctr] = x*M + y;
            col[ctr] = x*M + y;
            val[ctr] = -1.0;
            ctr = ctr + 1;
            
            row[ctr] = x*M + y;
            col[ctr] = x*M + y+1;
            val[ctr] = 1.0;
            ctr = ctr + 1;
    
    Ky = csr_matrix((val, (row, col)), shape=(M*N, M*N));

    # x-derivatives
    row = np.zeros(2*M*N);
    col = np.zeros(2*M*N); 
    val = np.zeros(2*M*N); 
    ctr = 1;
    for y in range(0,M):
        for x in range(0,N-1):
            row[ctr] = x*M + y;
            col[ctr] = x*M + y;
            val[ctr] = -1.0;
            ctr = ctr + 1;
            
            row[ctr] = x*M + y;
            col[ctr] = (x+1)*M + y;
            val[ctr] = 1.0;
            ctr = ctr + 1;
   

    Kx = csr_matrix((val, (row, col)), shape=(M*N, M*N));

    # x- and y-derivative (discrete gradient)
    K = sp.vstack([Kx,Ky]);

    return K;


# convert rgb image to gray scale image
def rgb2gray(img):
    if (img.shape[2]>1):
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        return img;



