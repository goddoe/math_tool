


from scipy import linalg
import numpy as np
#svd wrapper function
def svd_a_inv(a , b, full_matrices=True):
    U, s, Vh = linalg.svd(a, full_matrices)
    # print U.shape, s.shape, Vh.shape
    S = linalg.diagsvd(s, a.shape[1], a.shape[1])
    
    if full_matrices == True:
        S_inv = linalg.diagsvd(np.linalg.inv(S).diagonal(), a.shape[1], a.shape[0])
    else:
        S_inv = np.linalg.inv(S)

        ah = np.dot(Vh.transpose(), np.dot(S_inv, U.transpose()))
        s = np.dot(ah, b)
    return ah, s

#SVD population estimation
a = np.array([
    [3724900, 1930, 1]
    , [3763600, 1940, 1]
    , [3798601, 1949, 1]
    , [3841600, 1960, 1]
    , [3880900, 1970, 1]
    , [3920400, 1980, 1]
    , [3960100, 1990, 1]
    , [4000000, 2000, 1]
    , [4040100, 2010, 1]
    ])
b = np.array([2044, 2355, 2017, 2499, 3144, 3741, 4339, 4599, 4799])

#use svd
ah, s = svd_a_inv(a, b, full_matrices=False)
f = lambda x: s[0] * x**2 + s[1] * x + s[2]
print f(2020)
