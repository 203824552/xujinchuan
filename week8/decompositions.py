
import numpy as np
import tensorly as tl
import tqdm
#SVD分界
A = np.array([[1, 2], [2, 3], [3, 4]])
U, s, Vt = np.linalg.svd(A)

print("U = \n", U)
print("s = \n", s)
print("Vt = \n", Vt)

#CP分解

tl.set_backend('numpy')

a = np.array([[1,3,5,7,8],[8,4,6,2,10]])
b = np.array([[5,13,55,17,18],[58,14,46,12,1]])
c = np.array([[14,1,5,17,18],[58,14,46,12,1]])

X = np.einsum('i,j,k->ijk',a[0],b[0],c[0])
X += np.einsum('i,j,k->ijk',a[1],b[1],c[1])

def CP_ALS(X, r=1, max_iter=100, err=1e-10):
    N = tl.ndim(X)
    # random initialize
    A = []
    for n in range(N):
        A.append(tl.tensor(np.random.random((X.shape[n], r))))
    lbd = tl.ones(r)

    for epoch in range(max_iter):
        for n in range(N):
            V = np.ones((r,r))
            for i in range(N):
                if i != n:
                    V = A[i].T@A[i] * V

            A[n] = tl.unfold(X,mode=n)@tl.tenalg.khatri_rao(A, skip_matrix=n)@np.linalg.pinv(V)

        X_pred = tl.fold(A[0]@tl.tenalg.khatri_rao(A,skip_matrix = 0).T,mode=0,shape=X.shape)
        error = tl.norm(X-X_pred)
        print("epoch:",epoch,",error:",error)
        if error<err:
            break
    return A
A = CP_ALS(X,r=2,max_iter=1000)
