import numpy as np

def calcularLU(A):
    L, U = [],[]
    P = None
    # su código
    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)

    """
    Para aplicar el pivoteo parcial
    debo hallar la posicion del valor maximo en cada iteracion de i
    """
    for i in range(0, n-1):
        max_indice = i
        for j in range(i, n):
            max_valor = A[i][i]
            if abs( A[j][i] ) > abs(max_valor):
                max_valor = abs( A[j][i] )
                max_indice = j
            # max_index = np.argmax(np.abs(A[i:, i])) + i

        if A[max_indice][i] == 0:
            raise ValueError("La matriz es singular o requiere pivoteo.")

        if max_indice != i:
            A[[i, max_indice], :] = A[[max_indice, i], :]
            P[[i, max_indice], :] = P[[max_indice, i], :]
            if i > 0: 
                L[[i, max_indice], :i] = L[[max_indice, i], :i]

        for k in range(i+1,n):
            factor = A[j][i] / A[i][i]
            L[j][i] = factor
            for m in range(i,n):
                A[j][m] -= factor * A[i][m]
    ###########
    U = A
    return L, U, P


def inversaLU(L, U, P=None):
    Inv = []
    # su código
    n = L.shape[0]
    Inv = np.zeros((n, n))
    I = np.eye(n)

    # Resolver LY = I
    for i in range(0,n):
        Y = np.linalg.solve(L, I[:, i])
        # Resolver UX = Y
        X = np.linalg.solve(U, Y)
        Inv[:, i] = X
    ###########
    return Inv
