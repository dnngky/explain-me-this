import numpy as np
import math

class NonInvertibleError(Exception):
    pass


# Counts the number of zeros in a row/col of a matrix
def count_zeros(row):

    total = 0
    for entry in row:
        if entry == 0: total += 1
    return total


# Determines the most optimal axis to calculate the determinant
def get_optimal_axis(A):

    most_zeros = 0
    optimal_axis = (0, 0)

    for i in range(A.shape[0]):
        
        row = A[i]
        if count_zeros(row) > most_zeros:
            most_zeros = count_zeros(row)
            optimal_axis = (i, 0)
        
        col = A.T[i]
        if count_zeros(col) > most_zeros:
            most_zeros = count_zeros(col)
            optimal_axis = (i, 1)

    return optimal_axis


# Checks if a matrix is triangular
def is_triangular(A):

    left_row = []
    right_row = []

    for i in range(A.shape[0]):
        left_row.extend(A[i][0:i])
        right_row.extend(A[i][i+1:len(A)])

    return count_zeros(left_row) == len(left_row) or \
        count_zeros(right_row) == len(right_row)


# Returns the A_ij matrix
def get_Aij(A, i, j):

    return np.delete(np.delete(A, i, 0), j, 1)


# Calculates the cofactor
def compute_cofactor(A, i, j):
    
    return (-1)**(i+j+2) * det(get_Aij(A, i, j), print_msg=False)


# Calculates determinant using recursion
def det(A, print_msg=False):

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square!")

    # Returns the product of diagonal entries if matrix is triangular
    if is_triangular(A):

        if print_msg: print("This matrix is triangular!")
        det_A = 1
        
        for i in range(A.shape[0]):
            det_A *= A[i][i]
        
        return det_A

    # Returns (ad - bc) if the matrix is 2 x 2
    if (A.shape[0] == 2):
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]

    # Else, calculates determinant using recursion
    det_A = 0
    m, axis = get_optimal_axis(A)

    for n in range(A.shape[0]):

        if axis == 0:
            a_ij = A[m][n]
            A_ij = get_Aij(A, m, n)
            msg = f"\ni,j = [{m},{n}]\n"

        if axis == 1:
            a_ij = A[n][m]
            A_ij = get_Aij(A, n, m)
            msg = f"\ni,j = [{n},{m}]\n"

        if a_ij == 0:
            msg += f"a_ij: 0\nA_ij:\n{A_ij}\n(i,j)-cofactor: 0"
            continue

        cofactor = (-1)**(m+n+2) * det(A_ij, print_msg)
        det_A +=  a_ij * cofactor
        msg += f"a_ij: {a_ij}\nA_ij:\n{A_ij}\n(i,j)-cofactor: {cofactor}"

        if print_msg: print(msg)

    return det_A


# Calculates the inverse of A
def inverse(A):

    if det(A) == 0:
        raise NonInvertibleError("This matrix is not invertible!")

    if A.shape == (2, 2):
        return (1/det(A))*np.array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]])

    C = np.empty(A.shape)

    for i in range(len(A)):
        for j in range(len(A[i])):
            C[i][j] = compute_cofactor(A, i, j)

    return (1/det(A))*C.T


if __name__ == "__main__":

    # Adjust the numbers inside A or add more rows as needed
    A = [[ 1, 0, 1],
         [-4, 1,-1],
         [ 6,-2, 1]]

    print_msg = False
    
    A = np.array(A)
    print(f"A =\n{A}")
    print(f"\n|A| = {det(A, print_msg)}\n")
    print(f"A^-1 =\n{inverse(A)}\n")