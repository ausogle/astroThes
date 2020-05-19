import numpy as np
import scipy.linalg as la
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import get_lapack_funcs, strtrs



a = np.array([[1.5024e-5, -5.7278e-6, 2.1893e-6, .2237, -.3760, .1411],
              [-5.7278e-6, 2.2488e-6, -6.6191e-7, -.0853, .1424, -.0564],
              [2.1893e-6, -6.6191e-7, 7.7772e-7, 0.0326, -.0574, .0136],
              [.2237, -.0853, .0326, 3.3304e3, -5.5987e3, 2.1011e3],
              [-.3760, .1424, .0574, -5.5987e3, 9.4267e3, -3.4925e3],
              [.1411, -.0564, 0.0136, 2.1011e3, -3.4925e3, 1.4306e3]])

b = np.array([.004, -.0016, 3.5733e-4, 59.6263, -98.9440, 41.0392])


#
p, l, u = la.lu(a)
# inva = la.inv(u) @ la.inv(l) @ p.T
#
# x = inva@b
#
# eye = inva @ a
# print("Should ebe eye")
# print(eye)
#
# print("residual")
# print(a@x-b)
#
# x_help = solve_triangular(u, la.inv(l)@p.T@b)
# residual2 = a@x-b
# print("\nImprovemnt?")
# print(residual2)

trtrs, = get_lapack_funcs(('trtrs',), (u, la.inv(l)@p.T@b))
x, info = trtrs(a, b, lower=False)

print("x from get_lapack_function call")
print(x)
print("residual from this operation")
print(a@x-b)

print("\nx from lapack function directly")
x = strtrs(u, la.inv(l)@ p.T@ b, lower=False)
print(x)
print("residual")
print(a@x[0]-b)




def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, debug=None, check_finite=True):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in `a x = b`
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x  = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and
        will not be referenced.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system `a x = b`.  Shape of return matches `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular

    Notes
    -----
    .. versionadded:: 0.9.0

    Examples
    --------
    Solve the lower triangular system a x = b, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> from scipy.linalg import solve_triangular
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x)  # Check the result
    array([ 4.,  2.,  4.,  2.])

    """

    # Deprecate keyword "debug"
    if debug is not None:
        warn('Use of the "debug" keyword is deprecated '
             'and this keyword will be removed in the future '
             'versions of SciPy.', DeprecationWarning, stacklevel=2)

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    if a1.shape[0] != b1.shape[0]:
        raise ValueError('incompatible dimensions')
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if debug:
        print('solve:overwrite_b=', overwrite_b)
    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
    trtrs, = get_lapack_funcs(('trtrs',), (a1, b1))
    if a1.flags.f_contiguous or trans == 2:
        x, info = trtrs(a1, b1, overwrite_b=overwrite_b, lower=lower,
                        trans=trans, unitdiag=unit_diagonal)
    else:
        # transposed system is solved since trtrs expects Fortran ordering
        x, info = trtrs(a1.T, b1, overwrite_b=overwrite_b, lower=not lower,
                        trans=not trans, unitdiag=unit_diagonal)

    if info == 0: #info is if there is an issue such as singular
        return x
