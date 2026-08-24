import numpy as np


_PAULI_0 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
_PAULI_1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_PAULI_3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class InternalStructure:
    __slots__ = ("_dim", "_basis", "_product_cache")

    def __init__(self, basis):
        arr = np.asarray(basis, dtype=np.complex128)
        if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
            raise ValueError(f"basis must have shape (m, d, d), got {arr.shape}")
        self._basis = arr
        self._dim = int(arr.shape[1])
        self._product_cache = {(): np.eye(self._dim, dtype=np.complex128)}
        self._product_cache[(0,)] = arr[0]

    @property
    def dim(self):
        return self._dim

    @property
    def num_matrices(self):
        return self._basis.shape[0]

    def matrix(self, i):
        return self._basis[i]

    def product(self, indices):
        key = tuple(int(i) for i in indices)
        cached = self._product_cache.get(key)
        if cached is not None:
            return cached
        if not key:
            out = np.eye(self._dim, dtype=np.complex128)
        else:
            out = self._basis[key[0]].copy()
            for k in key[1:]:
                out = out @ self._basis[k]
        self._product_cache[key] = out
        return out


def clifford_basis(num_pairs):
    n = int(num_pairs)
    if n < 0:
        raise ValueError("num_pairs must be non-negative")
    basis = [_PAULI_0, _PAULI_1, _PAULI_2, _PAULI_3]
    for _ in range(n - 1):
        new = [np.kron(_PAULI_0, basis[0])]
        new.extend(np.kron(_PAULI_3, b) for b in basis[1:])
        new.append(np.kron(_PAULI_1, basis[0]))
        new.append(np.kron(_PAULI_2, basis[0]))
        basis = new
    return InternalStructure(np.stack(basis, axis=0))


def trivial_internal():
    return InternalStructure(np.eye(1, dtype=np.complex128).reshape(1, 1, 1))

