import numpy as np


class SiteRegistry:
    __slots__ = ("_n", "_labels")

    def __init__(self, n, labels):
        self._n = int(n)
        self._labels = {str(k): np.asarray(v) for k, v in labels.items()}
        for k, v in self._labels.items():
            if v.shape != (self._n,):
                raise ValueError(f"label '{k}' has shape {v.shape}, expected ({self._n},)")

    @property
    def n(self):
        return self._n

    def __len__(self):
        return self._n

    @property
    def label_names(self):
        return tuple(self._labels.keys())

    def labels(self, name):
        return self._labels[name]

    def has_label(self, name):
        return name in self._labels

    def mask(self, **eq):
        m = np.ones(self._n, dtype=bool)
        for k, v in eq.items():
            arr = self._labels[k]
            if np.ndim(v) == 0:
                m &= (arr == v)
            else:
                m &= np.isin(arr, v)
        return m

    def select(self, **eq):
        return np.nonzero(self.mask(**eq))[0]

    def with_extra_labels(self, **new):
        merged = dict(self._labels)
        for k, v in new.items():
            arr = np.asarray(v)
            if arr.shape != (self._n,):
                raise ValueError(f"label '{k}' has shape {arr.shape}, expected ({self._n},)")
            merged[k] = arr
        return SiteRegistry(self._n, merged)

