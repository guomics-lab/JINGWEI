"""Single-matrix iterative imputation without retaining obsolete regressors."""

from importlib.metadata import version


def make_imputer(*, optimized, **options):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    if not optimized or version("scikit-learn") != "1.6.1":
        return IterativeImputer(**options), False

    class FitCounter:
        count = 0

        def append(self, fitted):
            # fit_transform has already used this model to fill its column and
            # fits a fresh model next time. Only transform(new_data) needs it.
            self.count += 1

        def __len__(self):
            return self.count

    class SingleMatrixImputer(IterativeImputer):
        @property
        def imputation_sequence_(self):
            return self._fit_counter

        @imputation_sequence_.setter
        def imputation_sequence_(self, value):
            if value != []:
                raise RuntimeError("unsupported IterativeImputer sequence initialization")
            self._fit_counter = FitCounter()

        def transform(self, X):
            raise RuntimeError("single-matrix imputer does not retain models for new data")

    return SingleMatrixImputer(**options), True
