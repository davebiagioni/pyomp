# pyomp
Orthogonal Matching Pursuit (Python: NumPy + SciPy)

This is a simple implementation that uses scipy.optimize.nnls and numpy.  It 
computes non-negative solutions by default, since that's what I needed it for, but can 
also be used to find unconstrained solutions by setting `nonneg=False`.  The module favors
convenience over performance, but performs reasonably well for many problems.

The `Result` object returned by `omp` is a self-contained expression of the problem 
that was solved and stores both the (optionally standardized) inputs, runtime parameters, 
details of the iteration, residual, and reconstructed signal.

The Jupyter notebook [`examples.ipynb`](https://github.com/davebiagioni/omp/blob/master/examples.ipynb) contains step by step examples.
