# TurboQuant mathematical reference

Reference: https://arxiv.org/pdf/2504.19874v1, Algorithms 1 and 2.

This folder is an independent NumPy reference written for Skylight. It does not
contain official TurboQuant source or code from 0xSero/turboquant.

The oracle receives explicit rotation/projection matrices and codebooks. It
computes norm, lookup, original-basis reconstruction and QJL correction in FP64.
Payload bytes are decoded independently with Python integers.

Codebook conditional means use an analytic first moment and SciPy's regularized
incomplete Beta function, independent of the production theta-quadrature solver.
SciPy is installed by the project's existing scikit-learn dependency.

Conventions:

- Rows rotate as u @ R; coarse reconstruction uses y_hat @ R.T.
- Projection S has rows of independent standard-normal samples.
- Residual signs are sign(r @ S.T); zero uses +1.
- QJL correction is sqrt(pi/2)/d * gamma * (signs @ S).
- Gamma is the unit-vector residual norm; rho restores original magnitude.
- Lookup midpoint ties choose the lower index.
- Float32 runtime arithmetic is compared with explained roundoff bounds;
  explicit analytic examples and integer payloads receive exact checks.

vectors.json contains a fixed d=3 uniform-codebook example, generated from
these written equations. It is a checkable mathematical fixture, not an authors'
release or a pretrained-model benchmark.
