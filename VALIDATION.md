# Validation record

The package was checked before release with the following local tests:

1. Python syntax compilation for every added Python module.
2. SPA smoke test covering forward, backward, prototype update and orthogonal rotation.
3. Four unit tests covering Simplex ETF geometry, Procrustes recovery, SPA loss/head shapes and ICBHI metrics.
4. Bash syntax validation for every shell script.
5. Overlay installer test against a mock PAFA directory.
6. Result aggregation test using synthetic PAFA and PAFA+SPA metric files.

Local unit-test result:

```text
Ran 4 tests
OK
```

A full BEATs training run was not executed in the packaging environment because the ICBHI archive, the `BEATs_iter3_plus_AS2M.pt` checkpoint and a CUDA GPU were not available there. The required end-to-end installation checks are included in `tools/verify_install.py`, and the first-run procedure is documented in `README.md`.
