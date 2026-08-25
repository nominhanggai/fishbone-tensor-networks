# Benchmarks and regression baselines

The benchmark scripts are deliberately outside the unit suite. Run them from the
repository root after installing the development dependencies:

```console
python benchmarks/baseline_suite.py
python benchmarks/bath_discretization.py
python benchmarks/mpo_tdvp.py
python benchmarks/tree_engine.py
```

`baseline_suite.py` checks a reference observable and the number of Krylov calls
strictly. Peak bond and Krylov iteration count use the narrow tolerances recorded
in `baseline_reference.json`: floating-point contraction order and the BLAS
implementation can shift threshold decisions without changing the dynamics. Wall
time is printed for local comparison but is not a CI threshold. Update the
reference or its tolerances only for a justified numerical or algorithmic change,
and describe the reason in the commit.
