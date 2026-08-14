# Benchmarks and regression baselines

The benchmark scripts are deliberately outside the unit suite. Run them from the
repository root after installing the development dependencies:

```console
python benchmarks/baseline_suite.py
python benchmarks/bath_discretization.py
python benchmarks/mpo_tdvp.py
python benchmarks/tree_engine.py
```

`baseline_suite.py` checks machine-independent work metrics (peak bond and Krylov
call/iteration counts) plus a reference observable. Wall time is printed for local
comparison but is not used as a CI threshold because it is hardware and BLAS
dependent. Update `baseline_reference.json` only for an intentional numerical or
algorithmic change and describe the reason in the commit.

For an exact before/after trajectory check across every registry method, use:

```console
python tests/characterization/all_methods_golden.py capture before.pkl
# make the change
python tests/characterization/all_methods_golden.py check before.pkl
```

Those temporary pickle files are environment-specific and should not be committed.
