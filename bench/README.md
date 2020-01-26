# Benchmarks


## Transformation

### Affine

Results on Notebook, min of 5 runs, in seconds.

- 2d benchmark
	- elf (python)
		- order 0: 4.3399
		- order 1: 11.2030
	- elf (C++)
		- order 0: 0.0030
		- order 1: 0.0044
	- scipy
		- order 0: 0.0056
		- order 1: 0.0179
- 3d benchmark
	- elf (python)
		- order 0: 84.1674
		- order 1: 91.3547
	- elf (C++)
		- order 0: 0.0778
		- order 1: 0.0797
	- scipy
		- order 0: 0.1734
		- order 1: 0.1673


## Label Multisets

You can download the benchmark data from [here](https://drive.google.com/file/d/1E_Wpw9u8E4foYKk7wvx5RPSWvg_NCN7U/view?usp=sharing).

Results on gpu6, min of 5 runs.

Benchmarks for the label multi-set implementation.
- First Implementation (433b79d733128332428cdefe966b4f60b67d50de):
  - `bench_multist`: 3.16 s
  - `bench_multiset_grid`: 3.43 s
  - `bench_create`: 18.47 s
