Schwarz Library
-------------------

[![Build status](https://github.com/pratikvn/schwarz-lib/workflows/Build/badge.svg)](https://github.com/pratikvn/schwarz-lib/actions?query=workflow%3ABuild-status)
[![Documentation](https://github.com/pratikvn/schwarz-lib/workflows/Build-doc/badge.svg?branch=develop)](https://pratikvn.github.io/schwarz-lib/doc/develop/index.html)

Performance results
-------------------
1. [Paper in IJHPCA](https://journals.sagepub.com/doi/10.1177/1094342020946814); [Alternative arXiv version](https://arxiv.org/abs/2003.05361)
2. [Two stage update](https://ieeexplore.ieee.org/document/9308758)


## Required components

The required components include:
1. Ginkgo: The Ginkgo library is needed. It needs to be installed and preferably the installation path
   provided as an environment variable in `Ginkgo_DIR` variable.
2. MPI: As multiple nodes and a domain decomposition is used, an MPI implementation is necessary.

## Quick Install

### Building Schwarz-Lib 

To build Schwarz-Lib, you can use the standard CMake procedure. 

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" .. && make
```

By default, `SCHWARZ_BUILD_BENCHMARKING` is enabled. This allows you to quickly run an example with the timings if needed. For a detailed list of options available see the [Benchmarking page](./benchmarking/BENCHMARKING.md).

For more CMake options please refer to the [Installation page](./INSTALL.md)



## Currently implemented features

1. Executor paradigm:
+ [x] GPU.
+ [x] OpenMP.
+ [ ] Single rank per node and threading in one node.

1. Factorization paradigm:
  + [x] CHOLMOD.
  + [x] UMFPACK.

2. Solving paradigm:
  * Direct:
  + [x] Ginkgo.
  + [x] CHOLMOD.
  + [x] UMFPACK.
  * Iterative:
  + [x] Ginkgo.
  + [ ] deal.ii.

3. Partitioning paradigm:
+ [x] METIS.
+ [x] Regular, 1D.
+ [x] Regular, 2D.
+ [ ] Zoltan.


4. Convergence check:
+ [x] Centralized, tree based convergence (Yamazaki 2019).
+ [x] Decentralized, leader election based (Bahi 2005).

5. Communication paradigm.
+ [x] Onesided.
+ [x] Twosided.
+ [ ] Event based.

5. Communication strategies.
+ [x] Remote comm strategies: 
    + [x] MPI_Put , gathered.
    + [x] MPI_Put , one by one.
    + [x] MPI_Get , gathered .
    + [x] MPI_Get , one by one.
+ [x] Lock strategies: MPI_Win_lock / MPI_Win_lock_all .
    + [x] Lock all and unlock all.
    + [x] Lock local and unlock local.
+ [x] Flush strategies: MPI_Win_flush / MPI_Win_flush_local .
    + [x] Flush all.
    + [x] Flush local.

6. Schwarz problem type.
+ [x] RAS.
+ [ ] O-RAS.


Any of the implemented features can be permuted and tested.

## Known Issues

1. On Summit, the Spectrum MPI seems to have a bug with using `MPI_Put` with GPU buffers. `MPI_Get` works as expected. This bug has also been confirmed with an external micro-benchmarking library, [OSU Micro-Benchmarks](https://github.com/pratikvn/osu-bench-personal-fork).


For installing and building, please check the [Installation page](./INSTALL.md)


Credits: This code (written in C++, with additions and improvements) was inspired by the code from Ichitaro Yamazaki, ICL, UTK.
