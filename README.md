### Schwarz testbed

## Repository for testing schwarz methods.
--------------------------------------

Required components
-------------------

The required components include:
1. Ginkgo: The Ginkgo library is needed. It needs to be installed and preferably the installation path
   provided as an environment variable in `Ginkgo_DIR` variable.
2. MPI: As multiple nodes and a domain decomposition is used, an MPI implementation is necessary.
3. Boost: A Boost library is also required as one of its header files `mpi_datatype.hpp` is used to 
   detect the `MPI_type` needed at run-time. It may also be possible to get this specific file provide 
   its include path (NOT TESTED) if the complete Boost library is not easily available.


Quick Install
------------

### Building Schwarz-Lib 

To build Schwarz-Lib, you can use the standard CMake procedure. 

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" .. && make
```

By default, `SCHWARZ_BUILD_BENCHMARKING` is enabled. This allows you to quickly run an example with the timings if needed. For a detailed list of options available see the [Benchmarking page](./benchmarking/BENCHMARKING.md).

For more CMake options please refer to the [Installation page](./INSTALL.md)



Currently implemented features
-------------------------------

1. Executor paradigm:
+ [x] GPU.
+ [x] OpenMP.
+ [ ] Single rank per node and threading in one node.

2. Solving paradigm:
  * Direct:
  + [ ] Ginkgo (To Fix).
  + [x] CHOLMOD.
  * Iterative:
  + [x] Ginkgo.
  + [ ] deal.ii.

3. Partitioning paradigm:
+ [x] METIS.
+ [x] Naive, 1D.
+ [ ] Zoltan.


4. Convergence check:
+ [x] Tree convergence.
+ [ ] Bahi decentralized.

5. Communication paradigm.
+ [x] Onesided.
+ [x] Twosided.
+ [ ] Event based.

6. Schwarz problem type.
+ [x] RAS.
+ [ ] O-RAS.


Any of the implemented features can be permuted and tested.


For installing and building, please check the [Installation page](./INSTALL.md)
