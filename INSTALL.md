# Installation process.

## Building 

Use the standard cmake build procedure:

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" [OPTIONS] .. && make
```

Replace `[OPTIONS]` with desired cmake options for your build.
The library adds the following additional switches to control what is being built:

*   `-DSCHWARZ_BUILD_BENCHMARKING={ON, OFF}` Builds some example benchmarks.
    Default is `ON`
*   `-DSCHWARZ_BUILD_METIS={ON, OFF}` Builds with support for the `METIS` partitioner. User needs
    to provide the path to the installation of the `METIS` library in `METIS_DIR`, preferably as 
    an environment variable. Default is `OFF`
*   `-DSCHWARZ_BUILD_CHOLMOD={ON, OFF}` Builds with support for the `CHOLMOD` module from the 
    Suitesparse library. User needs to set an environment variable `CHOLMOD_DIR` to the 
    path containing the `CHOLMOD` installation. Default is `OFF`
*   `-DSCHWARZ_BUILD_CUDA={ON, OFF}` Builds with CUDA support. Though Ginkgo provides most of the 
    required CUDA support, we do need to link to CUDA for explicit setting of GPU affinities, 
    some custom gather and scatter operations. Default is `OFF`.
*   `-DSCHWARZ_BUILD_CLANG_TIDY={ON, OFF}` Builds with support for `clang-tidy`
    Default is `OFF`
*   `-DSCHWARZ_BUILD_DEALII={ON, OFF}` Builds with support for the finite element library `deal.ii`
    Default is `OFF`
*   `-DSCHWARZ_WITH_HWLOC={ON, OFF}` Builds with support for the hardware locality library used for binding hardware.
    `hwloc` is distributed as a part of the Open-MPI project. Default is `ON`
*   `-DSCHWARZ_DEVEL_TOOLS={ON, OFF}` Builds with some developer tools support. Default is `ON`. In particular uses [`git-cmake-format`](https://github.com/ginkgo-project/git-cmake-format) to automatically format the source files with `clang-format`. 


## Tips

* If you are having CUDA problems and you are not using CUDA, then feel free to switch the CUDA module off with `-DSCHWARZ_BUILD_CUDA=off`. 

* Installing CHOLMOD can be a bit annoying. TODO add some details on fixing Suitesparse compilation.

* When doing merge commits it is possible that make format does not work. You can run `cmake -DSCHWARZ_DEVEL_TOOLS=OFF ..` to temporarily switch off the formatting. Please switch it on again when committing normally.
