Benchmarking.                            {#benchmarking_schwarz}
-------------

The flag `-DSCHWARZ_BUILD_BENCHMARKING` (default `ON`) enables the examples and benchmarking snippets. 

If `schwarz-lib` has been built with `deal.ii`, then the `deal.ii` examples, `ex_6` and `ex_9` are also built, else only the `bench_ras` example is built. 
The following command line options are available for this example. This is setup using `gflags`.

The executable is run in the following fashion:

```sh
[MPI_COMMAND] [MPI_OPTIONS] PATH_TO_EXECUTABLE [FLAGS]
```

Where `FLAGS` are the options below with the template `flag_name [type][default_value]`. For example, to set the number of iterations of the RAS solver to 100 one would add `--num_iters=100` to the executable command above.

### Generic settings
* `executor` [std::string][reference] : The executor used to run the solver, one of `reference`, `cuda` or `omp`.
* `explicit_laplacian` [bool][false] : Use the explicit laplacian instead of deal.ii's matrix.
* `set_1d_laplacian_size`[uint32][16] : The number of grid points in one dimension for the 2D laplacian problem.
* `enable_random_rhs` [bool][false] : Use a random rhs instead of the default 1.0's .
* `overlap` [uint32][2] : Overlap between the domains.
* `timings_file` [std::string][null] : The filename for the timings.
* `partition` [std::string][regular] : The partitioner used. The choices are `metis`, `regular` or `regular2d`.
* `metis_objtype` [std::string][null] : The objective type to minimize for the metis partitioner. The choices are `edgecut` and `totalvol`.
* `num_threads` [uint32][1] : Number of threads to bind to a process.
* `non_symmetric_matrix` [bool][false] : Explicitly state that the matrix is non-symmetric so that the local GMRES solver is used.
* `use_mixed_precision` [bool][false] : Use mixed precision in the communication.

### Input settings
* `matrix_filename` [std::string][null] : The matrix file to read the global system matrix from.

### Output settings
* `enable_debug_write` [bool][false] : Enable some debugging outputs to stdout.
* `write_comm_data` [bool][false] : Write the number of sends and recvs of each subdomain to files.
* `write_perm_data` [bool][false] : Write the permutation data from CHOLMOD to a file.
* `print_config` [bool][true] : Print the configuration of the run.
* `print_matrices` [bool][false] : Print the local system matrices to a file.
* `debug` [bool][false] : Enable some possible expensive debug checks.
* `enable_logging` [bool][false] : Enable some possible expensive logging from Ginkgo.

### Solver settings
#### Generic settings
* `num_iters` [uint32][100] : The number of outer iterations for the RAS solver.
* `set_tol` [double][1e-6] : The Outer tolerance for the RAS solver.
* `local_tol` [double][1e-12] : The Inner tolerance for the local iterative solver.
#### Communication settings
* `enable_onesided` [bool][false] : Enable the onesided asynchronous communication.
* `enable_twosided` [bool][true] : Enable the twosided asynchronous communication. A dummy flag.
* `stage_through_host` [bool][false] : Enable staging transfers through host.
* `enable_one_by_one` [bool][false] : Enable putting/getting of each element in onesided communication.
* `enable_put_all_local_residual_norms`  [bool][false] : Enable putting of all local residual norms"
* `enable_comm_overlap` [bool][false] : Enable overlap of communication and computation.
* `flush_type` [std::string][flush-all] : The window flush strategy. The choices are `flush-local` and `flush-all`.
* `lock_type` [std::string][lock-all] : The window lock strategy. The choices are `lock-local` and `lock-all`.
* `remote_comm_type` [std::string][get] : The type of the remote communication. `get` uses `MPI_Get` and `put` uses `MPI_Put`.

#### Convergence settings
* `enable_global_check` [bool][false] : Use the global convergence check for twosided.
* `global_convergence_type` [std::string][centralized-tree] : Choose the convergence detection algorithm for onesided.
* `enable_decentralized_accumulate` [bool][false] : Use accumulate strategy for decentralized convergence check..
* `enable_global_check_iter_offset`  [bool][false] : Enable global convergence check only after a certain number of iterations.

#### Local solver settings 
* `local_solver` [std::string][iterative-ginkgo] : The local solver used in the local domains. The current choices are `direct-cholmod` , `direct-ginkgo` or `iterative-ginkgo`.
* `local_factorization` [std::string][cholmod] : The factorization for the local direct solver "cholmod" or "umfpack".
* `local_reordering` [std::string][none] : The reordering for the local direct solver "none", "metis_reordering" or "rcm_reordering".
* `factor_ordering_natural` [bool][false] : If true uses natural ordering instead of the default optimized ordering. This is needed for CUDA runs as the factorization ordering needs to be given to the solver.
* `enable_local_precond` [bool][false] : If true uses the Block jacobi preconditioning for the local iterative solver. 
* `precond_max_block_size` [uint32][16]:  Maximum size of the blocks for the block jacobi preconditioner
* `shifted_iter` [uint32][1] : The number of iterations to communicate for the local subdomains.
* `local_max_iters` [int32][-1] : The maximum number of iterations for the local iterative solver.
* `restart_iter` [uint32][1] : The restart iter for the GMRES solver.
* `reset_local_crit_iter` [int32][-1] : The RAS iteration to reset the local iteration count.


### Poisson solver using Restricted Additive Schwarz with overlap.

This example runs is written within the `benchmarking/bench_ras.cpp` file. This demonstrates the basic capabilities of `schwarz-lib`. You can use it to solve the 2D Poisson equation with a 5 point stencil or solve a generic matrix by providing it a matrix file.

### Examples with deal.ii
These examples use `deal.ii`'s capabilities to generate a matrix and solution is computed with the RAS method. 

Possible settings are: 

* `num_refine_cycles` [uint32][1][disabled] : The number of refinement cycles when used with `deal.ii`.
* `init_refine_level` [uint32][4] : The initial refinement level of the problem. This sets the initial number of dof's.
* `dealii_orig` [bool][false] : Solve with the deal.ii iterative CG instead of the RAS solver. 
* `vis_sol` [bool][false] : Print the solution for visualization. 

#### Solving the n-dimensional Poisson equation with FEM.
The `benchmarking/dealii_ex_6.cpp` demonstrates the solution of the Poisson equation with adaptive refinement as explained on the [`deal.ii example documentation page`](https://www.dealii.org/developer/doxygen/deal.II/step_6.html)  


#### Solving the Advection equation with FEM.
The `benchmarking/dealii_ex_9.cpp` demonstrates the solution of the Advection equation with adaptive refinement as explained on the [`deal.ii example documentation page`](https://www.dealii.org/developer/doxygen/deal.II/step_9.html)  
