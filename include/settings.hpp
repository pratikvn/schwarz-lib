
/*******************************<SCHWARZ LIB LICENSE>***********************
Copyright (c) 2019, the SCHWARZ LIB authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<SCHWARZ LIB LICENSE>*************************/


#ifndef settings_hpp
#define settings_hpp


#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>


#include <mpi.h>
#include <boost/mpi/datatype.hpp>
#include <ginkgo/ginkgo.hpp>


#include <device_guard.hpp>
#include <exception_helpers.hpp>
#include <gather_scatter.hpp>


#if SCHW_HAVE_METIS
#include <metis.h>
#define metis_indextype idx_t
#else
#define metis_indextype gko::int32
#endif


#define MINIMAL_OVERLAP 2


namespace schwz {


/**
 * @brief The struct that contains the solver settings and the parameters to be
 * set by the user.
 *
 * @ref settings
 * @ingroup init
 */
struct Settings {
    /**
     * The string that contains the ginkgo executor paradigm.
     */
    std::string executor_string;

    /**
     * The ginkgo executor the code is to be executed on.
     */
    std::shared_ptr<gko::Executor> executor = gko::ReferenceExecutor::create();

    /**
     * The ginkgo executor the code is to be executed on.
     */
    std::shared_ptr<device_guard> cuda_device_guard;

    /**
     * The partition algorithm to be used for partitioning the matrix.
     */
    enum partition_settings {
        partition_regular = 0x0,
        partition_regular2d = 0x4,
        partition_metis = 0x1,
        partition_zoltan = 0x2,
        partition_custom = 0x3
    };
    partition_settings partition = partition_settings::partition_regular;

    /**
     * The overlap between the subdomains.
     */
    gko::int32 overlap = MINIMAL_OVERLAP;

    /**
     * The string that contains the matrix file name to read from .
     */
    std::string matrix_filename = "null";

    /**
     * Flag if the laplacian matrix should be generated within the library. If
     * false, an external matrix and rhs needs to be provided
     */
    bool explicit_laplacian = true;

    /**
     * Flag to enable a random rhs.
     */
    bool enable_random_rhs = false;

    /**
     * Flag to enable printing of matrices.
     */
    bool print_matrices = false;

    /**
     * Flag to enable some debug printing.
     */
    bool debug_print = false;

    /**
     * The local solver algorithm for the local subdomain solves.
     */
    enum local_solver_settings {
        direct_solver_cholmod = 0x0,
        direct_solver_umfpack = 0x5,
        direct_solver_ginkgo = 0x1,
        iterative_solver_ginkgo = 0x2,
        iterative_solver_dealii = 0x3,
        solver_custom = 0x4
    };
    local_solver_settings local_solver =
        local_solver_settings::iterative_solver_ginkgo;

    bool non_symmetric_matrix = false;

    /**
     * Disables the re-ordering of the matrix before computing the triangular
     * factors during the CHOLMOD factorization
     *
     * @note This is mainly to allow compatibility with GPU solution.
     */
    bool naturally_ordered_factor = false;

    /**
     * This setting defines the objective type for the metis partitioning.
     */
    std::string metis_objtype;

    /**
     * Enable the block jacobi local preconditioner for the local solver.
     */
    bool use_precond = false;

    /**
     * Enable the writing of debug out to file.
     */
    bool write_debug_out = false;

    /**
     * Enable the local permutations from CHOLMOD to a file.
     */
    bool write_perm_data = false;

    /**
     * Iteration shift for node local communication.
     */
    int shifted_iter = 1;

    /**
     * The settings for the various available communication paradigms.
     */
    struct comm_settings {
        /**
         * Enable one-sided communication
         */
        bool enable_onesided = false;

        /**
         * Enable explicit overlap between communication and computation.
         */
        bool enable_overlap = false;

        /**
         * Put the data to the window using MPI_Put rather than get.
         */
        bool enable_put = false;

        /**
         * Get the data to the window using MPI_Get rather than put.
         */
        bool enable_get = true;

        /**
         * Push each element separately directly into the buffer.
         */
        bool enable_one_by_one = false;

        /**
         * Use local flush.
         */
        bool enable_flush_local = false;

        /**
         * Use flush all.
         */
        bool enable_flush_all = true;

        /**
         * Use local locks.
         */
        bool enable_lock_local = false;

        /**
         * Use lock all.
         */
        bool enable_lock_all = true;
    };
    comm_settings comm_settings;

    /**
     * The various convergence settings available.
     */
    struct convergence_settings {
        bool put_all_local_residual_norms = true;
        bool enable_global_simple_tree = false;
        bool enable_decentralized_leader_election = false;
        bool enable_global_check = true;
        bool enable_accumulate = false;

        bool enable_global_check_iter_offset = false;

        enum local_convergence_crit {
            residual_based = 0x0,
            solution_based = 0x1
        };

        local_convergence_crit convergence_crit =
            local_convergence_crit::solution_based;
    };
    convergence_settings convergence_settings;

    /**
     * The factorization for the local direct solver.
     */
    std::string factorization = "cholmod";

    /**
     * The reordering for the local solve.
     */
    std::string reorder;

    Settings(std::string executor_string = "reference")
        : executor_string(executor_string)
    {}
};


/**
 * The solver metadata struct.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ingroup init
 * @ingroup comm
 * @ingroup solve
 */
template <typename ValueType, typename IndexType>
struct Metadata {
    /**
     * The MPI communicator
     */
    MPI_Comm mpi_communicator;

    /**
     * The size of the global matrix.
     */
    gko::size_type global_size = 0;

    /**
     * The size of the 1 dimensional laplacian grid.
     */
    gko::size_type oned_laplacian_size = 0;

    /**
     * The size of the local subdomain matrix.
     */
    gko::size_type local_size = 0;

    /**
     * The size of the local subdomain matrix + the overlap.
     */
    gko::size_type local_size_x = 0;

    /**
     * The size of the local subdomain matrix + the overlap.
     */
    gko::size_type local_size_o = 0;

    /**
     * The size of the overlap between the subdomains.
     */
    gko::size_type overlap_size = 0;

    /**
     * The number of subdomains used within the solver.
     */
    gko::size_type num_subdomains = 1;

    /**
     * The rank of the subdomain.
     */
    int my_rank;

    /**
     * The local rank of the subdomain.
     */
    int my_local_rank;

    /**
     * The local number of procs in the subdomain.
     */
    int local_num_procs;

    /**
     * The number of subdomains used within the solver, size of the
     * communicator.
     */
    int comm_size;

    /**
     * The number of threads used within the solver for each subdomain.
     */
    int num_threads;

    /**
     * The iteration count of the solver.
     */
    IndexType iter_count;

    /**
     * The tolerance of the complete solver. The residual norm reduction
     * required.
     */
    ValueType tolerance;

    /**
     * The tolerance of the local solver in case of an iterative solve. The
     * residual norm reduction required.
     */
    ValueType local_solver_tolerance;

    /**
     * The maximum iteration count of the solver.
     */
    IndexType max_iters;

    /**
     * The maximum block size for the preconditioner.
     */
    unsigned int precond_max_block_size;

    /**
     * The current residual norm of the subdomain.
     */
    ValueType current_residual_norm = -1.0;

    /**
     * The minimum residual norm of the subdomain.
     */
    ValueType min_residual_norm = -1.0;

    /**
     * The struct used to measure the timings of each function within the solver
     * loop.
     */
    std::vector<std::tuple<int, int, int, std::string, std::vector<ValueType>>>
        time_struct;

    /**
     * The struct used to measure the timings of each function within the solver
     * loop.
     */
    std::vector<std::tuple<int, std::vector<std::tuple<int, int>>,
                           std::vector<std::tuple<int, int>>, int, int>>
        comm_data_struct;

    /**
     * The mapping containing the global to local indices.
     */
    std::shared_ptr<gko::Array<IndexType>> global_to_local;

    /**
     * The mapping containing the local to global indices.
     */
    std::shared_ptr<gko::Array<IndexType>> local_to_global;

    /**
     * The overlap row indices.
     */
    std::shared_ptr<gko::Array<IndexType>> overlap_row;

    /**
     * The starting row of each subdomain in the matrix.
     */
    std::shared_ptr<gko::Array<IndexType>> first_row;

    /**
     * The permutation used for the re-ordering.
     */
    std::shared_ptr<gko::Array<IndexType>> permutation;

    /**
     * The inverse permutation used for the re-ordering.
     */
    std::shared_ptr<gko::Array<IndexType>> i_permutation;
};


/**
 * This macro helps to measure the time of each of the functions.
 *
 * @param _func  The function call to be measured.
 * @param _id  A unique id for the function.
 * @parama _rank  The process calling the function.
 * @param _name  The name of the function as it should appear when printed.
 * @param _iter  The current iteration number.
 */
#define MEASURE_ELAPSED_FUNC_TIME(_func, _id, _rank, _name, _iter)     \
    {                                                                  \
        auto start_time = std::chrono::steady_clock::now();            \
        _func;                                                         \
        auto elapsed_time = std::chrono::duration<ValueType>(          \
            std::chrono::steady_clock::now() - start_time);            \
        if (_iter == 0) {                                              \
            std::vector<ValueType> temp_vec(1, elapsed_time.count());  \
            metadata.time_struct.push_back(                            \
                std::make_tuple(_id, _rank, _iter, #_name, temp_vec)); \
        } else {                                                       \
            std::get<2>(metadata.time_struct[_id]) = _iter;            \
            (std::get<4>(metadata.time_struct[_id]))                   \
                .push_back(elapsed_time.count());                      \
        }                                                              \
    }


#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, gko::int32);                   \
    template _macro(double, gko::int32);                  \
    template _macro(float, gko::int64);                   \
    template _macro(double, gko::int64);

// explicit instantiations for schwz
#define DECLARE_METADATA(ValueType, IndexType) \
    struct Metadata<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_METADATA);
#undef DECLARE_METADATA


}  // namespace schwz


#endif  // settings.hpp
