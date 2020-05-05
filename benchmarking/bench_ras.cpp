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

#include <bench_base.hpp>


template <typename ValueType, typename IndexType>
class BenchRas : public BenchBase<ValueType, IndexType> {
public:
    void run();

private:
    void solve(MPI_Comm mpi_communicator);
};

template <typename ValueType, typename IndexType>
void BenchRas<ValueType, IndexType>::solve(MPI_Comm mpi_communicator)
{
    schwz::Metadata<ValueType, IndexType> metadata;
    schwz::Settings settings(FLAGS_executor);

    // Set solver metadata from command line args.
    metadata.mpi_communicator = mpi_communicator;
    MPI_Comm_rank(metadata.mpi_communicator, &metadata.my_rank);
    MPI_Comm_size(metadata.mpi_communicator, &metadata.comm_size);
    metadata.tolerance = FLAGS_set_tol;
    metadata.max_iters = FLAGS_num_iters;
    metadata.num_subdomains = metadata.comm_size;
    metadata.num_threads = FLAGS_num_threads;
    metadata.oned_laplacian_size = FLAGS_set_1d_laplacian_size;

    // Generic settings
    settings.write_debug_out = FLAGS_enable_debug_write;
    settings.write_perm_data = FLAGS_write_perm_data;
    settings.write_iters_and_residuals = FLAGS_write_iters_and_residuals;
    settings.print_matrices = FLAGS_print_matrices;
    settings.shifted_iter = FLAGS_shifted_iter;

    // Set solver settings from command line args.
    // Comm settings
    settings.comm_settings.enable_onesided = FLAGS_enable_onesided;
    if (FLAGS_remote_comm_type == "put") {
        settings.comm_settings.enable_put = true;
        settings.comm_settings.enable_get = false;
    } else if (FLAGS_remote_comm_type == "get") {
        settings.comm_settings.enable_put = false;
        settings.comm_settings.enable_get = true;
    }
    settings.comm_settings.enable_one_by_one = FLAGS_enable_one_by_one;
    settings.comm_settings.enable_overlap = FLAGS_enable_comm_overlap;
    if (FLAGS_flush_type == "flush-all") {
        settings.comm_settings.enable_flush_all = true;
    } else if (FLAGS_flush_type == "flush-local") {
        settings.comm_settings.enable_flush_all = false;
        settings.comm_settings.enable_flush_local = true;
    }
    if (FLAGS_lock_type == "lock-all") {
        settings.comm_settings.enable_lock_all = true;
    } else if (FLAGS_lock_type == "lock-local") {
        settings.comm_settings.enable_lock_all = false;
        settings.comm_settings.enable_lock_local = true;
    }

    // Convergence settings
    settings.convergence_settings.put_all_local_residual_norms =
        FLAGS_enable_put_all_local_residual_norms;
    settings.convergence_settings.enable_global_check_iter_offset =
        FLAGS_enable_global_check_iter_offset;
    settings.convergence_settings.enable_global_check =
        FLAGS_enable_global_check;
    if (FLAGS_global_convergence_type == "centralized-tree") {
        settings.convergence_settings.enable_global_simple_tree = true;
    } else if (FLAGS_global_convergence_type == "decentralized") {
        settings.convergence_settings.enable_decentralized_leader_election =
            true;
        settings.convergence_settings.enable_accumulate =
            FLAGS_enable_decentralized_accumulate;
    }

    // General solver settings
    metadata.local_solver_tolerance = FLAGS_local_tol;
    metadata.local_precond = FLAGS_local_precond;
    metadata.local_max_iters = FLAGS_local_max_iters;
    settings.non_symmetric_matrix = FLAGS_non_symmetric_matrix;
    settings.restart_iter = FLAGS_restart_iter;
    metadata.precond_max_block_size = FLAGS_precond_max_block_size;
    metadata.precond_max_block_size = FLAGS_precond_max_block_size;
    settings.matrix_filename = FLAGS_matrix_filename;
    settings.explicit_laplacian = FLAGS_explicit_laplacian;
    settings.rhs_type = FLAGS_rhs_type;
    settings.overlap = FLAGS_overlap;
    settings.naturally_ordered_factor = FLAGS_factor_ordering_natural;
    settings.reorder = FLAGS_local_reordering;
    settings.factorization = FLAGS_local_factorization;
    if (FLAGS_partition == "metis") {
        settings.partition =
            schwz::Settings::partition_settings::partition_metis;
        settings.metis_objtype = FLAGS_metis_objtype;
    } else if (FLAGS_partition == "regular") {
        settings.partition =
            schwz::Settings::partition_settings::partition_regular;
    } else if (FLAGS_partition == "regular2d") {
        settings.partition =
            schwz::Settings::partition_settings::partition_regular2d;
    }
    if (FLAGS_local_solver == "iterative-ginkgo") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::iterative_solver_ginkgo;
    } else if (FLAGS_local_solver == "direct-cholmod") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::direct_solver_cholmod;
    } else if (FLAGS_local_solver == "direct-umfpack") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::direct_solver_umfpack;
    } else if (FLAGS_local_solver == "direct-ginkgo") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::direct_solver_ginkgo;
    }
    settings.debug_print = FLAGS_debug;

    // Event settings
    metadata.constant = FLAGS_constant;
    metadata.gamma = FLAGS_gamma;

    // The global solution vector to be passed in to the RAS solver.
    std::shared_ptr<gko::matrix::Dense<ValueType>> explicit_laplacian_solution;

    if (metadata.my_rank == 0) {
        std::cout << " Running on the " << FLAGS_executor << " executor on "
                  << metadata.num_subdomains << " ranks with "
                  << FLAGS_num_threads << " threads" << std::endl;
        std::cout << " Problem Size: " << metadata.global_size << std::endl;
    }
    if (FLAGS_print_config) {
        if (metadata.my_rank == 0) {
            this->print_config();
        }
    }

    schwz::SolverRAS<ValueType, IndexType> solver(settings, metadata);
    solver.initialize();
    solver.run(explicit_laplacian_solution);
    if (FLAGS_timings_file != "null") {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename = FLAGS_timings_file + "_" + rank_string + ".csv";
        this->write_timings(metadata.time_struct, filename,
                            settings.comm_settings.enable_onesided);
    }
    if (FLAGS_write_comm_data) {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename_send = "num_send_" + rank_string + ".csv";
        std::string filename_recv = "num_recv_" + rank_string + ".csv";
        this->write_comm_data(metadata.num_subdomains, metadata.my_rank,
                              metadata.comm_data_struct, filename_send,
                              filename_recv);
    }

    if (FLAGS_debug) {
        // Print final solution to file
        std::string fname("final_sol.csv");
        std::ofstream fp;
        if (metadata.my_rank == 0) {
            fp.open(fname);
            auto num_elem = explicit_laplacian_solution->get_size()[0];
            auto dim = (int)sqrt(num_elem);
            // iterating over
            for (auto i = 0; i < num_elem; i++) {
                if (i % dim == 0 && i != 0) fp << std::endl;
                fp << explicit_laplacian_solution->get_values()[i] << ",";
            }
            fp.close();
        }
    }
}


template <typename ValueType, typename IndexType>
void BenchRas<ValueType, IndexType>::run()
{
    solve(MPI_COMM_WORLD);
}


int main(int argc, char *argv[])
{
    try {
        initialize_argument_parsing(&argc, &argv);
        BenchRas<double, int> laplace_problem_2d;

        if (FLAGS_num_threads > 1) {
            int req_thread_support = MPI_THREAD_MULTIPLE;
            int prov_thread_support = MPI_THREAD_MULTIPLE;

            MPI_Init_thread(&argc, &argv, req_thread_support,
                            &prov_thread_support);
            if (prov_thread_support != req_thread_support) {
                std::cout << "Required thread support is " << req_thread_support
                          << " but provided thread support is only "
                          << prov_thread_support << std::endl;
            }
        } else {
            MPI_Init(&argc, &argv);
        }
        laplace_problem_2d.run();
        MPI_Finalize();
    } catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    } catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
