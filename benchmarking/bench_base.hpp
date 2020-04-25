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

#ifndef bench_base_hpp
#define bench_base_hpp

#include <mpi.h>

#include <fstream>
#include <numeric>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include <communicate.hpp>
#include <initialization.hpp>
#include <restricted_schwarz.hpp>
#include <solve.hpp>

DEFINE_bool(debug, false, "Enable some possibly expensive debug checks");
DEFINE_bool(non_symmetric_matrix, false, "The matrix is non-symmetric");
DEFINE_uint32(num_iters, 100, "Number of Schwarz iterations");
DEFINE_double(set_tol, 1e-6, "Tolerance for the Schwarz solver");
DEFINE_double(local_tol, 1e-12, "Tolerance for the local solver");
DEFINE_uint32(set_1d_laplacian_size, 16,
              "Problem size for explicit laplacian problem without deal.ii");
DEFINE_uint32(shifted_iter, 1,
              "Use a shifted communication after every x iterations.");
DEFINE_bool(enable_debug_write, false, "Enable some debug writes.");
DEFINE_bool(enable_onesided, false,
            "Use the onesided communication version for the solver");
DEFINE_bool(enable_twosided, true,
            "Use the twosided communication version for the solver");
DEFINE_bool(enable_one_by_one, false,
            "Enable one element after another in onesided");
DEFINE_string(remote_comm_type, "get",
              " The remote memory function to use, MPI_Put / MPI_Get, options "
              "are put or get");
DEFINE_bool(enable_put_all_local_residual_norms, false,
            "Enable putting of all local residual norms");
DEFINE_bool(enable_comm_overlap, false,
            "Enable overlap of communication and computation");
DEFINE_bool(enable_global_check, false,
            "Use the global convergence check for twosided");
DEFINE_string(global_convergence_type, "centralized-tree",
              "The type of global convergence check strategy for onesided. "
              "Choices are centralized-tree or decentralized");
DEFINE_bool(
    enable_decentralized_accumulate, false,
    "Use accumulate strategy for decentralized global convergence check");
DEFINE_bool(enable_global_check_iter_offset, false,
            "Enable global convergence check only after a certain number of "
            "iterations");
DEFINE_string(matrix_filename, "null", "The matrix filename to read from.");
DEFINE_bool(explicit_laplacian, false,
            "Use the explicit laplacian generated by us");
DEFINE_bool(enable_random_rhs, false,
            "Use a random rhs instead of the default 1.0's ");
DEFINE_uint32(overlap, 2, "Overlap between the domains");
DEFINE_string(
    executor, "reference",
    "The executor used to run the solver, one of reference, cuda or omp");
DEFINE_string(flush_type, "flush-all",
              "The window flush. The choices are flush-local and flush-all");
DEFINE_string(lock_type, "lock-all",
              "The window locking. The choices are lock-local and lock-all");
DEFINE_string(timings_file, "null", "The filename for the timings");
DEFINE_bool(write_comm_data, false,
            "Write the number of elements sent and received by each subdomain "
            "to a file.");
DEFINE_bool(write_perm_data, false,
            "Write the permutation from CHOLMOD to a file");
DEFINE_bool(write_iters_and_residuals, false,
            "Write the iteration and residual log to a file");
DEFINE_bool(print_matrices, false,
            "Write the local system matrices for debugging");
DEFINE_bool(print_config, true, "Print the configuration of the run ");
DEFINE_string(
    partition, "regular",
    "The partitioner used. The choices are metis, regular, regular2d");
DEFINE_string(local_solver, "iterative-ginkgo",
              "The local solver used in the local domains. The current choices "
              "include direct-cholmod , direct-ginkgo or iterative-ginkgo");
DEFINE_string(local_reordering, "none",
              "The reordering for the local direct solver. Choices"
              "include none , rcm_reordering (symmetric matrices) or "
              "metis_reordering (all)");
DEFINE_string(local_factorization, "cholmod",
              "The factorization for the local direct solver. Choices"
              "include cholmod or umfpack");
DEFINE_uint32(num_threads, 1, "Number of threads to bind to a process");
DEFINE_bool(factor_ordering_natural, false,
            "If true uses natural ordering instead of the default optimized "
            "ordering. ");
DEFINE_int32(local_max_iters, -1,
             "Number of maximum iterations for the local iterative solver");
DEFINE_string(local_precond, "null",
              "Choices are ilu, isai and block-jacobi for the local "
              "iterative solver. ");
DEFINE_uint32(precond_max_block_size, 16,
              "Maximum size of the blocks for the block jacobi preconditioner");
DEFINE_string(metis_objtype, "null",
              "Defines the objective type for the metis partitioning, options "
              "are edgecut and totalvol ");


void initialize_argument_parsing(int *argc, char **argv[])
{
    std::ostringstream msg;
    msg << "Flags";
    gflags::SetUsageMessage(msg.str());
    gflags::ParseCommandLineFlags(argc, argv, true);
}

template <typename ValueType, typename IndexType>
class BenchBase {
public:
    BenchBase() = default;

    void write_timings(
        std::vector<std::tuple<int, int, int, std::string,
                               std::vector<ValueType>>> &time_struct,
        std::string filename, bool enable_onesided);

    void write_comm_data(
        int num_subd, int my_rank,
        std::vector<std::tuple<int, std::vector<std::tuple<int, int>>,
                               std::vector<std::tuple<int, int>>, int, int>>
            &comm_data_struct,
        std::string filename_send, std::string filename_recv);

    int get_local_rank(MPI_Comm mpi_communicator);

    void print_config();
};


template <typename ValueType, typename IndexType>
void BenchBase<ValueType, IndexType>::write_comm_data(
    int num_subd, int my_rank,
    std::vector<std::tuple<int, std::vector<std::tuple<int, int>>,
                           std::vector<std::tuple<int, int>>, int, int>>
        &comm_data_struct,
    std::string filename_send, std::string filename_recv)
{
    {
        std::ofstream file;
        file.open(filename_send);
        file << "subdomain " << my_rank << " has "
             << std::get<4>(comm_data_struct[my_rank]) << " neighbors\n";
        file << "my_id,to_id,num_send\n";
        for (auto i = 0; i < num_subd; ++i) {
            file << my_rank << ","
                 << std::get<0>(std::get<2>(comm_data_struct[my_rank])[i])
                 << ","
                 << std::get<1>(std::get<2>(comm_data_struct[my_rank])[i])
                 << "\n";
        }
        file.close();
    }
    {
        std::ofstream file;
        file.open(filename_recv);
        file << "subdomain " << my_rank << " has "
             << std::get<3>(comm_data_struct[my_rank]) << " neighbors\n";
        file << "my_id,from_id,num_recv\n";
        for (auto i = 0; i < num_subd; ++i) {
            file << my_rank << ","
                 << std::get<0>(std::get<1>(comm_data_struct[my_rank])[i])
                 << ","
                 << std::get<1>(std::get<1>(comm_data_struct[my_rank])[i])
                 << "\n";
        }
        file.close();
    }
}


template <typename ValueType, typename IndexType>
void BenchBase<ValueType, IndexType>::write_timings(
    std::vector<std::tuple<int, int, int, std::string, std::vector<ValueType>>>
        &time_struct,
    std::string filename, bool enable_onesided)
{
    std::ofstream file;
    file.open(filename);
    for (auto id = 0; id < time_struct.size(); id++)
        std::sort(std::get<4>(time_struct[id]).begin(),
                  std::get<4>(time_struct[id]).end());
    file << "func,total,avg,min,med,max\n";
    auto vec_size = time_struct.size() + 1;
    std::vector<std::string> func_name(vec_size);
    std::vector<ValueType> avg_time(vec_size), med_time(vec_size),
        min_time(vec_size), max_time(vec_size), total_time(vec_size);
    for (auto id = 0; id < time_struct.size(); id++) {
        func_name[id] = std::get<3>(time_struct[id]);
        total_time[id] =
            std::accumulate(std::get<4>(time_struct[id]).begin(),
                            std::get<4>(time_struct[id]).end(), 0.0);
        avg_time[id] = total_time[id] / (std::get<2>(time_struct[id]));
        min_time[id] = *std::min_element(std::get<4>(time_struct[id]).begin(),
                                         std::get<4>(time_struct[id]).end());
        med_time[id] = std::get<4>(
            time_struct[id])[std::get<4>(time_struct[id]).size() / 2];
        max_time[id] = *std::max_element(std::get<4>(time_struct[id]).begin(),
                                         std::get<4>(time_struct[id]).end());
    }
    func_name[time_struct.size()] = "other";
    if (enable_onesided) {
        total_time[time_struct.size()] = total_time[0] + total_time[2];
        total_time[0] = 0.0;
        total_time[2] = 0.0;
        avg_time[time_struct.size()] = avg_time[0] + avg_time[2];
        avg_time[0] = 0.0;
        avg_time[2] = 0.0;
        min_time[time_struct.size()] = min_time[0] + min_time[2];
        min_time[0] = 0.0;
        min_time[2] = 0.0;
        max_time[time_struct.size()] = max_time[0] + max_time[2];
        max_time[0] = 0.0;
        max_time[2] = 0.0;
        med_time[time_struct.size()] = med_time[0] + med_time[2];
        med_time[0] = 0.0;
        med_time[2] = 0.0;
    }

    for (auto i = 0; i < func_name.size(); ++i) {
        file << func_name[i] << "," << total_time[i] << "," << avg_time[i]
             << "," << min_time[i] << "," << med_time[i] << "," << max_time[i]
             << "\n";
    }
    file.close();
}


template <typename ValueType, typename IndexType>
int BenchBase<ValueType, IndexType>::get_local_rank(MPI_Comm mpi_communicator)
{
    MPI_Comm local_comm;
    int rank;
    MPI_Comm_split_type(mpi_communicator, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &rank);
    return rank;
}


template <typename ValueType, typename IndexType>
void BenchBase<ValueType, IndexType>::print_config()
{
    std::cout << " Executor: " << FLAGS_executor << "\n"
              << " Comm type: "
              << (FLAGS_enable_onesided ? "onesided" : "twosided") << "\n"
              << " Remote comm type: " << FLAGS_remote_comm_type << "\n"
              << " Element sending strategy:  "
              << (FLAGS_enable_one_by_one ? "one by one" : "gathered") << "\n"
              << std::endl;
}


#endif
