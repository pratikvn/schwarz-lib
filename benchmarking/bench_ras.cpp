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


#include <fstream>
#include <numeric>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include <communicate.hpp>
#include <initialization.hpp>
#include <schwarz_solver.hpp>
#include <solve.hpp>


DEFINE_uint32(num_iters, 100, "Number of Schwarz iterations");
DEFINE_double(set_tol, 1e-6, "Tolerance for the Schwarz solver");
DEFINE_double(local_tol, 1e-12, "Tolerance for the local solver");
DEFINE_uint32(set_1d_laplacian_size, 16,
              "Problem size for explicit laplacian problem without deal.ii");
DEFINE_uint32(
    num_refine_cycles, 1,
    "Number of refinement cycles for the adaptive refinement within deal.ii");
DEFINE_uint32(shifted_iter, 1,
              "Use a shifter communication after every x iterations.");
DEFINE_bool(enable_debug_write, false, "Enable some debug writes.");
DEFINE_bool(enable_onesided, false,
            "Use the onesided communication version for the solver");
DEFINE_bool(enable_twosided, true,
            "Use the twosided communication version for the solver");
DEFINE_bool(enable_push_one_by_one, false,
            "Enable push one element after another in onesided");
DEFINE_bool(enable_get, false, "Enable MPI_Get instead of the MPI_Put");
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
DEFINE_bool(explicit_laplacian, false,
            "Use the explicit laplacian instead of deal.ii's matrix");
DEFINE_bool(enable_random_rhs, false,
            "Use a random rhs instead of the default 1.0's ");
DEFINE_uint32(overlap, 2, "Overlap between the domains");
DEFINE_string(
    executor, "reference",
    "The executor used to run the solver, one of reference, cuda or omp");
DEFINE_string(enable_flush, "flush-all",
              "The window flush. The choices are flush-local and flush-all");
DEFINE_string(timings_file, "null", "The filename for the timings");
DEFINE_bool(write_comm_data, false,
            "Write the number of elements sent and received by each subdomain "
            "to a file.");
DEFINE_string(
    partition, "regular",
    "The partitioner used. The choices are metis, regular, regular2d");
DEFINE_string(local_solver, "direct-cholmod",
              "The local solver used in the local domains. The current choices "
              "include direct-cholmod , direct-ginkgo or iterative-ginkgo");
DEFINE_uint32(num_threads, 1, "Number of threads to bind to a process");
DEFINE_bool(factor_ordering_natural, false,
            "If true uses natural ordering instead of the default optimized "
            "ordering. ");
DEFINE_bool(enable_local_precond, false,
            "If true uses the Block jacobi preconditioning for the local "
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
class BenchRas {
public:
    void run();

private:
    void solve(MPI_Comm mpi_communicator);
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
};


template <typename ValueType, typename IndexType>
void BenchRas<ValueType, IndexType>::write_comm_data(
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
void BenchRas<ValueType, IndexType>::write_timings(
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
int BenchRas<ValueType, IndexType>::get_local_rank(MPI_Comm mpi_communicator)
{
    MPI_Comm local_comm;
    int rank;
    MPI_Comm_split_type(mpi_communicator, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &rank);
    return rank;
}


template <typename ValueType, typename IndexType>
void BenchRas<ValueType, IndexType>::solve(MPI_Comm mpi_communicator)
{
    SchwarzWrappers::Metadata<ValueType, IndexType> metadata;
    SchwarzWrappers::Settings settings(FLAGS_executor);

    // Set solver metadata from command line args.
    metadata.mpi_communicator = mpi_communicator;
    MPI_Comm_rank(metadata.mpi_communicator, &metadata.my_rank);
    MPI_Comm_size(metadata.mpi_communicator, &metadata.comm_size);
    metadata.tolerance = FLAGS_set_tol;
    metadata.max_iters = FLAGS_num_iters;
    metadata.num_subdomains = metadata.comm_size;
    metadata.num_threads = FLAGS_num_threads;
    metadata.oned_laplacian_size = FLAGS_set_1d_laplacian_size;
    metadata.global_size =
        metadata.oned_laplacian_size * metadata.oned_laplacian_size;

    // Generic settings
    settings.write_debug_out = FLAGS_enable_debug_write;
    settings.shifted_iter = FLAGS_shifted_iter;

    // Set solver settings from command line args.
    // Comm settings
    settings.comm_settings.enable_onesided = FLAGS_enable_onesided;
    settings.comm_settings.enable_push_one_by_one =
        FLAGS_enable_push_one_by_one;
    settings.comm_settings.enable_push = !(FLAGS_enable_get);
    settings.comm_settings.enable_overlap = FLAGS_enable_comm_overlap;
    if (FLAGS_enable_flush == "flush-all") {
        settings.comm_settings.enable_flush_all = true;
    } else if (FLAGS_enable_flush == "flush-local") {
        settings.comm_settings.enable_flush_all = false;
        settings.comm_settings.enable_flush_local = true;
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
    settings.use_precond = FLAGS_enable_local_precond;
    metadata.precond_max_block_size = FLAGS_precond_max_block_size;
    settings.explicit_laplacian = FLAGS_explicit_laplacian;
    settings.enable_random_rhs = FLAGS_enable_random_rhs;
    settings.overlap = FLAGS_overlap;
    settings.naturally_ordered_factor = FLAGS_factor_ordering_natural;
    if (FLAGS_partition == "metis") {
        settings.partition =
            SchwarzWrappers::Settings::partition_settings::partition_metis;
        settings.metis_objtype = FLAGS_metis_objtype;
    } else if (FLAGS_partition == "regular") {
        settings.partition =
            SchwarzWrappers::Settings::partition_settings::partition_regular;
    } else if (FLAGS_partition == "regular2d") {
        settings.partition =
            SchwarzWrappers::Settings::partition_settings::partition_regular2d;
    }
    if (FLAGS_local_solver == "iterative-ginkgo") {
        settings.local_solver = SchwarzWrappers::Settings::
            local_solver_settings::iterative_solver_ginkgo;
    } else if (FLAGS_local_solver == "direct-cholmod") {
        settings.local_solver = SchwarzWrappers::Settings::
            local_solver_settings::direct_solver_cholmod;
    } else if (FLAGS_local_solver == "direct-ginkgo") {
        settings.local_solver = SchwarzWrappers::Settings::
            local_solver_settings::direct_solver_ginkgo;
    }

    // The global solution vector to be passed in to the RAS solver.
    std::shared_ptr<gko::matrix::Dense<ValueType>> explicit_laplacian_solution =
        gko::matrix::Dense<ValueType>::create(
            settings.executor->get_master(),
            gko::dim<2>(metadata.global_size, 1));

    if (metadata.my_rank == 0) {
        std::cout << " Running on the " << FLAGS_executor << " executor on "
                  << metadata.num_subdomains << " ranks with "
                  << FLAGS_num_threads << " threads" << std::endl;
        std::cout << " Problem Size: " << metadata.global_size << std::endl;
    }
    SchwarzWrappers::SolverRAS<ValueType, IndexType> solver(settings, metadata);

    solver.initialize();
    solver.run(explicit_laplacian_solution);
    if (FLAGS_timings_file != "null") {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename = FLAGS_timings_file + "_" + rank_string + ".csv";
        write_timings(metadata.time_struct, filename,
                      settings.comm_settings.enable_onesided);
    }
    if (FLAGS_write_comm_data) {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename_send = "num_send_" + rank_string + ".csv";
        std::string filename_recv = "num_recv_" + rank_string + ".csv";
        write_comm_data(metadata.num_subdomains, metadata.my_rank,
                        metadata.comm_data_struct, filename_send,
                        filename_recv);
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

        int req_thread_support = MPI_THREAD_SINGLE;
        int prov_thread_support = MPI_THREAD_SINGLE;
        if (FLAGS_num_threads > 1) {
            req_thread_support = MPI_THREAD_MULTIPLE;
            prov_thread_support = MPI_THREAD_MULTIPLE;
        } else {
            req_thread_support = MPI_THREAD_SINGLE;
            prov_thread_support = MPI_THREAD_SINGLE;
        }

        MPI_Init_thread(&argc, &argv, req_thread_support, &prov_thread_support);
        if (prov_thread_support != req_thread_support) {
            std::cout << "Required thread support is " << req_thread_support
                      << " but provided thread support is only "
                      << prov_thread_support << std::endl;
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
