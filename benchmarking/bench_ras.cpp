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
DEFINE_bool(enable_onesided, false,
            "Use the onesided communication version for the solver");
DEFINE_bool(enable_twosided, true,
            "Use the twosided communication version for the solver");
DEFINE_bool(enable_push_one_by_one, false,
            "Enable push one element after another in onesided");
DEFINE_bool(enable_put_all_local_residual_norms, false,
            "Enable putting of all local residual norms");
DEFINE_bool(enable_comm_overlap, false,
            "Enable overlap of communication and computation");
DEFINE_bool(enable_global_check, false,
            "Use the global convergence check for twosided");
DEFINE_bool(enable_global_tree_check, false,
            "Use the global convergence tree check for onesided");
DEFINE_bool(explicit_laplacian, false,
            "Use the explicit laplacian instead of deal.ii's matrix");
DEFINE_bool(enable_random_rhs, false,
            "Use a random rhs instead of the default 1.0's ");
DEFINE_uint32(overlap, 2, "Overlap between the domains");
DEFINE_string(
    executor, "reference",
    "The executor used to run the solver, one of reference, cuda or omp");
DEFINE_string(enable_flush, "flush_all",
              "The window flush. The choices are flush_local and flush_all");
DEFINE_string(timings_file, "null", "The filename for the timings");
DEFINE_string(partition, "naive",
              "The partitioner used. The choices are metis or naive");
DEFINE_string(local_solver, "direct_cholmod",
              "The local solver used in the local domains. The current choices "
              "include direct_cholmod , direct_ginkgo or iterative_ginkgo");
DEFINE_uint32(num_threads, 1, "Number of threads to bind to a process");
DEFINE_bool(factor_ordering_natural, false,
            "If true uses natural ordering instead of the default optimized "
            "ordering. ");


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
        std::string filename);
    int get_local_rank(MPI_Comm mpi_communicator);
};


template <typename ValueType, typename IndexType>
void BenchRas<ValueType, IndexType>::write_timings(
    std::vector<std::tuple<int, int, int, std::string, std::vector<ValueType>>>
        &time_struct,
    std::string filename)
{
    std::ofstream file;
    file.open(filename);
    for (auto id = 0; id < time_struct.size(); id++)
        std::sort(std::get<4>(time_struct[id]).begin(),
                  std::get<4>(time_struct[id]).end());
    file << "func,total,avg,min,med,max\n";
    for (auto id = 0; id < time_struct.size(); id++) {
        ValueType total_time =
            std::accumulate(std::get<4>(time_struct[id]).begin(),
                            std::get<4>(time_struct[id]).end(), 0.0);
        file << std::get<3>(time_struct[id]) << "," << total_time << ","
             << total_time / (std::get<2>(time_struct[id])) << ","
             << *std::min_element(std::get<4>(time_struct[id]).begin(),
                                  std::get<4>(time_struct[id]).end())
             << ","
             << std::get<4>(
                    time_struct[id])[std::get<4>(time_struct[id]).size() / 2]
             << ","
             << *std::max_element(std::get<4>(time_struct[id]).begin(),
                                  std::get<4>(time_struct[id]).end())
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
    metadata.local_solver_tolerance = FLAGS_local_tol;
    metadata.tolerance = FLAGS_set_tol;
    metadata.max_iters = FLAGS_num_iters;
    metadata.num_subdomains = metadata.comm_size;
    metadata.num_threads = FLAGS_num_threads;
    metadata.oned_laplacian_size = FLAGS_set_1d_laplacian_size;
    metadata.global_size =
        metadata.oned_laplacian_size * metadata.oned_laplacian_size;

    // Set solver settings from command line args.
    // Comm settings
    settings.comm_settings.enable_onesided = FLAGS_enable_onesided;
    settings.comm_settings.enable_push_one_by_one =
        FLAGS_enable_push_one_by_one;
    settings.comm_settings.enable_overlap = FLAGS_enable_comm_overlap;
    if (FLAGS_enable_flush == "flush_all") {
        settings.comm_settings.enable_flush_all = true;
    } else if (FLAGS_enable_flush == "flush_local") {
        settings.comm_settings.enable_flush_all = false;
        settings.comm_settings.enable_flush_local = true;
    }
    // Convergence settings
    settings.convergence_settings.put_all_local_residual_norms =
        FLAGS_enable_put_all_local_residual_norms;
    settings.convergence_settings.enable_global_check =
        FLAGS_enable_global_check;
    settings.convergence_settings.enable_global_simple_tree =
        FLAGS_enable_global_tree_check;

    // General solver settings
    settings.explicit_laplacian = FLAGS_explicit_laplacian;
    settings.enable_random_rhs = FLAGS_enable_random_rhs;
    settings.overlap = FLAGS_overlap;
    settings.naturally_ordered_factor = FLAGS_factor_ordering_natural;
    if (FLAGS_partition == "metis") {
        settings.partition =
            SchwarzWrappers::Settings::partition_settings::partition_metis;
    } else if (FLAGS_partition == "naive") {
        settings.partition =
            SchwarzWrappers::Settings::partition_settings::partition_naive;
    }
    if (FLAGS_local_solver == "iterative_ginkgo") {
        settings.local_solver = SchwarzWrappers::Settings::
            local_solver_settings::iterative_solver_ginkgo;
    } else if (FLAGS_local_solver == "direct_cholmod") {
        settings.local_solver = SchwarzWrappers::Settings::
            local_solver_settings::direct_solver_cholmod;
    } else if (FLAGS_local_solver == "direct_ginkgo") {
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
        std::string filename = FLAGS_timings_file + "_" +
                               std::to_string(metadata.my_rank) + ".csv";
        write_timings(metadata.time_struct, filename);
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

        MPI_Init(&argc, &argv);
        // int req_thread_support = MPI_THREAD_MULTIPLE;
        // int prov_thread_support = MPI_THREAD_MULTIPLE;

        // MPI_Init_thread(&argc, &argv, req_thread_support,
        // &prov_thread_support); if (prov_thread_support != req_thread_support)
        // {
        //     std::cout << "Required thread support is " << req_thread_support
        //               << " but provided thread support is only "
        //               << prov_thread_support << std::endl;
        // }
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
