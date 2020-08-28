
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


#include <schwarz/config.hpp>


#include <exception_helpers.hpp>
#include <process_topology.hpp>
#include <schwarz_base.hpp>
#include <utils.hpp>

#define CHECK_HERE                                                   \
    std::cout << "Here " << __LINE__ << " rank " << metadata.my_rank \
              << std::endl;

namespace schwz {


template <typename ValueType, typename IndexType>
void write_iters_and_residuals(
    int num_subd, int my_rank, int iter_count,
    std::vector<ValueType> &local_res_vec_out,
    std::vector<IndexType> &local_converged_iter_count,
    std::vector<ValueType> &local_converged_resnorm,
    std::vector<ValueType> &local_timestamp, std::string filename)
{
    {
        std::ofstream file;
        file.open(filename);
        file << "iter,resnorm,localiter,localresnorm,timestamp\n";
        for (auto i = 0; i < iter_count; ++i) {
            file << i << "," << local_res_vec_out[i] << ","
                 << local_converged_iter_count[i] << ","
                 << local_converged_resnorm[i] << "," << local_timestamp[i]
                 << "\n";
        }
        file.close();
    }
}


template <typename ValueType, typename IndexType, typename MixedValueType>
SchwarzBase<ValueType, IndexType, MixedValueType>::SchwarzBase(
    Settings &settings, Metadata<ValueType, IndexType> &metadata)
    : Initialize<ValueType, IndexType>(settings, metadata),
      settings(settings),
      metadata(metadata)
{
    using vec_itype = gko::Array<IndexType>;
    using vec_vecshared = gko::Array<IndexType *>;
    metadata.my_local_rank =
        Utils<ValueType, IndexType>::get_local_rank(metadata.mpi_communicator);
    metadata.local_num_procs = Utils<ValueType, IndexType>::get_local_num_procs(
        metadata.mpi_communicator);
    auto my_local_rank = metadata.my_local_rank;
    if (settings.executor_string == "omp") {
        settings.executor = gko::OmpExecutor::create();
        auto exec_info =
            static_cast<gko::OmpExecutor *>(settings.executor.get())
                ->get_exec_info();
        exec_info->bind_to_core(metadata.my_local_rank);

    } else if (settings.executor_string == "cuda") {
        int num_devices = 0;
#if SCHW_HAVE_CUDA
        SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaGetDeviceCount(&num_devices));
#else
        SCHWARZ_NOT_IMPLEMENTED;
#endif
        Utils<ValueType, IndexType>::assert_correct_cuda_devices(
            num_devices, metadata.my_rank);
        settings.executor = gko::CudaExecutor::create(
            my_local_rank, gko::OmpExecutor::create());
        auto exec_info = static_cast<gko::OmpExecutor *>(
                             settings.executor->get_master().get())
                             ->get_exec_info();
        exec_info->bind_to_core(my_local_rank);
        settings.cuda_device_guard =
            std::make_shared<schwz::device_guard>(my_local_rank);

        std::cout << " Rank " << metadata.my_rank << " with local rank "
                  << my_local_rank << " has "
                  << (static_cast<gko::CudaExecutor *>(settings.executor.get()))
                         ->get_device_id()
                  << " id of gpu" << std::endl;
        MPI_Barrier(metadata.mpi_communicator);
    } else if (settings.executor_string == "reference") {
        settings.executor = gko::ReferenceExecutor::create();
        auto exec_info =
            static_cast<gko::ReferenceExecutor *>(settings.executor.get())
                ->get_exec_info();
        exec_info->bind_to_core(my_local_rank);
    }
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void SchwarzBase<ValueType, IndexType, MixedValueType>::initialize(
#if SCHW_HAVE_DEALII
    const dealii::SparseMatrix<ValueType> &matrix,
    const dealii::Vector<ValueType> &system_rhs)
#else
)
#endif
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using vec_itype = gko::Array<IndexType>;
    using vec_vecshared = gko::Array<IndexType *>;
    // Setup the global matrix
    // if explicit_laplacian has been enabled or an external matrix has been
    if (settings.explicit_laplacian || settings.matrix_filename != "null") {
#if !SCHW_HAVE_DEALII
        Initialize<ValueType, IndexType>::setup_global_matrix(
            settings.matrix_filename, metadata.oned_laplacian_size,
            this->global_matrix);
#endif
    } else {
        // If not, then check if deal.ii has been enabled for matrix generation.
#if SCHW_HAVE_DEALII
        Initialize<ValueType, IndexType>::setup_global_matrix(
            settings.matrix_filename, metadata.oned_laplacian_size, matrix,
            this->global_matrix);
#else
        std::cerr << " Explicit laplacian needs to be enabled with the "
                     "--explicit_laplacian flag or deal.ii support needs to be "
                     "enabled to generate the matrices"
                  << std::endl;
        std::exit(-1);
#endif
    }
    this->metadata.global_size = this->global_matrix->get_size()[0];
    auto my_rank = this->metadata.my_rank;
    auto comm_size = this->metadata.comm_size;
    auto num_subdomains = this->metadata.num_subdomains;
    auto global_size = this->metadata.global_size;

    // Setup the right hand side vector.
    std::vector<ValueType> rhs(metadata.global_size, 1.0);
    if (metadata.my_rank == 0 && settings.explicit_laplacian) {
        if (settings.rhs_type == "random") {
            Initialize<ValueType, IndexType>::generate_rhs(rhs);
            std::cout << "Random rhs." << std::endl;
        } else if (settings.rhs_type == "sinusoidal") {
            Initialize<ValueType, IndexType>::generate_sin_rhs(rhs);
            std::cout << "Sinusoidal rhs." << std::endl;
        } else if (settings.rhs_type == "dipole") {
            Initialize<ValueType, IndexType>::generate_dipole_rhs(rhs);
            std::cout << "Dipole rhs." << std::endl;
        } else {
            std::cout << "Default rhs with ones." << std::endl;
        } 
    }
#if SCHW_HAVE_DEALII
    if (metadata.my_rank == 0 && !settings.explicit_laplacian) {
        std::copy(system_rhs.begin(), system_rhs.begin() + metadata.global_size,
                  rhs.begin());
    }
#endif
    auto mpi_vtype = schwz::mpi::get_mpi_datatype(*rhs.data());
    MPI_Bcast(rhs.data(), metadata.global_size, mpi_vtype, 0, MPI_COMM_WORLD);


    // Some arrays for partitioning and local matrix creation.
    metadata.first_row = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_itype>());
    metadata.permutation = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), global_size),
        std::default_delete<vec_itype>());
    metadata.i_permutation = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), global_size),
        std::default_delete<vec_itype>());
    metadata.global_to_local = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), global_size),
        std::default_delete<vec_itype>());
    metadata.local_to_global = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), global_size),
        std::default_delete<vec_itype>());

    // Some arrays for communication.
    comm_struct.local_neighbors_in = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_itype>());
    comm_struct.local_neighbors_out = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_itype>());
    comm_struct.neighbors_in = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_itype>());
    comm_struct.neighbors_out = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_itype>());
    comm_struct.is_local_neighbor = std::vector<bool>(num_subdomains + 1, 0);
    comm_struct.global_get = std::shared_ptr<vec_vecshared>(
        new vec_vecshared(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_vecshared>());
    comm_struct.global_put = std::shared_ptr<vec_vecshared>(
        new vec_vecshared(settings.executor->get_master(), num_subdomains + 1),
        std::default_delete<vec_vecshared>());
    // Need this to initialize the arrays with zeros.
    std::vector<IndexType> temp(num_subdomains + 1, 0);
    comm_struct.get_displacements = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), temp.begin(),
                      temp.end()),
        std::default_delete<vec_itype>());
    comm_struct.put_displacements = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), temp.begin(),
                      temp.end()),
        std::default_delete<vec_itype>());

    // Partition the global matrix.
    Initialize<ValueType, IndexType>::partition(
        settings, metadata, this->global_matrix, this->partition_indices);

    // Setup the local matrices on each of the subddomains.
    this->setup_local_matrices(this->settings, this->metadata,
                               this->partition_indices, this->global_matrix,
                               this->local_matrix, this->interface_matrix);
    std::cout << "Subdomain " << this->metadata.my_rank
              << " has local problem size " << this->local_matrix->get_size()[0]
              << " with " << this->local_matrix->get_num_stored_elements()
              << " non-zeros " << std::endl;
    // Debug to print matrices.
    if (settings.print_matrices && settings.executor_string != "cuda") {
        Utils<ValueType, IndexType>::print_matrix(
            this->local_matrix.get(), metadata.my_rank, "local_mat");
        Utils<ValueType, IndexType>::print_matrix(this->interface_matrix.get(),
                                                  metadata.my_rank, "int_mat");
    }

    // Setup the local vectors on each of the subddomains.
    Initialize<ValueType, IndexType>::setup_vectors(
        this->settings, this->metadata, rhs, this->local_rhs, this->global_rhs,
        this->local_solution);

    // Setup the local solver on each of the subddomains.
    Solve<ValueType, IndexType, MixedValueType>::setup_local_solver(
        this->settings, this->metadata, this->local_matrix, this->triangular_factor_l,
        this->triangular_factor_u, this->local_perm, this->local_inv_perm,
        this->local_rhs);
    // Setup the communication buffers on each of the subddomains.
    this->setup_comm_buffers();
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void gather_comm_data(
    int num_subdomains,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
    std::vector<std::tuple<int, std::vector<std::tuple<int, int>>,
                           std::vector<std::tuple<int, int>>, int, int>>
        &comm_data_struct)
{
    for (auto i = 0; i < num_subdomains; ++i) {
        auto owning_subd = i;
        auto num_neighbors_out = comm_struct.num_neighbors_out;
        auto num_neighbors_in = comm_struct.num_neighbors_in;
        auto neighbors_in = comm_struct.neighbors_in->get_data();
        auto neighbors_out = comm_struct.neighbors_out->get_data();
        auto global_put = comm_struct.global_put->get_data();
        auto global_get = comm_struct.global_get->get_data();

        std::vector<int> count_out(num_subdomains, 0);
        std::vector<int> count_in(num_subdomains, 0);
        std::vector<std::tuple<int, int>> send_tuple;
        std::vector<std::tuple<int, int>> recv_tuple;
        for (auto j = 0; j < num_neighbors_out; ++j) {
            send_tuple.push_back(
                std::make_tuple(neighbors_out[j], (global_put[j])[0]));
            count_out[neighbors_out[j]] = 1;
        }

        for (auto j = 0; j < num_neighbors_in; ++j) {
            recv_tuple.push_back(
                std::make_tuple(neighbors_in[j], (global_get[j])[0]));
            count_in[neighbors_in[j]] = 1;
        }
        for (auto j = 0; j < num_subdomains; ++j) {
            if (count_out[j] == 0) {
                send_tuple.push_back(std::make_tuple(j, 0));
            }
            if (count_in[j] == 0) {
                recv_tuple.push_back(std::make_tuple(j, 0));
            }
        }
        comm_data_struct.push_back(std::make_tuple(owning_subd, recv_tuple,
                                                   send_tuple, num_neighbors_in,
                                                   num_neighbors_out));
    }
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void SchwarzBase<ValueType, IndexType, MixedValueType>::run(
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    if (!solution.get()) {
        solution =
            vec_vtype::create(settings.executor->get_master(),
                              gko::dim<2>(this->metadata.global_size, 1));
    }
    MixedValueType dummy1 = 0.0;
    ValueType dummy2 = 1.0;

    if (metadata.my_rank == 0) {
        std::cout << " MixedValueType: " << typeid(dummy1).name()
                  << " ValueType: " << typeid(dummy2).name() << std::endl;
    }
    // The main solution vector
    std::shared_ptr<vec_vtype> global_solution = vec_vtype::create(
        this->settings.executor, gko::dim<2>(this->metadata.global_size, 1));
    // The previous iteration solution vector
    std::shared_ptr<vec_vtype> prev_global_solution = vec_vtype::create(
        this->settings.executor, gko::dim<2>(this->metadata.global_size, 1));
    // A work vector.
    std::shared_ptr<vec_vtype> work_vector = vec_vtype::create(
        settings.executor, gko::dim<2>(2 * this->metadata.local_size_x, 1));
    // An initial guess.
    std::shared_ptr<vec_vtype> init_guess = vec_vtype::create(
        settings.executor, gko::dim<2>(this->metadata.local_size_x, 1));
    // init_guess->copy_from(local_rhs.get());

    if (settings.executor_string == "omp") {
        ValueType sum_rhs = std::accumulate(
            local_rhs->get_values(),
            local_rhs->get_values() + local_rhs->get_size()[0], 0.0);
        std::cout << " Rank " << this->metadata.my_rank << " sum local rhs "
                  << sum_rhs << std::endl;
    }

    // Initialize all vectors - tbd

    // std::vector<IndexType> local_converged_iter_count;

    // Setup the windows for the onesided communication.
    this->setup_windows(this->settings, this->metadata, global_solution);

    const auto solver_settings =
        (Settings::local_solver_settings::direct_solver_cholmod |
         Settings::local_solver_settings::direct_solver_umfpack |
         Settings::local_solver_settings::direct_solver_ginkgo |
         Settings::local_solver_settings::iterative_solver_dealii |
         Settings::local_solver_settings::iterative_solver_ginkgo) &
        settings.local_solver;
    prev_global_solution->copy_from(gko::lend(global_solution));

    ValueType local_residual_norm = -1.0, local_residual_norm0 = -1.0,
              global_residual_norm = 0.0, global_residual_norm0 = -1.0;
    metadata.iter_count = 0;
    int num_converged_procs = 0;

    std::ofstream fps;  // file for sending log
    std::ofstream fpr;  // file for receiving log

    if (settings.debug_print) {
        // Opening files for event logs
        char send_name[30], recv_name[30], pe_str[3];
        sprintf(pe_str, "%d", metadata.my_rank);

        strcpy(send_name, "send");
        strcat(send_name, pe_str);
        strcat(send_name, ".txt");

        strcpy(recv_name, "recv");
        strcat(recv_name, pe_str);
        strcat(recv_name, ".txt");

        fps.open(send_name);
        fpr.open(recv_name);
    }

    if (metadata.my_rank == 0) {
        std::cout << "Send history - " << metadata.sent_history
                  << ", Recv history - " << metadata.recv_history << std::endl;
        std::cout << "Thres type - " << settings.thres_type << std::endl;
        std::cout << "Overlap - " << settings.overlap << std::endl;
    }   

    auto start_time = std::chrono::steady_clock::now();

    for (; metadata.iter_count < metadata.max_iters; ++(metadata.iter_count)) {
        // Exchange the boundary values. The communication part.
        MEASURE_ELAPSED_FUNC_TIME(
            this->exchange_boundary(settings, metadata, prev_global_solution,
                                    global_solution, fps, fpr),
            0, metadata.my_rank, boundary_exchange, metadata.iter_count);
        prev_global_solution->copy_from(gko::lend(global_solution));

        // Update the boundary and interior values after the exchanging from
        // other processes.
        MEASURE_ELAPSED_FUNC_TIME(
            this->update_boundary(settings, metadata, this->local_solution,
                                  this->local_rhs, global_solution,
                                  this->interface_matrix),
            1, metadata.my_rank, boundary_update, metadata.iter_count);

        // Check for the convergence of the solver.
        // num_converged_procs = 0;
        MEASURE_ELAPSED_FUNC_TIME(
            (Solve<ValueType, IndexType, MixedValueType>::check_convergence(
                settings, metadata, this->comm_struct, this->convergence_vector,
                global_solution, this->local_solution, this->local_matrix,
                work_vector, local_residual_norm, local_residual_norm0,
                global_residual_norm, global_residual_norm0,
                num_converged_procs)),
            2, metadata.my_rank, convergence_check, metadata.iter_count);

        // break if the solution diverges.
        if (std::isnan(global_residual_norm) || global_residual_norm > 1e12) {
            std::cout << " Rank " << metadata.my_rank << " diverged in "
                      << metadata.iter_count << " iters " << std::endl;
            std::exit(-1);
        }

        // break if all processes detect that all other processes have
        // converged otherwise continue iterations.
        if (num_converged_procs == metadata.num_subdomains) {
            break;
        } else {
            MEASURE_ELAPSED_FUNC_TIME(
                (Solve<ValueType, IndexType, MixedValueType>::local_solve(
                    settings, metadata, this->local_matrix,
                    this->triangular_factor_l, this->triangular_factor_u,
                    this->local_perm, this->local_inv_perm, work_vector,
                    init_guess, this->local_solution)),
                3, metadata.my_rank, local_solve, metadata.iter_count);

            // Gather the local vector into the locally global vector for
            // communication.
            MEASURE_ELAPSED_FUNC_TIME(
                (Communicate<ValueType, IndexType, MixedValueType>::
                     local_to_global_vector(settings, metadata,
                                            this->local_solution,
                                            global_solution)),
                4, metadata.my_rank, expand_local_vec, metadata.iter_count);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto elapsed_time = std::chrono::duration<ValueType>(
        std::chrono::steady_clock::now() - start_time);

    if (settings.debug_print) {
        // Closing event log files
        fps.close();
        fpr.close();
    }

    // adding 1 to include the 0-th iteration
    metadata.iter_count = metadata.iter_count + 1;

    // number of messages a PE would send without event-based
    int noevent_msg_count = metadata.iter_count * num_neighbors_out;

    int total_events = 0;

    // Printing msg count
    for (int k = 0; k < num_neighbors_out; k++) {
        std::cout << " Rank: " << metadata.my_rank << " to " << neighbors_out[k]
                  << " : " << this->comm_struct.msg_count->get_data()[k];
        total_events += this->comm_struct.msg_count->get_data()[k];
    }
    std::cout << std::endl;

    // Total no of messages in all PEs
    MPI_Allreduce(MPI_IN_PLACE, &total_events, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &noevent_msg_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (metadata.my_rank == 0) {
        std::cout << "Total number of events - " << total_events << std::endl;
        std::cout << "Total number of msgs without event - "
                  << noevent_msg_count << std::endl;
    }

    // Write the residuals and iterations to files
    if (settings.write_iters_and_residuals &&
        solver_settings ==
            Settings::local_solver_settings::iterative_solver_ginkgo) {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename = "iter_res_" + rank_string + ".csv";
        write_iters_and_residuals(
            metadata.num_subdomains, metadata.my_rank,
            metadata.post_process_data.local_residual_vector_out.size(),
            metadata.post_process_data.local_residual_vector_out,
            metadata.post_process_data.local_converged_iter_count,
            metadata.post_process_data.local_converged_resnorm,
            metadata.post_process_data.local_timestamp, filename);
    }
    if (num_converged_procs < metadata.num_subdomains) {
        std::cout << "Rank " << metadata.my_rank << " did not converge in "
                  << metadata.iter_count << " iterations." << std::endl;
    } else {
        std::cout << " Rank " << metadata.my_rank << " converged in "
                  << metadata.iter_count << " iterations " << std::endl;
        ValueType mat_norm = -1.0, rhs_norm = -1.0, sol_norm = -1.0,
                  residual_norm = -1.0;

        // Compute the final residual norm. Also gathers the solution from all
        // subdomains.
        Solve<ValueType, IndexType, MixedValueType>::compute_residual_norm(
            settings, metadata, global_matrix, global_rhs, global_solution,
            mat_norm, rhs_norm, sol_norm, residual_norm);
        gather_comm_data<ValueType, IndexType, MixedValueType>(
            metadata.num_subdomains, this->comm_struct,
            metadata.comm_data_struct);
        // clang-format off
        if (metadata.my_rank == 0)
          {
            std::cout
              << " residual norm " << residual_norm << "\n"
              << " relative residual norm of solution " << residual_norm/rhs_norm << "\n"
              << " Time taken for solve " << elapsed_time.count()
              << std::endl;
          }
        // clang-format on
    }
    if (metadata.my_rank == 0) {
        solution->copy_from(global_solution.get());
    }

    // Communicate<ValueType, IndexType>::clear(settings);
}

#define DECLARE_SCHWARZ_BASE(ValueType, IndexType, MixedValueType) \
    class SchwarzBase<ValueType, IndexType, MixedValueType>
INSTANTIATE_FOR_EACH_VALUE_MIXEDVALUE_AND_INDEX_TYPE(DECLARE_SCHWARZ_BASE);
#undef DECLARE_SCHWARZ_BASE


}  // namespace schwz
