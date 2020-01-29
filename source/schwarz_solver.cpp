
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

//This is branch test

#include <schwarz/config.hpp>


#include <exception_helpers.hpp>
#include <process_topology.hpp>
#include <schwarz_solver.hpp>
#include <utils.hpp>


namespace SchwarzWrappers {


template <typename ValueType, typename IndexType>
SolverBase<ValueType, IndexType>::SolverBase(
    Settings &settings, Metadata<ValueType, IndexType> &metadata)
    : Initialize<ValueType, IndexType>(settings, metadata),
      settings(settings),
      metadata(metadata)
{
    using vec_itype = gko::Array<IndexType>;
    using vec_vecshared = gko::Array<IndexType *>;
    auto local_rank =
        Utils<ValueType, IndexType>::get_local_rank(metadata.mpi_communicator);
    auto local_num_procs = Utils<ValueType, IndexType>::get_local_num_procs(
        metadata.mpi_communicator);

    if (settings.executor_string == "omp") {
        settings.executor = gko::OmpExecutor::create();
        auto exec_info =
            static_cast<gko::OmpExecutor *>(settings.executor.get())
                ->get_exec_info();
        exec_info->bind_to_core(local_rank);

    } else if (settings.executor_string == "cuda") {
        int num_devices = 0;
#if SCHW_HAVE_CUDA
        SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaGetDeviceCount(&num_devices));
#else
        SCHWARZ_NOT_IMPLEMENTED;
#endif
        if (num_devices > 0) {
            if (metadata.my_rank == 0) {
                std::cout << " Number of available devices: " << num_devices
                          << std::endl;
            }
        } else {
            std::cout << " No CUDA devices available for rank "
                      << metadata.my_rank << std::endl;
            std::exit(-1);
        }
        settings.executor =
            gko::CudaExecutor::create(local_rank, gko::OmpExecutor::create());
        auto exec_info = static_cast<gko::OmpExecutor *>(
                             settings.executor->get_master().get())
                             ->get_exec_info();
        exec_info->bind_to_core(local_rank);
        settings.cuda_device_guard =
            std::make_shared<SchwarzWrappers::device_guard>(local_rank);

        std::cout << " Rank " << metadata.my_rank << " with local rank "
                  << local_rank << " has "
                  << (static_cast<gko::CudaExecutor *>(settings.executor.get()))
                         ->get_device_id()
                  << " id of gpu" << std::endl;
        MPI_Barrier(metadata.mpi_communicator);
    } else if (settings.executor_string == "reference") {
        settings.executor = gko::ReferenceExecutor::create();
        auto exec_info =
            static_cast<gko::ReferenceExecutor *>(settings.executor.get())
                ->get_exec_info();
        exec_info->bind_to_core(local_rank);
    }

    auto my_rank = this->metadata.my_rank;
    auto comm_size = this->metadata.comm_size;
    auto num_subdomains = this->metadata.num_subdomains;
    auto global_size = this->metadata.global_size;

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
    comm_struct.neighbors_in = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains),
        std::default_delete<vec_itype>());
    comm_struct.neighbors_out = std::shared_ptr<vec_itype>(
        new vec_itype(settings.executor->get_master(), num_subdomains),
        std::default_delete<vec_itype>());
    comm_struct.global_get = std::shared_ptr<vec_vecshared>(
        new vec_vecshared(settings.executor->get_master(), num_subdomains),
        std::default_delete<vec_vecshared>());
    comm_struct.global_put = std::shared_ptr<vec_vecshared>(
        new vec_vecshared(settings.executor->get_master(), num_subdomains),
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
}


template <typename ValueType, typename IndexType>
void SolverBase<ValueType, IndexType>::initialize()
{
    using vec_vtype = gko::matrix::Dense<ValueType>;

    // Setup the right hand side vector.
    std::vector<ValueType> rhs(metadata.global_size, 1.0);
    if (settings.enable_random_rhs && settings.explicit_laplacian) {
        if (metadata.my_rank == 0) {
            Initialize<ValueType, IndexType>::generate_rhs(rhs);
        }
        auto mpi_vtype = boost::mpi::get_mpi_datatype(*rhs.data());
        MPI_Bcast(rhs.data(), metadata.global_size, mpi_vtype, 0,
                  MPI_COMM_WORLD);
    }

    // Setup the global matrix
    if (settings.explicit_laplacian) {
        Initialize<ValueType, IndexType>::setup_global_matrix_laplacian(
            metadata.oned_laplacian_size, this->global_matrix);
    } else {
        std::cerr << " Explicit laplacian needs to be enabled with the "
                     "--explicit_laplacian flag or deal.ii support needs to be "
                     "enabled to generate the matrices"
                  << std::endl;
        std::exit(-1);
    }
    // Partition the global matrix.
    Initialize<ValueType, IndexType>::partition(
        settings, metadata, this->global_matrix, this->partition_indices);

    // Setup the local matrices on each of the subddomains.
    this->setup_local_matrices(this->settings, this->metadata,
                               this->partition_indices, this->global_matrix,
                               this->local_matrix, this->interface_matrix);
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
        this->local_solution, this->global_solution);

    // Setup the local solver on each of the subddomains.
    Solve<ValueType, IndexType>::setup_local_solver(
        this->settings, metadata, this->local_matrix, this->triangular_factor,
        this->local_rhs);

    // Setup the communication buffers on each of the subddomains.
    this->setup_comm_buffers();
}

#if SCHW_HAVE_DEALII
template <typename ValueType, typename IndexType>
void SolverBase<ValueType, IndexType>::initialize(
    const dealii::SparseMatrix<ValueType> &matrix,
    const dealii::Vector<ValueType> &system_rhs)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;

    // Setup the right hand side vector.
    std::vector<ValueType> rhs(metadata.global_size, 1.0);
    if (settings.enable_random_rhs && settings.explicit_laplacian) {
        if (metadata.my_rank == 0) {
            Initialize<ValueType, IndexType>::generate_rhs(rhs);
        }
        auto mpi_vtype = boost::mpi::get_mpi_datatype(*rhs.data());
        MPI_Bcast(rhs.data(), metadata.global_size, mpi_vtype, 0,
                  MPI_COMM_WORLD);
    }
    if (metadata.my_rank == 0 && !settings.explicit_laplacian) {
        std::copy(system_rhs.begin(), system_rhs.begin() + metadata.global_size,
                  rhs.begin());
    }

    // Setup the global matrix
    if (settings.explicit_laplacian) {
        Initialize<ValueType, IndexType>::setup_global_matrix_laplacian(
            metadata.oned_laplacian_size, this->global_matrix);
    } else {
        Initialize<ValueType, IndexType>::setup_global_matrix(
            matrix, this->global_matrix);
    }

    // Partition the global matrix.
    Initialize<ValueType, IndexType>::partition(
        settings, metadata, this->global_matrix, this->partition_indices);

    // Setup the local matrices on each of the subddomains.
    this->setup_local_matrices(this->settings, this->metadata,
                               this->partition_indices, this->global_matrix,
                               this->local_matrix, this->interface_matrix);
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
        this->local_solution, this->global_solution);

    // Setup the local solver on each of the subddomains.
    Solve<ValueType, IndexType>::setup_local_solver(
        this->settings, metadata, this->local_matrix, this->triangular_factor,
        this->local_rhs);

    // Setup the communication buffers on each of the subddomains.
    this->setup_comm_buffers();
}
#endif


template <typename ValueType, typename IndexType>
void gather_comm_data(
    int num_subdomains,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
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


template <typename ValueType, typename IndexType>
void SolverBase<ValueType, IndexType>::run(
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    // The main solution vector
    std::shared_ptr<vec_vtype> solution_vector = vec_vtype::create(
        settings.executor, gko::dim<2>(metadata.global_size, 1));
    // A temp local solution
    std::shared_ptr<vec_vtype> init_guess =
        vec_vtype::create(settings.executor, this->local_solution->get_size());
    // A global gathered solution of the previous iteration.
    std::shared_ptr<vec_vtype> global_old_solution = vec_vtype::create(
        settings.executor, gko::dim<2>(metadata.global_size, 1));
    // Setup the windows for the onesided communication.
    this->setup_windows(settings, metadata, solution_vector);

    const auto solver_settings =
        (Settings::local_solver_settings::direct_solver_cholmod |
         Settings::local_solver_settings::direct_solver_ginkgo |
         Settings::local_solver_settings::iterative_solver_dealii |
         Settings::local_solver_settings::iterative_solver_ginkgo) &
        settings.local_solver;

    ValueType local_residual_norm = -1.0, local_residual_norm0 = -1.0,
              global_residual_norm = 0.0, global_residual_norm0 = -1.0;
    metadata.iter_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    int num_converged_procs = 0;

    for (; metadata.iter_count < metadata.max_iters; ++(metadata.iter_count)) {
        // Exchange the boundary values. The communication part.
        MEASURE_ELAPSED_FUNC_TIME(
            this->exchange_boundary(settings, metadata, solution_vector), 0,
            metadata.my_rank, boundary_exchange, metadata.iter_count);

        // Update the boundary and interior values after the exchanging from
        // other processes.
        MEASURE_ELAPSED_FUNC_TIME(
            this->update_boundary(settings, metadata, this->local_solution,
                                  this->local_rhs, solution_vector,
                                  global_old_solution, this->interface_matrix),
            1, metadata.my_rank, boundary_update, metadata.iter_count);

        // Check for the convergence of the solver.
        num_converged_procs = 0;
        MEASURE_ELAPSED_FUNC_TIME(
            (Solve<ValueType, IndexType>::check_convergence(
                settings, metadata, this->comm_struct, this->convergence_vector,
                global_old_solution, this->local_solution, this->local_matrix,
                local_residual_norm, local_residual_norm0, global_residual_norm,
                global_residual_norm0, num_converged_procs)),
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
            std::cout << " Rank " << metadata.my_rank << " converged in "
                      << metadata.iter_count << " iters " << std::endl;
            break;
        } else {
            MEASURE_ELAPSED_FUNC_TIME(
                (Solve<ValueType, IndexType>::local_solve(
                    settings, metadata, this->triangular_factor, init_guess,
                    this->local_solution)),
                3, metadata.my_rank, local_solve, metadata.iter_count);
            // init_guess->copy_from(this->local_solution.get());
            // Gather the local vector into the locally global vector for
            // communication.
            MEASURE_ELAPSED_FUNC_TIME(
                (Communicate<ValueType, IndexType>::local_to_global_vector(
                    settings, metadata, this->local_solution, solution_vector)),
                4, metadata.my_rank, expand_local_vec, metadata.iter_count);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto elapsed_time = std::chrono::duration<ValueType>(
        std::chrono::steady_clock::now() - start_time);
    ValueType mat_norm = -1.0, rhs_norm = -1.0, sol_norm = -1.0,
              residual_norm = -1.0;
    // Compute the final residual norm. Also gathers the solution from all
    // subdomains.
    Solve<ValueType, IndexType>::compute_residual_norm(
        settings, metadata, global_matrix, global_rhs, solution_vector,
        mat_norm, rhs_norm, sol_norm, residual_norm);
    gather_comm_data<ValueType, IndexType>(
        metadata.num_subdomains, this->comm_struct, metadata.comm_data_struct);
    // clang-format off
    if (metadata.my_rank == 0)
      {
        std::cout
              << " residual norm " << residual_norm << "\n"
              << " relative residual norm of solution " << residual_norm/rhs_norm << "\n"
              << " Time taken for solve " << elapsed_time.count()
              << std::endl;
        if (num_converged_procs < metadata.num_subdomains)
          {
            std::cout << " Did not converge in " << metadata.iter_count
                      << " iterations."
                      << std::endl;
          }
      }
    // clang-format on
    if (metadata.my_rank == 0) {
        solution->copy_from(solution_vector.get());
    }
}


template <typename ValueType, typename IndexType>
SolverRAS<ValueType, IndexType>::SolverRAS(
    Settings &settings, Metadata<ValueType, IndexType> &metadata,
    const AdditionalData &data)
    : SolverBase<ValueType, IndexType>(settings, metadata),
      additional_data(data)
{}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::setup_local_matrices(
    Settings &settings, Metadata<ValueType, IndexType> &metadata,
    std::vector<unsigned int> &partition_indices,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &interface_matrix)
{
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec_itype = gko::Array<IndexType>;
    auto my_rank = metadata.my_rank;
    auto comm_size = metadata.comm_size;
    auto num_subdomains = metadata.num_subdomains;
    auto global_size = metadata.global_size;
    auto mpi_itype = boost::mpi::get_mpi_datatype(*partition_indices.data());

    MPI_Bcast(partition_indices.data(), global_size, mpi_itype, 0,
              MPI_COMM_WORLD);

    std::vector<IndexType> local_p_size(num_subdomains);
    auto global_to_local = metadata.global_to_local->get_data();
    auto local_to_global = metadata.local_to_global->get_data();

    auto first_row = metadata.first_row->get_data();
    auto permutation = metadata.permutation->get_data();
    auto i_permutation = metadata.i_permutation->get_data();

    auto nb = (global_size + num_subdomains - 1) / num_subdomains;
    auto partition_settings =
        (Settings::partition_settings::partition_zoltan |
         Settings::partition_settings::partition_metis |
         Settings::partition_settings::partition_regular |
         Settings::partition_settings::partition_regular2d |
         Settings::partition_settings::partition_custom) &
        settings.partition;

    IndexType *gmat_row_ptrs = global_matrix->get_row_ptrs();
    IndexType *gmat_col_idxs = global_matrix->get_col_idxs();
    ValueType *gmat_values = global_matrix->get_values();

    // default local p size set for 1 subdomain.
    first_row[0] = 0;
    for (auto p = 0; p < num_subdomains; ++p) {
        local_p_size[p] = std::min(global_size - first_row[p], nb);
        first_row[p + 1] = first_row[p] + local_p_size[p];
    }

    if (partition_settings == Settings::partition_settings::partition_metis ||
        partition_settings ==
            Settings::partition_settings::partition_regular2d) {
        if (num_subdomains > 1) {
            for (auto p = 0; p < num_subdomains; p++) {
                local_p_size[p] = 0;
            }
            for (auto i = 0; i < global_size; i++) {
                local_p_size[partition_indices[i]]++;
            }
            first_row[0] = 0;
            for (auto p = 0; p < num_subdomains; ++p) {
                first_row[p + 1] = first_row[p] + local_p_size[p];
            }
            // permutation
            for (auto i = 0; i < global_size; i++) {
                permutation[first_row[partition_indices[i]]] = i;
                first_row[partition_indices[i]]++;
            }
            for (auto p = num_subdomains; p > 0; p--) {
                first_row[p] = first_row[p - 1];
            }
            first_row[0] = 0;

            // iperm
            for (auto i = 0; i < global_size; i++) {
                i_permutation[permutation[i]] = i;
            }
        }

        auto gmat_temp = mtx::create(settings.executor->get_master(),
                                     global_matrix->get_size(),
                                     global_matrix->get_num_stored_elements());
        auto nnz = 0;
        gmat_temp->get_row_ptrs()[0] = 0;
        for (auto row = 0; row < metadata.global_size; ++row) {
            for (auto col = gmat_row_ptrs[permutation[row]];
                 col < gmat_row_ptrs[permutation[row] + 1]; ++col) {
                gmat_temp->get_col_idxs()[nnz] =
                    i_permutation[gmat_col_idxs[col]];
                gmat_temp->get_values()[nnz] = gmat_values[col];
                nnz++;
            }
            gmat_temp->get_row_ptrs()[row + 1] = nnz;
        }
        global_matrix->copy_from(gmat_temp.get());
    }
    for (auto i = 0; i < global_size; i++) {
        global_to_local[i] = 0;
        local_to_global[i] = 0;
    }
    auto num = 0;
    for (auto i = first_row[my_rank]; i < first_row[my_rank + 1]; i++) {
        global_to_local[i] = 1 + num;
        local_to_global[num] = i;
        num++;
    }

    IndexType old = 0;
    for (auto k = 1; k < settings.overlap; k++) {
        auto now = num;
        for (auto i = old; i < now; i++) {
            for (auto j = gmat_row_ptrs[local_to_global[i]];
                 j < gmat_row_ptrs[local_to_global[i] + 1]; j++) {
                if (global_to_local[gmat_col_idxs[j]] == 0) {
                    local_to_global[num] = gmat_col_idxs[j];
                    global_to_local[gmat_col_idxs[j]] = 1 + num;
                    num++;
                }
            }
        }
        old = now;
    }
    metadata.local_size = local_p_size[my_rank];
    metadata.local_size_x = num;
    metadata.local_size_o = global_size;
    auto local_size = metadata.local_size;
    auto local_size_x = metadata.local_size_x;

    metadata.overlap_size = num - metadata.local_size;
    metadata.overlap_row = std::shared_ptr<vec_itype>(
        new vec_itype(gko::Array<IndexType>::view(
            settings.executor, metadata.overlap_size,
            &(metadata.local_to_global->get_data()[metadata.local_size]))),
        std::default_delete<vec_itype>());

    auto nnz_local = 0;
    auto nnz_interface = 0;

    for (auto i = first_row[my_rank]; i < first_row[my_rank + 1]; ++i) {
        for (auto j = gmat_row_ptrs[i]; j < gmat_row_ptrs[i + 1]; j++) {
            if (global_to_local[gmat_col_idxs[j]] != 0) {
                nnz_local++;
            } else {
                std::cout << " debug: invalid edge?" << std::endl;
            }
        }
    }
    auto temp = 0;
    for (auto k = 0; k < metadata.overlap_size; k++) {
        temp = metadata.overlap_row->get_data()[k];
        for (auto j = gmat_row_ptrs[temp]; j < gmat_row_ptrs[temp + 1]; j++) {
            if (global_to_local[gmat_col_idxs[j]] != 0) {
                nnz_local++;
            } else {
                nnz_interface++;
            }
        }
    }

    std::shared_ptr<mtx> local_matrix_compute;
    local_matrix_compute = mtx::create(settings.executor->get_master(),
                                       gko::dim<2>(local_size_x), nnz_local);
    IndexType *lmat_row_ptrs = local_matrix_compute->get_row_ptrs();
    IndexType *lmat_col_idxs = local_matrix_compute->get_col_idxs();
    ValueType *lmat_values = local_matrix_compute->get_values();

    std::shared_ptr<mtx> interface_matrix_compute;
    if (nnz_interface > 0) {
        interface_matrix_compute =
            mtx::create(settings.executor->get_master(),
                        gko::dim<2>(local_size_x), nnz_interface);
    } else {
        interface_matrix_compute = mtx::create(settings.executor->get_master());
    }

    IndexType *imat_row_ptrs = interface_matrix_compute->get_row_ptrs();
    IndexType *imat_col_idxs = interface_matrix_compute->get_col_idxs();
    ValueType *imat_values = interface_matrix_compute->get_values();

    num = 0;
    nnz_local = 0;
    auto nnz_interface_temp = 0;
    lmat_row_ptrs[0] = nnz_local;
    if (nnz_interface > 0) {
        imat_row_ptrs[0] = nnz_interface_temp;
    }
    // Local interior matrix
    for (auto i = first_row[my_rank]; i < first_row[my_rank + 1]; ++i) {
        for (auto j = gmat_row_ptrs[i]; j < gmat_row_ptrs[i + 1]; ++j) {
            if (global_to_local[gmat_col_idxs[j]] != 0) {
                lmat_col_idxs[nnz_local] =
                    global_to_local[gmat_col_idxs[j]] - 1;
                lmat_values[nnz_local] = gmat_values[j];
                nnz_local++;
            }
        }
        if (nnz_interface > 0) {
            imat_row_ptrs[num + 1] = nnz_interface_temp;
        }
        lmat_row_ptrs[num + 1] = nnz_local;
        num++;
    }

    // Interface matrix
    if (nnz_interface > 0) {
        nnz_interface = 0;
        for (auto k = 0; k < metadata.overlap_size; k++) {
            temp = metadata.overlap_row->get_data()[k];
            for (auto j = gmat_row_ptrs[temp]; j < gmat_row_ptrs[temp + 1];
                 j++) {
                if (global_to_local[gmat_col_idxs[j]] != 0) {
                    lmat_col_idxs[nnz_local] =
                        global_to_local[gmat_col_idxs[j]] - 1;
                    lmat_values[nnz_local] = gmat_values[j];
                    nnz_local++;
                } else {
                    imat_col_idxs[nnz_interface] = gmat_col_idxs[j];
                    imat_values[nnz_interface] = gmat_values[j];
                    nnz_interface++;
                }
            }
            lmat_row_ptrs[num + 1] = nnz_local;
            imat_row_ptrs[num + 1] = nnz_interface;
            num++;
        }
    }
    auto now = num;
    for (auto i = old; i < now; i++) {
        for (auto j = gmat_row_ptrs[local_to_global[i]];
             j < gmat_row_ptrs[local_to_global[i] + 1]; j++) {
            if (global_to_local[gmat_col_idxs[j]] == 0) {
                local_to_global[num] = gmat_col_idxs[j];
                global_to_local[gmat_col_idxs[j]] = 1 + num;
                num++;
            }
        }
    }

    local_matrix = mtx::create(settings.executor);
    local_matrix->copy_from(gko::lend(local_matrix_compute));
    interface_matrix = mtx::create(settings.executor);
    interface_matrix->copy_from(gko::lend(interface_matrix_compute));
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::setup_comm_buffers()
{
    using vec_itype = gko::Array<IndexType>;
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using vec_request = gko::Array<MPI_Request>;
    using vec_vecshared = gko::Array<IndexType *>;
    auto metadata = this->metadata;
    auto settings = this->settings;
    auto comm_struct = this->comm_struct;
    auto my_rank = metadata.my_rank;
    auto num_subdomains = metadata.num_subdomains;
    auto first_row = metadata.first_row->get_data();
    auto global_to_local = metadata.global_to_local->get_data();
    auto local_to_global = metadata.local_to_global->get_data();
    auto comm_size = metadata.comm_size;

    auto neighbors_in = this->comm_struct.neighbors_in->get_data();
    auto global_get = this->comm_struct.global_get->get_data();

    this->comm_struct.num_neighbors_in = 0;
    int num_recv = 0;
    std::vector<int> recv(num_subdomains, 0);
    for (auto p = 0; p < num_subdomains; p++) {
        if (p != my_rank) {
            int count = 0;
            for (auto i = first_row[p]; i < first_row[p + 1]; i++) {
                if (global_to_local[i] != 0) {
                    count++;
                }
            }
            if (count > 0) {
                int pp = this->comm_struct.num_neighbors_in;
                global_get[pp] = new IndexType[1 + count];
                (global_get[pp])[0] = 0;
                for (auto i = first_row[p]; i < first_row[p + 1]; i++) {
                    if (global_to_local[i] != 0) {
                        // global index
                        (global_get[pp])[1 + (global_get[pp])[0]] = i;
                        (global_get[pp])[0]++;
                    }
                }
                neighbors_in[pp] = p;
                this->comm_struct.num_neighbors_in++;

                num_recv += (global_get[pp])[0];
                recv[p] = 1;
            }
        }
    }

    std::vector<MPI_Request> send_req1(comm_size);
    std::vector<MPI_Request> send_req2(comm_size);
    std::vector<MPI_Request> recv_req1(comm_size);
    std::vector<MPI_Request> recv_req2(comm_size);
    int zero = 0;
    int pp = 0;
    std::vector<int> send(num_subdomains, 0);

    auto mpi_itype = boost::mpi::get_mpi_datatype(global_get[pp][0]);
    for (auto p = 0; p < num_subdomains; p++) {
        if (p != my_rank) {
            if (recv[p] != 0) {
                MPI_Isend((global_get[pp]), 1, mpi_itype, p, 1, MPI_COMM_WORLD,
                          &send_req1[p]);
                MPI_Isend((global_get[pp]), 1 + (global_get[pp])[0], mpi_itype,
                          p, 2, MPI_COMM_WORLD, &send_req2[p]);
                pp++;
            } else {
                MPI_Isend(&zero, 1, mpi_itype, p, 1, MPI_COMM_WORLD,
                          &send_req1[p]);
            }
            MPI_Irecv(&send[p], 1, mpi_itype, p, 1, MPI_COMM_WORLD,
                      &recv_req1[p]);
        }
    }
    this->comm_struct.num_neighbors_out = 0;
    auto neighbors_out = this->comm_struct.neighbors_out->get_data();
    auto global_put = this->comm_struct.global_put->get_data();
    mpi_itype = boost::mpi::get_mpi_datatype((global_put[pp])[0]);
    int pflag = 0;
    pp = 0;
    int num_send = 0;
    for (auto p = 0; p < num_subdomains; p++) {
        if (p != my_rank) {
            MPI_Status status;
            MPI_Wait(&recv_req1[p], &status);

            if (send[p] > 0) {
                neighbors_out[pp] = p;

                // global_put[pp] = std::make_shared<IndexType>(1 +
                // send[p]);
                global_put[pp] = new IndexType[1 + send[p]];
                // global_get[pp] = new IndexType[1 + count];
                (global_put[pp])[0] = send[p];

                MPI_Irecv((global_put[pp]), 1 + send[p], mpi_itype, p, 2,
                          MPI_COMM_WORLD, &recv_req2[p]);
                MPI_Wait(&recv_req2[p], &status);

                num_send += send[p];
                pp++;
            }
            MPI_Wait(&send_req1[p], &status);
            if (recv[p] != 0) {
                MPI_Wait(&send_req2[p], &status);
            }
        }
    }
    this->comm_struct.num_neighbors_out = pp;

    // allocate MPI buffer
    // one-sided
    if (settings.comm_settings.enable_onesided) {
        if (num_recv > 0) {
            this->comm_struct.recv_buffer = vec_vtype::create(
                settings.executor->get_master(), gko::dim<2>(num_recv, 1));

            MPI_Win_create(this->comm_struct.recv_buffer->get_values(),
                           num_recv * sizeof(ValueType), sizeof(ValueType),
                           MPI_INFO_NULL, MPI_COMM_WORLD,
                           &(this->comm_struct.window_buffer));
            this->comm_struct.windows_from = std::shared_ptr<vec_itype>(
                new vec_itype(settings.executor->get_master(),
                              this->comm_struct.num_neighbors_in),
                std::default_delete<vec_itype>());
            for (auto j = 0; j < this->comm_struct.num_neighbors_in; j++) {
                // j-th neighbor mapped to j-th window
                this->comm_struct.windows_from->get_data()[j] = j;
            }
        } else {
            this->comm_struct.recv_buffer = vec_vtype::create(
                settings.executor->get_master(), gko::dim<2>(1, 1));
        }
    }
    // two-sided
    else {
        if (num_recv > 0) {
            this->comm_struct.recv_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(num_recv, 1));
            // for gpu ? TODO : DONE
        } else {
            this->comm_struct.recv_buffer = nullptr;
        }

        this->comm_struct.put_request = std::shared_ptr<vec_request>(
            new vec_request(settings.executor->get_master(),
                            this->comm_struct.num_neighbors_out),
            std::default_delete<vec_request>());
        this->comm_struct.get_request = std::shared_ptr<vec_request>(
            new vec_request(settings.executor->get_master(),
                            this->comm_struct.num_neighbors_in),
            std::default_delete<vec_request>());
    }

    // one-sided
    if (settings.comm_settings.enable_onesided) {
        if (num_send > 0) {
            this->comm_struct.send_buffer = vec_vtype::create(
                settings.executor->get_master(), gko::dim<2>(num_send, 1));
            this->comm_struct.windows_to = std::shared_ptr<vec_itype>(
                new vec_itype(settings.executor->get_master(),
                              this->comm_struct.num_neighbors_out),
                std::default_delete<vec_itype>());
            for (auto j = 0; j < this->comm_struct.num_neighbors_out; j++) {
                this->comm_struct.windows_to->get_data()[j] =
                    j;  // j-th neighbor maped to j-th window
            }
        } else {
            this->comm_struct.send_buffer = vec_vtype::create(
                settings.executor->get_master(), gko::dim<2>(1, 1));
        }
    }
    // two-sided
    else {
        if (num_send > 0) {
            this->comm_struct.send_buffer =
                vec_vtype::create(settings.executor, gko::dim<2>(num_send, 1));
            // for gpu ? TODO : DONE
        } else {
            this->comm_struct.send_buffer = nullptr;
        }
    }
    this->comm_struct.local_put = this->comm_struct.global_put;
    this->comm_struct.local_get = this->comm_struct.global_get;
    this->comm_struct.remote_put = this->comm_struct.global_put;
    this->comm_struct.remote_get = this->comm_struct.global_get;
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::setup_windows(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer)
{
    using vec_itype = gko::Array<IndexType>;
    using vec_vtype = gko::matrix::Dense<ValueType>;
    auto num_subdomains = metadata.num_subdomains;
    auto local_size_o = metadata.local_size_o;
    auto neighbors_in = this->comm_struct.neighbors_in->get_data();
    auto global_get = this->comm_struct.global_get->get_data();

    // set displacement for the MPI buffer
    auto get_displacements = this->comm_struct.get_displacements->get_data();
    get_displacements[0] = 0;
    for (auto j = 0; j < this->comm_struct.num_neighbors_in; j++) {
        if ((global_get[j])[0] > 0) {
            int p = neighbors_in[j];
            get_displacements[p + 1] = (global_get[j])[0];
        }
    }
    for (auto j = 1; j <= num_subdomains; j++) {
        get_displacements[j] += get_displacements[j - 1];
    }

    auto put_displacements = this->comm_struct.put_displacements->get_data();
    auto mpi_itype = boost::mpi::get_mpi_datatype(get_displacements[0]);
    MPI_Alltoall(get_displacements, 1, mpi_itype, put_displacements, 1,
                 mpi_itype, MPI_COMM_WORLD);

    // setup windows
    if (settings.comm_settings.enable_onesided) {
        // Onesided
        // for gpu ? TODO (DONE) To improve: Win_create on a gpu buffer ?
        MPI_Win_create(main_buffer->get_values(),
                       main_buffer->get_size()[0] * sizeof(ValueType),
                       sizeof(ValueType), MPI_INFO_NULL, MPI_COMM_WORLD,
                       &(this->comm_struct.window_x));
    } else {
        // Twosided
    }


    if (settings.comm_settings.enable_onesided) {
        // MPI_Alloc_mem ? Custom allocator ?  TODO
        MPI_Win_create(this->local_residual_vector->get_values(),
                       (num_subdomains) * sizeof(ValueType), sizeof(ValueType),
                       MPI_INFO_NULL, MPI_COMM_WORLD,
                       &(this->window_residual_vector));
        std::vector<IndexType> zero_vec(num_subdomains, 0);
        gko::Array<IndexType> temp_array{settings.executor->get_master(),
                                         zero_vec.begin(), zero_vec.end()};
        this->convergence_vector = std::shared_ptr<vec_itype>(
            new vec_itype(settings.executor->get_master(), temp_array),
            std::default_delete<vec_itype>());
        this->convergence_sent = std::shared_ptr<vec_itype>(
            new vec_itype(settings.executor->get_master(), num_subdomains),
            std::default_delete<vec_itype>());
        this->convergence_local = std::shared_ptr<vec_itype>(
            new vec_itype(settings.executor->get_master(), num_subdomains),
            std::default_delete<vec_itype>());
        MPI_Win_create(this->convergence_vector->get_data(),
                       (num_subdomains) * sizeof(IndexType), sizeof(IndexType),
                       MPI_INFO_NULL, MPI_COMM_WORLD,
                       &(this->window_convergence));
    } else {
        // Twosided
    }

    if (settings.comm_settings.enable_onesided && num_subdomains > 1) {
        // Lock all windows.
        MPI_Win_lock_all(0, this->comm_struct.window_buffer);
        MPI_Win_lock_all(0, this->comm_struct.window_x);
        MPI_Win_lock_all(0, this->window_residual_vector);
        MPI_Win_lock_all(0, this->window_convergence);
    }
}


template <typename ValueType, typename IndexType>
void exchange_boundary_onesided(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using arr = gko::Array<IndexType>;
    // TODO: NOT SETUP FOR GPU, copies from local solution is needs to be
    // copied to gpu memory. (DONE)
    auto num_neighbors_out = comm_struct.num_neighbors_out;
    auto neighbors_out = comm_struct.neighbors_out->get_data();
    auto global_put = comm_struct.global_put->get_data();
    auto local_put = comm_struct.local_put->get_data();
    auto remote_put = comm_struct.remote_put->get_data();
    auto put_displacements = comm_struct.put_displacements->get_data();
    auto send_buffer = comm_struct.send_buffer->get_values();

    auto num_neighbors_in = comm_struct.num_neighbors_in;
    auto neighbors_in = comm_struct.neighbors_in->get_data();
    auto global_get = comm_struct.global_get->get_data();
    auto local_get = comm_struct.local_get->get_data();
    auto remote_get = comm_struct.remote_get->get_data();
    auto recv_buffer = comm_struct.recv_buffer->get_values();
    ValueType dummy = 1.0;
    auto mpi_vtype = boost::mpi::get_mpi_datatype(dummy);
    if (settings.comm_settings.enable_push) {
        if (settings.comm_settings.enable_push_one_by_one) {
            for (auto p = 0; p < num_neighbors_out; p++) {
                if ((global_put[p])[0] > 0) {
                    // push
                    for (auto i = 1; i <= (global_put[p])[0]; i++) {
                        MPI_Put(
                            &local_solution->get_values()[(local_put[p])[i]], 1,
                            mpi_vtype, neighbors_out[p], (remote_put[p])[i], 1,
                            mpi_vtype, comm_struct.window_x);
                    }
                    if (settings.comm_settings.enable_flush_all) {
                        MPI_Win_flush(neighbors_out[p], comm_struct.window_x);
                    } else if (settings.comm_settings.enable_flush_local) {
                        MPI_Win_flush_local(neighbors_out[p],
                                            comm_struct.window_x);
                    }
                }
            }
        } else {
            // accumulate
            int num_put = 0;
            for (auto p = 0; p < num_neighbors_out; p++) {
                // send
                if ((global_put[p])[0] > 0) {
                    // Evil. Change this. TODO
                    if (settings.executor_string == "cuda") {
                        // TODO: Optimize this for GPU with GPU buffers.
                        auto tmp_send_buf = vec_vtype::create(
                            settings.executor,
                            gko::dim<2>((global_put[p])[0], 1));
                        auto tmp_idx_s =
                            arr(settings.executor,
                                arr::view(settings.executor->get_master(),
                                          ((global_put)[p])[0],
                                          &((local_put[p])[1])));
                        settings.executor->run(
                            GatherScatter<ValueType, IndexType>(
                                true, (global_put[p])[0], tmp_idx_s.get_data(),
                                local_solution->get_values(),
                                tmp_send_buf->get_values()));
                        auto cpu_send_buf =
                            vec_vtype::create(settings.executor->get_master(),
                                              gko::dim<2>((global_put[p])[0]));
                        cpu_send_buf->copy_from(tmp_send_buf.get());
                        for (auto i = 1; i <= (global_put[p])[0]; i++) {
                            send_buffer[num_put + i - 1] =
                                cpu_send_buf->get_values()[i - 1];
                        }
                    } else {
                        for (auto i = 1; i <= (global_put[p])[0]; i++) {
                            send_buffer[num_put + i - 1] =
                                local_solution->get_values()[(local_put[p])[i]];
                        }
                    }
                    // use same window for all neighbors
                    MPI_Put(&send_buffer[num_put], (global_put[p])[0],
                            mpi_vtype, neighbors_out[p],
                            put_displacements[neighbors_out[p]],
                            (global_put[p])[0], mpi_vtype,
                            comm_struct.window_buffer);
                    if (settings.comm_settings.enable_flush_all) {
                        MPI_Win_flush(neighbors_out[p],
                                      comm_struct.window_buffer);
                    } else if (settings.comm_settings.enable_flush_local) {
                        MPI_Win_flush_local(neighbors_out[p],
                                            comm_struct.window_buffer);
                    }
                    num_put += (global_put[p])[0];
                }
            }
            // unpack receive buffer
            int num_get = 0;
            for (auto p = 0; p < num_neighbors_in; p++) {
                if ((global_get[p])[0] > 0) {
                    // Evil. Change this. TODO
                    if (settings.executor_string == "cuda") {
                        auto cpu_recv_buf =
                            vec_vtype::create(settings.executor->get_master(),
                                              gko::dim<2>((global_get[p])[0]));
                        for (auto i = 1; i <= (global_get[p])[0]; i++) {
                            cpu_recv_buf->get_values()[i - 1] =
                                recv_buffer[num_get + i - 1];
                        }
                        auto tmp_recv_buf = vec_vtype::create(
                            settings.executor,
                            gko::dim<2>((global_get[p])[0], 1));
                        tmp_recv_buf->copy_from(cpu_recv_buf.get());
                        auto tmp_idx_r =
                            arr(settings.executor,
                                arr::view(settings.executor->get_master(),
                                          ((global_get)[p])[0],
                                          &((local_get[p])[1])));
                        settings.executor->run(
                            GatherScatter<ValueType, IndexType>(
                                false, (global_get[p])[0], tmp_idx_r.get_data(),
                                tmp_recv_buf->get_values(),
                                local_solution->get_values()));
                    } else {
                        for (auto i = 1; i <= (global_get[p])[0]; i++) {
                            local_solution->get_values()[(local_get[p])[i]] =
                                recv_buffer[num_get + i - 1];
                        }
                    }
                    num_get += (global_get[p])[0];
                }
            }
        }
    } else {
        // TODO: needs to be updated and create window on the sender
        for (auto p = 0; p < num_neighbors_in; p++) {
            if ((global_put[p])[0] > 0) {
                // pull
                for (auto i = 1; i <= (global_get[p])[0]; i++) {
                    MPI_Get(&local_solution->get_values()[(local_get[p])[i]], 1,
                            mpi_vtype, neighbors_in[p], (remote_get[p])[i], 1,
                            mpi_vtype, comm_struct.window_x);
                }
                if (settings.comm_settings.enable_flush_all) {
                    MPI_Win_flush(neighbors_in[p], comm_struct.window_x);
                } else if (settings.comm_settings.enable_flush_local) {
                    MPI_Win_flush_local(neighbors_in[p], comm_struct.window_x);
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void exchange_boundary_twosided(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector)
{
    using vec = gko::matrix::Dense<ValueType>;
    MPI_Status status;
    auto num_neighbors_out = comm_struct.num_neighbors_out;
    auto neighbors_out = comm_struct.neighbors_out->get_data();
    auto global_put = comm_struct.global_put->get_data();
    auto local_put = comm_struct.local_put->get_data();
    auto put_request = comm_struct.put_request->get_data();
    int num_put = 0;
    ValueType dummy = 0.0;
    for (auto p = 0; p < num_neighbors_out; p++) {
        // send
        if ((global_put[p])[0] > 0) {
            if (settings.comm_settings.enable_overlap &&
                metadata.iter_count > 1) {
                // wait for the previous send
                auto p_r = put_request[p];
                MPI_Wait(&p_r, &status);
            }
            auto send_buffer = comm_struct.send_buffer->get_values();
            auto mpi_vtype = boost::mpi::get_mpi_datatype(dummy);
            // TODO: For GPU (DONE)
            settings.executor->run(GatherScatter<ValueType, IndexType>(
                true, (global_put[p])[0], &((local_put[p])[1]),
                solution_vector->get_values(), &send_buffer[num_put]));

            MPI_Isend(&send_buffer[num_put], (global_put[p])[0], mpi_vtype,
                      neighbors_out[p], 0, MPI_COMM_WORLD, &put_request[p]);
            num_put += (global_put[p])[0];
        }
    }

    int num_get = 0;
    auto get_request = comm_struct.get_request->get_data();
    auto num_neighbors_in = comm_struct.num_neighbors_in;
    auto neighbors_in = comm_struct.neighbors_in->get_data();
    auto global_get = comm_struct.global_get->get_data();
    auto local_get = comm_struct.local_get->get_data();
    if (!settings.comm_settings.enable_overlap || metadata.iter_count == 0) {
        for (auto p = 0; p < num_neighbors_in; p++) {
            // receive
            if ((global_get[p])[0] > 0) {
                auto recv_buffer = comm_struct.recv_buffer->get_values();
                auto mpi_vtype = boost::mpi::get_mpi_datatype(
                    recv_buffer[0]);  // works for GPU buffers ?
                MPI_Irecv(&recv_buffer[num_get], (global_get[p])[0], mpi_vtype,
                          neighbors_in[p], 0, MPI_COMM_WORLD, &get_request[p]);
                num_get += (global_get[p])[0];
            }
        }
    }

    num_get = 0;
    // wait for receive
    for (auto p = 0; p < num_neighbors_in; p++) {
        if ((global_get[p])[0] > 0) {
            // auto g_r = get_request[p];
            // MPI_Wait(&g_r, &status);
            auto recv_buffer = comm_struct.recv_buffer->get_values();
            auto mpi_vtype = boost::mpi::get_mpi_datatype(
                recv_buffer[0]);  // works for GPU buffers ?
            // TODO: GPU DONE
            settings.executor->run(GatherScatter<ValueType, IndexType>(
                false, (global_get[p])[0], &((local_get[p])[1]),
                &recv_buffer[num_get], solution_vector->get_values()));

            if (settings.comm_settings.enable_overlap) {
                // start the next receive
                MPI_Irecv(&recv_buffer[num_get], (global_get[p])[0], mpi_vtype,
                          neighbors_in[p], 0, MPI_COMM_WORLD, &get_request[p]);
            }
            num_get += (global_get[p])[0];
        }
    }

    // wait for send
    if (!settings.comm_settings.enable_overlap) {
        for (auto p = 0; p < num_neighbors_out; p++) {
            if ((global_put[p])[0] > 0) {
                auto p_r = put_request[p];
                MPI_Wait(&p_r, &status);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::exchange_boundary(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector)
{
    if (settings.comm_settings.enable_onesided) {
        exchange_boundary_onesided<ValueType, IndexType>(
            settings, metadata, this->comm_struct, solution_vector);
    } else {
        exchange_boundary_twosided<ValueType, IndexType>(
            settings, metadata, this->comm_struct, solution_vector);
    }
}


template <typename ValueType, typename IndexType>
void SolverRAS<ValueType, IndexType>::update_boundary(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &interface_matrix)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    auto one = gko::initialize<gko::matrix::Dense<ValueType>>(
        {1.0}, settings.executor);
    auto neg_one = gko::initialize<gko::matrix::Dense<ValueType>>(
        {-1.0}, settings.executor);
    auto local_size_x = metadata.local_size_x;
    local_solution->copy_from(local_rhs.get());
    global_old_solution->copy_from(solution_vector.get());
    if (metadata.num_subdomains > 1 && settings.overlap > 0) {
        auto temp_solution = vec_vtype::create(
            settings.executor, local_solution->get_size(),
            gko::Array<ValueType>::view(
                settings.executor, local_solution->get_size()[0],
                &(global_old_solution->get_values()[0])),
            1);
        interface_matrix->apply(neg_one.get(), temp_solution.get(), one.get(),
                                (local_solution).get());
    }
}


#define DECLARE_SOLVER_BASE(ValueType, IndexType) \
    class SolverBase<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SOLVER_BASE);
#undef DECLARE_SOLVER_BASE

#define DECLARE_SOLVER_RAS(ValueType, IndexType) \
    class SolverRAS<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SOLVER_RAS);
#undef DECLARE_SOLVER_RAS


}  // namespace SchwarzWrappers
