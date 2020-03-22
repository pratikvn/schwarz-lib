
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


#if SCHW_HAVE_CHOLMOD
#include <cholmod.h>
#endif

#include <exception_helpers.hpp>
#include <solve.hpp>
#include <utils.hpp>


namespace SchwarzWrappers {


template <typename ValueType, typename IndexType>
class Initialize;


template <typename ValueType>
double get_relative_error(const gko::matrix::Dense<ValueType> *first,
                          const gko::matrix::Dense<ValueType> *second)
{
    double diff = 0.0;
    double first_norm = 0.0;
    double second_norm = 0.0;
    for (gko::size_type row = 0; row < first->get_size()[0]; ++row) {
        for (gko::size_type col = 0; col < first->get_size()[1]; ++col) {
            const auto first_val = first->at(row, col);
            const auto second_val = second->at(row, col);
            diff += gko::squared_norm(first_val - second_val);
            first_norm += gko::squared_norm(first_val);
            second_norm += gko::squared_norm(second_val);
        }
    }
    if (first_norm == 0.0 && second_norm == 0.0) {
        first_norm = 1.0;
    }
    return std::sqrt(diff / std::max(first_norm, second_norm));
}


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::compute_local_factors(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_perm,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_inv_perm)
{
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using perm_type = gko::matrix::Permutation<IndexType>;
    auto temp_local_matrix = mtx::create(settings.executor->get_master());
    // Need to copy the matrix back to the CPU for the factorization.
    temp_local_matrix->copy_from(gko::lend(local_matrix));
    auto num_rows = temp_local_matrix->get_size()[0];
    auto num_nonzeros = temp_local_matrix->get_const_row_ptrs()[num_rows];

    const IndexType *row_ptrs = temp_local_matrix->get_const_row_ptrs();
    const IndexType *col_idxs = temp_local_matrix->get_const_col_idxs();
    const ValueType *lmat_values = temp_local_matrix->get_const_values();

    if (settings.factorization == "cholmod") {
#if SCHW_HAVE_CHOLMOD
        cholmod_start(&(cholmod.settings));
        cholmod.settings.final_ll = 1;
        cholmod.settings.supernodal = 0;
        cholmod.settings.final_super = 0;
        cholmod.settings.prefer_upper = 0;
        cholmod.settings.final_monotonic = 0;

        // Option to not re-order the matrix.
        if (settings.naturally_ordered_factor) {
            cholmod.settings.nmethods = 1;
            cholmod.settings.method[0].ordering = CHOLMOD_NATURAL;
        }
        auto sorted = 1;
        auto packed = 1;
        auto stype = 1;
        cholmod.system_matrix = cholmod_allocate_sparse(
            num_rows, num_rows, num_nonzeros, sorted, packed, stype,
            CHOLMOD_REAL, &(cholmod.settings));

        IndexType *col_ptrs =
            static_cast<IndexType *>(cholmod.system_matrix->p);
        IndexType *row_idxs =
            static_cast<IndexType *>(cholmod.system_matrix->i);
        ValueType *cmat_values =
            static_cast<ValueType *>(cholmod.system_matrix->x);

        int row_nnz = 0;
        col_ptrs[0] = row_nnz;
        for (auto i = 0; i < num_rows; ++i) {
            for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
                if (col_idxs[j] <= i) {
                    row_idxs[row_nnz] = col_idxs[j];
                    cmat_values[row_nnz] = lmat_values[j];
                    row_nnz++;
                }
            }
            col_ptrs[i + 1] = row_nnz;
        }

        cholmod.rhs =
            cholmod_ones(cholmod.system_matrix->nrow, 1,
                         cholmod.system_matrix->xtype, &(cholmod.settings));
        // analyze
        cholmod.L_factor =
            cholmod_analyze(cholmod.system_matrix, &(cholmod.settings));
        auto chol_perm = (IndexType *)(cholmod.L_factor->Perm);
        local_perm = perm_type::create(
            settings.executor, gko::dim<2>(num_rows),
            gko::Array<IndexType>(
                settings.executor, (IndexType *)(cholmod.L_factor->Perm),
                num_rows + (IndexType *)(cholmod.L_factor->Perm)),
            gko::matrix::row_permute);
        local_inv_perm = perm_type::create(
            settings.executor, gko::dim<2>(num_rows),
            gko::Array<IndexType>(
                settings.executor, (IndexType *)(cholmod.L_factor->Perm),
                num_rows + (IndexType *)(cholmod.L_factor->Perm)),
            gko::matrix::row_permute | gko::matrix::inverse_permute);


        // factor
        cholmod_factorize(cholmod.system_matrix, cholmod.L_factor,
                          &(cholmod.settings));
#endif
    } else if (settings.factorization == "umfpack") {
#if SCHW_HAVE_UMFPACK
        double *null = (double *)NULL;
        void *symbolic;
        umfpack_di_symbolic(num_rows, num_rows, (const int *)row_ptrs,
                            (const int *)col_idxs, (const double *)lmat_values,
                            &symbolic, null, null);
        umfpack_di_numeric((const int *)row_ptrs, (const int *)col_idxs,
                           (const double *)lmat_values, symbolic,
                           &umfpack.numeric, null, null);
        umfpack_di_free_symbolic(&symbolic);
    }
#endif

    if (settings.executor_string != "cuda") {
        if (settings.debug_print) {
            if (Utils<ValueType, IndexType>::assert_correct_permutation(
                    local_perm.get())) {
                std::cout << " Here " << __LINE__ << " Rank "
                          << metadata.my_rank << " Permutation is correct"
                          << std::endl;
            } else {
                std::cout << " Here " << __LINE__ << " Rank "
                          << metadata.my_rank << " Permutation is incorrect"
                          << std::endl;
            }
            if (Utils<ValueType, IndexType>::assert_correct_permutation(
                    local_inv_perm.get())) {
                std::cout << " Here " << __LINE__ << " Rank "
                          << metadata.my_rank
                          << " Inverse Permutation is correct" << std::endl;
            } else {
                std::cout << " Here " << __LINE__ << " Rank "
                          << metadata.my_rank
                          << " Inverse Permutation is incorrect" << std::endl;
            }
        }
        if (settings.write_perm_data) {
            std::ofstream file;
            std::string fname =
                "perm_" + std::to_string(metadata.my_rank) + ".csv";
            file.open(fname);
            for (auto i = 0; i < num_rows; ++i) {
                file << local_perm->get_permutation()[i] << "\n";
            }
            file << std::endl;
            file.close();
            fname = "inv_perm_" + std::to_string(metadata.my_rank) + ".csv";
            file.open(fname);
            for (auto i = 0; i < num_rows; ++i) {
                file << local_inv_perm->get_permutation()[i] << "\n";
            }
            file << std::endl;
            file.close();
        }
    }
}

template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::setup_local_solver(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &triangular_factor,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_perm,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_inv_perm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs)
{
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using perm_type = gko::matrix::Permutation<IndexType>;

    const auto solver_settings =
        settings.local_solver &
        (Settings::local_solver_settings::direct_solver_cholmod |
         Settings::local_solver_settings::direct_solver_umfpack |
         Settings::local_solver_settings::direct_solver_ginkgo |
         Settings::local_solver_settings::iterative_solver_dealii |
         Settings::local_solver_settings::iterative_solver_ginkgo);

    local_residual_vector = vec::create(settings.executor->get_master(),
                                        gko::dim<2>(metadata.max_iters + 1, 1));
    local_residual_vector_out =
        vec::create(settings.executor->get_master(),
                    gko::dim<2>(metadata.max_iters + 1, 1));
    global_residual_vector_out = vec::create(
        settings.executor->get_master(),
        gko::dim<2>(metadata.max_iters + 1, metadata.num_subdomains));

    // If direct solver is chosen, then we need to compute a factorization as a
    // first step. Only the triangular solves are done in the loop.
    if ((solver_settings ==
         Settings::local_solver_settings::direct_solver_cholmod) ||
        (solver_settings ==
         Settings::local_solver_settings::direct_solver_umfpack) ||
        (solver_settings ==
         Settings::local_solver_settings::direct_solver_ginkgo)) {
#if (!SCHW_HAVE_CHOLMOD && !SCHW_HAVE_UMFPACK)
        SCHWARZ_MODULE_NOT_IMPLEMENTED("cholmod and umfpack");
#endif

        // Factorize the matrix.
        compute_local_factors(settings, metadata, local_matrix, local_perm,
                              local_inv_perm);
        auto num_rows = local_matrix->get_size()[0];
        if (solver_settings ==
            Settings::local_solver_settings::direct_solver_cholmod) {
            auto factor_nnz =
                (static_cast<IndexType *>(cholmod.L_factor->p))[num_rows];
            std::cout << " Process " << metadata.my_rank << " has factor with "
                      << num_rows << " rows and " << factor_nnz << " non-zeros "
                      << std::endl;
            if (metadata.my_rank == 0) {
                std::cout << " Local direct solve with CHOLMOD" << std::endl;
            }
        } else if (solver_settings ==
                   Settings::local_solver_settings::direct_solver_umfpack) {
            if (metadata.my_rank == 0) {
                std::cout << " Local direct solve with UMFPACK" << std::endl;
            }
        }
        if (solver_settings ==
            Settings::local_solver_settings::direct_solver_ginkgo) {
            // Copy the triangular factor to the current executor.
            if (settings.factorization == "cholmod") {
#if (SCHW_HAVE_CHOLMOD)
                auto factor_nnz =
                    (static_cast<IndexType *>(cholmod.L_factor->p))[num_rows];
                if (metadata.my_rank == 0)
                    std::cout << " Local direct factorization with CHOLMOD"
                              << std::endl;
                triangular_factor = mtx::create(
                    settings.executor, gko::dim<2>(num_rows),
                    gko::Array<ValueType>(
                        (settings.executor->get_master()), factor_nnz,
                        (static_cast<ValueType *>(cholmod.L_factor->x))),
                    gko::Array<IndexType>(
                        (settings.executor->get_master()), factor_nnz,
                        (static_cast<IndexType *>(cholmod.L_factor->i))),
                    gko::Array<IndexType>(
                        (settings.executor->get_master()), num_rows + 1,
                        (static_cast<IndexType *>(cholmod.L_factor->p))));
#endif
            } else if (settings.factorization == "umfpack") {
#if (SCHW_HAVE_UMFPACK)
                if (metadata.my_rank == 0)
                    std::cout << " Local direct factorization with UMFPACK"
                              << std::endl;
                    // triangular_factor = mtx::create(
                    //     settings.executor, gko::dim<2>(num_rows),
                    //     gko::Array<ValueType>(
                    //         (settings.executor->get_master()), factor_nnz,
                    //         (static_cast<ValueType *>(cholmod.L_factor->x))),
                    //     gko::Array<IndexType>(
                    //         (settings.executor->get_master()), factor_nnz,
                    //         (static_cast<IndexType *>(cholmod.L_factor->i))),
                    //     gko::Array<IndexType>(
                    //         (settings.executor->get_master()), num_rows + 1,
                    //         (static_cast<IndexType
                    //         *>(cholmod.L_factor->p))));
#endif
            }

            if (metadata.my_rank == 0) {
                std::cout << " Local direct solve with Ginkgo TRS" << std::endl;
            }
            using u_trs = gko::solver::UpperTrs<ValueType, IndexType>;
            using l_trs = gko::solver::LowerTrs<ValueType, IndexType>;
            // Setup the Ginkgo triangular solver.
            this->U_solver = u_trs::build()
                                 .on(settings.executor)
                                 ->generate((triangular_factor));
            this->L_solver = l_trs::build()
                                 .on(settings.executor)
                                 ->generate(triangular_factor->transpose());

            if (settings.print_matrices && settings.executor_string != "cuda") {
                Utils<ValueType, IndexType>::print_matrix(
                    this->U_solver->get_system_matrix().get(), metadata.my_rank,
                    "U_mat");
                Utils<ValueType, IndexType>::print_matrix(
                    this->L_solver->get_system_matrix().get(), metadata.my_rank,
                    "L_mat");
            }
        }
    } else if (solver_settings ==
               Settings::local_solver_settings::iterative_solver_ginkgo) {
        if (metadata.my_rank == 0) {
            std::cout << " Local iterative solve with Ginkgo CG " << std::endl;
        }
        using cg = gko::solver::Cg<ValueType>;
        using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
        // Setup the Ginkgo iterative CG solver.
        if (settings.use_precond) {
            this->solver =
                cg::build()
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(local_matrix->get_size()[0])
                            .on(settings.executor),
                        gko::stop::ResidualNormReduction<ValueType>::build()
                            .with_reduction_factor(
                                metadata.local_solver_tolerance)
                            .on(settings.executor))
                    .with_preconditioner(
                        bj::build()
                            .with_max_block_size(
                                metadata.precond_max_block_size)
                            .on(settings.executor))
                    .on(settings.executor)
                    ->generate(local_matrix);
        } else {
            this->solver =
                cg::build()
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(local_matrix->get_size()[0])
                            .on(settings.executor),
                        gko::stop::ResidualNormReduction<ValueType>::build()
                            .with_reduction_factor(
                                metadata.local_solver_tolerance)
                            .on(settings.executor))
                    .on(settings.executor)
                    ->generate(local_matrix);
        }
    } else if (solver_settings ==
               Settings::local_solver_settings::iterative_solver_dealii) {
        SCHWARZ_NOT_IMPLEMENTED
    } else if (solver_settings ==
               Settings::local_solver_settings::solver_custom) {
        // Implement your own solver
        SCHWARZ_NOT_IMPLEMENTED
    } else {
        SCHWARZ_NOT_IMPLEMENTED
    }
}


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::local_solve(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &triangular_factor,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_perm,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_inv_perm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &init_guess,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution)
{
    const auto solver_settings =
        (Settings::local_solver_settings::direct_solver_cholmod |
         Settings::local_solver_settings::direct_solver_umfpack |
         Settings::local_solver_settings::direct_solver_ginkgo |
         Settings::local_solver_settings::iterative_solver_dealii |
         Settings::local_solver_settings::iterative_solver_ginkgo) &
        settings.local_solver;
    if (solver_settings ==
        Settings::local_solver_settings::direct_solver_cholmod) {
#if SCHW_HAVE_CHOLMOD
        SolverTools::solve_direct_cholmod(settings, metadata, cholmod.settings,
                                          cholmod.L_factor, cholmod.rhs,
                                          local_solution);
#endif
    } else if (solver_settings ==
               Settings::local_solver_settings::direct_solver_umfpack) {
#if SCHW_HAVE_UMFPACK
        double *null = (double *)NULL;
        auto temp_sol = gko::matrix::Dense<ValueType>::create(
            settings.executor, local_solution->get_size());
        temp_sol->copy_from(local_solution.get());
        umfpack_di_solve(
            UMFPACK_A, (const int *)local_matrix->get_const_row_ptrs(),
            (const int *)local_matrix->get_const_col_idxs(),
            (const double *)local_matrix->get_const_values(),
            (double *)local_solution->get_values(),
            (double *)temp_sol->get_values(), umfpack.numeric, null, null);
        // SolverTools::solve_direct_umfpack(settings, metadata, umfpack,
        //                                   local_solution);
#endif
    } else if (solver_settings ==
               Settings::local_solver_settings::direct_solver_ginkgo) {
        auto perm_sol = gko::matrix::Dense<ValueType>::create(
            settings.executor, local_solution->get_size());

        local_perm->apply(local_solution.get(), perm_sol.get());
        SolverTools::solve_direct_ginkgo(settings, metadata, this->L_solver,
                                         this->U_solver, perm_sol.get());
        // local_solution->copy_from(perm_sol.get());
        local_inv_perm->apply(perm_sol.get(), local_solution.get());

    } else if (solver_settings ==
               Settings::local_solver_settings::iterative_solver_ginkgo) {
        SolverTools::solve_iterative_ginkgo(settings, metadata, this->solver,
                                            local_solution, init_guess);
        local_solution->copy_from(init_guess.get());
    } else if (solver_settings ==
               Settings::local_solver_settings::iterative_solver_dealii) {
        SCHWARZ_NOT_IMPLEMENTED
    } else if (solver_settings ==
               Settings::local_solver_settings::solver_custom) {
        // Implement your own solver
        SCHWARZ_NOT_IMPLEMENTED
    } else {
        SCHWARZ_NOT_IMPLEMENTED
    }
}


template <typename ValueType, typename IndexType>
bool Solve<ValueType, IndexType>::check_local_convergence(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    ValueType &local_resnorm, ValueType &local_resnorm0)
{
    using vec = gko::matrix::Dense<ValueType>;
    bool locally_converged = false;
    local_resnorm = -1.0;
    auto tolerance = metadata.tolerance;
    auto one = gko::initialize<vec>({1.0}, settings.executor);
    auto neg_one = gko::initialize<vec>({-1.0}, settings.executor);

    // tol = 0.0 (compute local residual, but no global convergence check)
    // tol < 0.0 (no local residual computation)
    if (tolerance >= 0.0) {
        std::shared_ptr<vec> local_b = vec::create(
            settings.executor, gko::dim<2>(metadata.local_size_x, 1));
        std::shared_ptr<vec> local_x = vec::create(
            settings.executor, gko::dim<2>(metadata.local_size_x, 1));
        // extract local parts of b and x (with local matrix (interior+overlap),
        // but only with the interior part)
        local_b->copy_from(local_solution.get());
        SolverTools::extract_local_vector(
            settings, metadata, local_x, global_old_solution,
            metadata.first_row->get_data()[metadata.my_rank]);

        // SpMV with interior including overlap, b - A*x
        local_matrix->apply(neg_one.get(), gko::lend(local_x), one.get(),
                            gko::lend(local_b));

        // compute squared-sum of local residual (interior)
        auto temp = gko::initialize<vec>({0.0}, settings.executor);
        auto cpu_temp =
            gko::initialize<vec>({0.0}, settings.executor->get_master());
        local_b->compute_norm2(gko::lend(temp));
        cpu_temp->copy_from(gko::lend(temp));
        local_resnorm = cpu_temp->at(0);

        if (local_resnorm0 < 0.0) local_resnorm0 = local_resnorm;

        // locally_converged = (local_resnorm) / (local_resnorm0) < tolerance;
        locally_converged = (local_resnorm * local_resnorm) /
                                (local_resnorm0 * local_resnorm0) <
                            (tolerance * tolerance);
    }
    return locally_converged;
}


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::check_global_convergence(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    ValueType &local_resnorm, ValueType &local_resnorm0,
    ValueType &global_resnorm, ValueType &global_resnorm0,
    int &converged_all_local, int &num_converged_procs)
{
    auto num_subdomains = metadata.num_subdomains;
    auto my_rank = metadata.my_rank;
    auto iter = metadata.iter_count;
    auto tolerance = metadata.tolerance;
    auto l_res_vec = this->local_residual_vector->get_values();
    auto mpi_vtype = boost::mpi::get_mpi_datatype(l_res_vec[0]);

    if (settings.comm_settings.enable_onesided) {
        if (settings.convergence_settings.put_all_local_residual_norms) {
            ConvergenceTools::put_all_local_residual_norms(
                settings, metadata, local_resnorm, this->local_residual_vector,
                this->global_residual_vector_out, this->window_residual_vector);
        } else {
            ConvergenceTools::propagate_all_local_residual_norms(
                settings, metadata, comm_struct, local_resnorm,
                this->local_residual_vector, this->global_residual_vector_out,
                this->window_residual_vector);
        }
    }
    if (settings.convergence_settings.enable_global_check &&
        !settings.comm_settings.enable_onesided) {
        MPI_Allgather(&local_resnorm, 1, mpi_vtype, l_res_vec, 1, mpi_vtype,
                      MPI_COMM_WORLD);

        // compute the global residual norm by summing the local residual
        // norms
        global_resnorm = 0.0;
        for (auto j = 0; j < num_subdomains; j++) {
            global_residual_vector_out->at(iter, j) = l_res_vec[j];
            if (l_res_vec[j] != std::numeric_limits<ValueType>::max()) {
                global_resnorm += l_res_vec[j];
            } else {
                global_resnorm = -1.0;
                break;
            }
        }

        // check for the global convergence locally
        if (global_resnorm >= 0.0) {
            if ((global_resnorm0) < 0.0) global_resnorm0 = global_resnorm;
            if ((global_resnorm) / (global_resnorm0) <= tolerance)
                (converged_all_local)++;
        }
    } else if (settings.comm_settings.enable_onesided) {
        if (local_resnorm / local_resnorm0 <= tolerance)
            (converged_all_local)++;

        l_res_vec[my_rank] = std::min(l_res_vec[my_rank], local_resnorm);

        // save for post-processing
        for (auto j = 0; j < num_subdomains; j++) {
            global_residual_vector_out->at(iter, j) = l_res_vec[j];
        }
    }
    // check for global convergence (count how many processes have detected the
    // global convergence)
    if (settings.comm_settings.enable_onesided) {
        if (settings.convergence_settings.enable_global_simple_tree) {
            ConvergenceTools::global_convergence_check_onesided_tree(
                settings, metadata, convergence_vector, converged_all_local,
                num_converged_procs, window_convergence);
        } else if (settings.convergence_settings
                       .enable_decentralized_leader_election) {
            ConvergenceTools::global_convergence_decentralized(
                settings, metadata, comm_struct, convergence_vector,
                convergence_sent, convergence_local, converged_all_local,
                num_converged_procs, window_convergence);
        } else {
            std::cout << "Global Convergence check type unspecified"
                      << std::endl;
            std::exit(-1);
        }
    } else {  // two-sided MPI
        if (settings.convergence_settings.enable_global_check) {
            if (converged_all_local == 1) {
                num_converged_procs = num_subdomains;
            }
        } else {
            int local_converged = (converged_all_local == 0 ? 0 : 1);
            MPI_Allreduce(&local_converged, &num_converged_procs, 1, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD);
        }
    }
}  // namespace SchwarzWrappers


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::check_convergence(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    ValueType &local_residual_norm, ValueType &local_residual_norm0,
    ValueType &global_residual_norm, ValueType &global_residual_norm0,
    int &num_converged_procs)
{
    int num_converged_p = 0;
    auto tolerance = metadata.tolerance;
    auto iter = metadata.iter_count;
    if (check_local_convergence(settings, metadata, local_solution,
                                global_old_solution, local_matrix,
                                local_residual_norm, local_residual_norm0)) {
        num_converged_p = 1;
    } else {
        num_converged_p = 0;
    }
    local_residual_vector_out->at(iter) = local_residual_norm;
    metadata.current_residual_norm = local_residual_norm;
    metadata.min_residual_norm =
        (iter == 0 ? local_residual_norm
                   : std::min(local_residual_norm, metadata.min_residual_norm));

    auto iter_cond =
        settings.convergence_settings.enable_global_check_iter_offset
            ? ((iter > (metadata.max_iters * 0.05)) ||
               metadata.max_iters < 1000)
            : true;
    if (tolerance > 0.0 && iter_cond) {
        int converged_all_local = 0;
        check_global_convergence(
            settings, metadata, comm_struct, convergence_vector,
            local_residual_norm, local_residual_norm0, global_residual_norm,
            global_residual_norm0, converged_all_local, num_converged_p);
        num_converged_procs = num_converged_p;
    }
}


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::update_residual(
    const Settings &settings,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto one = gko::initialize<vec>({1.0}, settings.executor);
    auto neg_one = gko::initialize<vec>({-1.0}, settings.executor);

    local_matrix->apply(neg_one.get(), gko::lend(global_old_solution),
                        one.get(), gko::lend(solution_vector));
}


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::compute_residual_norm(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &global_matrix,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
    ValueType &mat_norm, ValueType &rhs_norm, ValueType &sol_norm,
    ValueType &residual_norm)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto global_sol = vec::create(settings.executor->get_master(),
                                  gko::dim<2>(metadata.global_size, 1));
    auto local_sol = vec::create(settings.executor->get_master(),
                                 gko::dim<2>(metadata.global_size, 1));
    auto first_row = metadata.first_row->get_data()[metadata.my_rank];

    for (auto i = 0; i < metadata.global_size; ++i) {
        local_sol->get_values()[i] = 0.0;
    }
    auto temp_vector = vec::create(
        settings.executor->get_master(), gko::dim<2>(metadata.local_size, 1),
        (gko::Array<ValueType>::view(settings.executor->get_master(),
                                     metadata.local_size,
                                     &local_sol->get_values()[first_row])),
        1);
    auto temp_vector2 =
        vec::create(settings.executor, gko::dim<2>(metadata.local_size, 1),
                    (gko::Array<ValueType>::view(
                        settings.executor, metadata.local_size,
                        &solution_vector->get_values()[first_row])),
                    1);
    temp_vector->copy_from(temp_vector2.get());
    auto mpi_vtype = boost::mpi::get_mpi_datatype(local_sol->get_values()[0]);
    auto rnorm = gko::initialize<vec>({0.0}, settings.executor);
    auto rhsnorm = gko::initialize<vec>({0.0}, settings.executor);
    auto xnorm = gko::initialize<vec>({0.0}, settings.executor);
    auto cpu_resnorm =
        gko::initialize<vec>({0.0}, settings.executor->get_master());
    auto cpu_rhsnorm =
        gko::initialize<vec>({0.0}, settings.executor->get_master());
    auto cpu_solnorm =
        gko::initialize<vec>({0.0}, settings.executor->get_master());
    MPI_Allreduce(local_sol->get_values(), global_sol->get_values(),
                  metadata.global_size, mpi_vtype, MPI_SUM, MPI_COMM_WORLD);
    global_rhs->compute_norm2(rhsnorm.get());
    cpu_rhsnorm->copy_from(gko::lend(rhsnorm));
    rhs_norm = cpu_rhsnorm->at(0);
    global_sol->compute_norm2(xnorm.get());
    cpu_solnorm->copy_from(gko::lend(xnorm));
    sol_norm = cpu_solnorm->at(0);
    solution_vector->copy_from(global_sol.get());
    auto one = gko::initialize<vec>({1.0}, settings.executor);
    auto neg_one = gko::initialize<vec>({-1.0}, settings.executor);

    global_matrix->apply(neg_one.get(), gko::lend(solution_vector), one.get(),
                         gko::lend(global_rhs));

    global_rhs->compute_norm2(rnorm.get());
    cpu_resnorm->copy_from(gko::lend(rnorm));
    residual_norm = cpu_resnorm->at(0);
}


template <typename ValueType, typename IndexType>
void Solve<ValueType, IndexType>::clear(Settings &settings)
{
    if (settings.comm_settings.enable_onesided) {
        MPI_Win_unlock_all(window_convergence);
        MPI_Win_unlock_all(window_residual_vector);

        MPI_Win_free(&window_residual_vector);
    }

    if (settings.local_solver ==
        Settings::local_solver_settings::direct_solver_cholmod) {
#if SCHW_HAVE_CHOLMOD
        cholmod_finish(&cholmod.settings);
#endif
    }
}


#define DECLARE_FUNCTION(ValueType)                                  \
    double get_relative_error(const gko::matrix::Dense<ValueType> *, \
                              const gko::matrix::Dense<ValueType> *) \
        INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_SOLVER(ValueType, IndexType) class Solve<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SOLVER);
#undef DECLARE_SOLVER


}  // namespace SchwarzWrappers
