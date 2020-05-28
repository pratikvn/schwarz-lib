
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


namespace schwz {


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


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::compute_local_factors(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix)
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

        // factor
        cholmod_factorize(cholmod.system_matrix, cholmod.L_factor,
                          &(cholmod.settings));
#endif
    } else if (settings.factorization == "umfpack") {
#if SCHW_HAVE_UMFPACK
        std::vector<IndexType> col_ptrs(num_rows + 1, 0);
        std::vector<IndexType> row_idxs(local_matrix->get_num_stored_elements(),
                                        0);
        std::vector<ValueType> umat_values(
            local_matrix->get_num_stored_elements(), 0);

        int row_nnz = 0;
        col_ptrs[0] = row_nnz;
        for (auto i = 0; i < num_rows; ++i) {
            for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
                row_idxs[row_nnz] = col_idxs[j];
                umat_values[row_nnz] = lmat_values[j];
                row_nnz++;
            }
            col_ptrs[i + 1] = row_nnz;
        }
        void *symbolic;
        SCHWARZ_ASSERT_NO_UMFPACK_ERRORS(umfpack_di_symbolic(
            num_rows, num_rows, (const int *)row_ptrs, (const int *)col_idxs,
            (const double *)lmat_values, &symbolic, umfpack.control,
            umfpack.info));
        SCHWARZ_ASSERT_NO_UMFPACK_ERRORS(umfpack_di_numeric(
            (const int *)row_ptrs, (const int *)col_idxs,
            (const double *)lmat_values, symbolic, &umfpack.numeric,
            umfpack.control, umfpack.info));
        umfpack_di_free_symbolic(&symbolic);
#endif
    }
}


template <typename ValueType, typename IndexType>
void update_diagonals(
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &matrix,
    std::vector<ValueType> &diags)
{
    auto num_rows = matrix->get_size()[0];
    auto row_ptrs = matrix->get_row_ptrs();
    auto col_idxs = matrix->get_col_idxs();
    auto values = matrix->get_values();
    for (auto i = 0; i < num_rows; ++i) {
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            if (i == col_idxs[j]) {
                values[j] = diags.data()[i];
            }
        }
    }
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::setup_local_solver(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &triangular_factor_l,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &triangular_factor_u,
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
    metadata.post_process_data.global_residual_vector_out =
        std::vector<std::vector<ValueType>>(metadata.num_subdomains);

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
        compute_local_factors(settings, metadata, local_matrix);
        auto num_rows = local_matrix->get_size()[0];
        if (settings.factorization == "cholmod") {
#if (SCHW_HAVE_CHOLMOD)
            if (metadata.my_rank == 0)
                std::cout << " Local direct factorization with CHOLMOD"
                          << std::endl;
            auto factor_nnz =
                (static_cast<IndexType *>(cholmod.L_factor->p))[num_rows];
            std::cout << " Process " << metadata.my_rank << " has factor with "
                      << num_rows << " rows and " << factor_nnz << " non-zeros "
                      << std::endl;
#else
            SCHWARZ_MODULE_NOT_IMPLEMENTED("cholmod");
#endif
        } else if (settings.factorization == "umfpack") {
#if (SCHW_HAVE_UMFPACK)
            if (metadata.my_rank == 0)
                std::cout << " Local direct factorization with UMFPACK"
                          << std::endl;
            SCHWARZ_ASSERT_NO_UMFPACK_ERRORS(umfpack_di_get_lunz(
                &umfpack.factor_l_nnz, &umfpack.factor_u_nnz, &umfpack.n_row,
                &umfpack.n_col, &umfpack.nz_udiag, umfpack.numeric));
            std::cout << " Process " << metadata.my_rank << " has factors "
                      << " L(rows, nnz): (" << umfpack.n_row << ","
                      << umfpack.factor_l_nnz << ") "
                      << ";  U(cols, nnz): (" << umfpack.n_col << ","
                      << umfpack.factor_u_nnz << ") " << std::endl;
#else
            SCHWARZ_MODULE_NOT_IMPLEMENTED("umfpack");
#endif
        }
        if (solver_settings ==
            Settings::local_solver_settings::direct_solver_cholmod) {
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
                triangular_factor_u = mtx::create(
                    settings.executor->get_master(), gko::dim<2>(num_rows),
                    gko::Array<ValueType>(
                        (settings.executor->get_master()), factor_nnz,
                        (static_cast<ValueType *>(cholmod.L_factor->x))),
                    gko::Array<IndexType>(
                        (settings.executor->get_master()), factor_nnz,
                        (static_cast<IndexType *>(cholmod.L_factor->i))),
                    gko::Array<IndexType>(
                        (settings.executor->get_master()), num_rows + 1,
                        (static_cast<IndexType *>(cholmod.L_factor->p))));

                triangular_factor_l =
                    mtx::create(settings.executor->get_master(),
                                gko::dim<2>(num_rows), factor_nnz);
                triangular_factor_l->copy_from(
                    triangular_factor_u->transpose());

                local_perm = perm_type::create(
                    settings.executor, gko::dim<2>(num_rows),
                    gko::Array<IndexType>(
                        settings.executor,
                        (IndexType *)(cholmod.L_factor->Perm),
                        num_rows + (IndexType *)(cholmod.L_factor->Perm)),
                    gko::matrix::row_permute);
                local_inv_perm = perm_type::create(
                    settings.executor, gko::dim<2>(num_rows),
                    gko::Array<IndexType>(
                        settings.executor,
                        (IndexType *)(cholmod.L_factor->Perm),
                        num_rows + (IndexType *)(cholmod.L_factor->Perm)),
                    gko::matrix::row_permute | gko::matrix::inverse_permute);
#endif
            } else if (settings.factorization == "umfpack") {
#if (SCHW_HAVE_UMFPACK)
                std::vector<IndexType> Lp(umfpack.n_row + 1, 0);
                std::vector<IndexType> Li(umfpack.factor_l_nnz, 0);
                std::vector<ValueType> Lx(umfpack.factor_l_nnz, 0);
                std::vector<IndexType> Up(umfpack.n_col + 1, 0);
                std::vector<ValueType> diag(umfpack.n_col, 0);
                std::vector<IndexType> Ui(umfpack.factor_u_nnz, 0);
                std::vector<ValueType> Ux(umfpack.factor_u_nnz, 0);
                std::vector<IndexType> umf_row_perm(umfpack.n_row, 0);
                std::vector<IndexType> umf_col_perm(umfpack.n_col, 0);
                umfpack.row_scale = gko::matrix::Dense<ValueType>::create(
                    settings.executor->get_master(),
                    gko::dim<2>(1, umfpack.n_row));
                SCHWARZ_ASSERT_NO_UMFPACK_ERRORS(umfpack_di_get_numeric(
                    (int *)Lp.data(), (int *)Li.data(), (double *)Lx.data(),
                    (int *)Up.data(), (int *)Ui.data(), (double *)Ux.data(),
                    (int *)umf_row_perm.data(), (int *)umf_col_perm.data(),
                    (double *)diag.data(), &umfpack.do_reciproc,
                    (double *)umfpack.row_scale->get_values(),
                    umfpack.numeric));

                local_perm = perm_type::create(
                    settings.executor, gko::dim<2>(num_rows),
                    gko::Array<IndexType>(settings.executor,
                                          (umf_row_perm.data()),
                                          num_rows + (umf_row_perm.data())),
                    gko::matrix::row_permute);
                local_inv_perm = perm_type::create(
                    settings.executor, gko::dim<2>(num_rows),
                    gko::Array<IndexType>(settings.executor,
                                          (umf_col_perm.data()),
                                          num_rows + (umf_col_perm.data())),
                    gko::matrix::row_permute | gko::matrix::inverse_permute);

                // The matrices can be on the CPU because the solver creates the
                // matrix on the executor anyway.
                triangular_factor_l = mtx::create(
                    settings.executor->get_master(), gko::dim<2>(umfpack.n_row),
                    gko::Array<ValueType>(settings.executor->get_master(),
                                          Lx.data(),
                                          umfpack.factor_l_nnz + Lx.data()),
                    gko::Array<IndexType>(settings.executor->get_master(),
                                          Li.data(),
                                          umfpack.factor_l_nnz + Li.data()),
                    gko::Array<IndexType>((settings.executor->get_master()),
                                          Lp.data(),
                                          Lp.data() + umfpack.n_row + 1));

                auto temp_u = mtx::create(
                    settings.executor->get_master(), gko::dim<2>(umfpack.n_row),
                    gko::Array<ValueType>(settings.executor->get_master(),
                                          Ux.data(),
                                          umfpack.factor_u_nnz + Ux.data()),
                    gko::Array<IndexType>(settings.executor->get_master(),
                                          Ui.data(),
                                          umfpack.factor_u_nnz + Ui.data()),
                    gko::Array<IndexType>(settings.executor->get_master(),
                                          Up.data(),
                                          Up.data() + umfpack.n_row + 1));
                triangular_factor_u =
                    mtx::create(settings.executor->get_master(),
                                gko::dim<2>(num_rows), umfpack.factor_u_nnz);
                triangular_factor_u->copy_from(temp_u->transpose());
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
                                 ->generate(triangular_factor_u);
            this->L_solver = l_trs::build()
                                 .on(settings.executor)
                                 ->generate(triangular_factor_l);

            if (settings.print_matrices && settings.executor_string != "cuda") {
                Utils<ValueType, IndexType>::print_matrix(
                    this->U_solver->get_system_matrix().get(), metadata.my_rank,
                    "U_mat");
                Utils<ValueType, IndexType>::print_matrix(
                    this->L_solver->get_system_matrix().get(), metadata.my_rank,
                    "L_mat");
            }

            if (settings.executor_string != "cuda") {
                if (settings.debug_print) {
                    if (Utils<ValueType, IndexType>::assert_correct_permutation(
                            local_perm.get())) {
                        std::cout << " Here " << __LINE__ << " Rank "
                                  << metadata.my_rank
                                  << " Permutation is correct" << std::endl;
                    } else {
                        std::cout << " Here " << __LINE__ << " Rank "
                                  << metadata.my_rank
                                  << " Permutation is incorrect" << std::endl;
                    }
                    if (Utils<ValueType, IndexType>::assert_correct_permutation(
                            local_inv_perm.get())) {
                        std::cout << " Here " << __LINE__ << " Rank "
                                  << metadata.my_rank
                                  << " Inverse Permutation is correct"
                                  << std::endl;
                    } else {
                        std::cout << " Here " << __LINE__ << " Rank "
                                  << metadata.my_rank
                                  << " Inverse Permutation is incorrect"
                                  << std::endl;
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
                    fname =
                        "inv_perm_" + std::to_string(metadata.my_rank) + ".csv";
                    file.open(fname);
                    for (auto i = 0; i < num_rows; ++i) {
                        file << local_inv_perm->get_permutation()[i] << "\n";
                    }
                    file << std::endl;
                    file.close();
                }
            }
        }
    } else if (solver_settings ==
               Settings::local_solver_settings::iterative_solver_ginkgo) {
        int l_max_iters = 0;
        if (metadata.local_max_iters == -1) {
            l_max_iters = local_matrix->get_size()[0];
        } else {
            l_max_iters = metadata.local_max_iters;
        }
        if (metadata.my_rank == 0) {
            std::cout << " Local max iters " << l_max_iters
                      << " with restart iter " << settings.restart_iter
                      << std::endl;
        }
        this->combined_criterion =
            gko::stop::Combined::build()
                .with_criteria(
                    gko::stop::Iteration::build()
                        .with_max_iters(l_max_iters)
                        .on(settings.executor),
                    gko::stop::ResidualNormReduction<ValueType>::build()
                        .with_reduction_factor(metadata.local_solver_tolerance)
                        .on(settings.executor))
                .on(settings.executor);
        if (settings.enable_logging) {
            this->record_logger = gko::log::Record::create(
                settings.executor, settings.executor->get_mem_space(),
                gko::log::Logger::iteration_complete_mask |
                    gko::log::Logger::criterion_check_completed_mask);
            this->combined_criterion->add_logger(this->record_logger);
        }
        if (settings.non_symmetric_matrix) {
            using solver = gko::solver::Gmres<ValueType>;
            using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
            // Setup the Ginkgo iterative GMRES solver.
            if (metadata.local_precond == "block-jacobi") {
                if (metadata.my_rank == 0) {
                    std::cout << " Local Ginkgo iterative solve(GMRES) with "
                                 "Block-Jacobi preconditioning "
                              << std::endl;
                }
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_krylov_dim(settings.restart_iter)
                                   .with_preconditioner(
                                       bj::build()
                                           .with_max_block_size(
                                               metadata.precond_max_block_size)
                                           .on(settings.executor))
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else if (metadata.local_precond == "ilu") {
                if (metadata.my_rank == 0) {
                    std::cout
                        << " Local Ginkgo iterative solve(GMRES) with ParILU "
                           "preconditioning "
                        << std::endl;
                }
                auto exec = settings.executor;
                auto par_ilu_fact =
                    gko::factorization::ParIlu<ValueType, IndexType>::build()
                        .on(exec);
                auto par_ilu = par_ilu_fact->generate(local_matrix);
                auto ilu_pre_factory =
                    gko::preconditioner::Ilu<
                        gko::solver::LowerTrs<ValueType, IndexType>,
                        gko::solver::UpperTrs<ValueType, IndexType>,
                        false>::build()
                        .on(exec);
                auto ilu_preconditioner =
                    ilu_pre_factory->generate(gko::share(par_ilu));
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_krylov_dim(settings.restart_iter)
                                   .with_generated_preconditioner(
                                       gko::share(ilu_preconditioner))
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else if (metadata.local_precond == "isai") {
                if (metadata.my_rank == 0) {
                    std::cout
                        << " Local Ginkgo iterative solve(GMRES) with ISAI"
                           "preconditioning "
                        << std::endl;
                }
                auto exec = settings.executor;
                using LowerIsai =
                    gko::preconditioner::LowerIsai<ValueType, IndexType>;
                using UpperIsai =
                    gko::preconditioner::UpperIsai<ValueType, IndexType>;
                auto ilu_pre_factory =
                    gko::preconditioner::Ilu<LowerIsai, UpperIsai>::build().on(
                        exec);
                auto ilu_preconditioner =
                    ilu_pre_factory->generate(gko::share(local_matrix));
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_krylov_dim(settings.restart_iter)
                                   .with_generated_preconditioner(
                                       gko::share(ilu_preconditioner))
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else if (metadata.local_precond == "null") {
                if (metadata.my_rank == 0) {
                    std::cout << " Local Ginkgo iterative solve(GMRES) with no "
                                 "preconditioning "
                              << std::endl;
                }
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_krylov_dim(settings.restart_iter)
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else {
                std::cerr << "Unsupported preconditioner." << std::endl;
            }
        } else {
            using solver = gko::solver::Cg<ValueType>;
            using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
            // Setup the Ginkgo iterative CG solver.
            if (metadata.local_precond == "block-jacobi") {
                if (metadata.my_rank == 0) {
                    std::cout << " Local Ginkgo iterative solve(CG) with "
                                 "Block-Jacobi preconditioning "
                              << std::endl;
                }
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_preconditioner(
                                       bj::build()
                                           .with_max_block_size(
                                               metadata.precond_max_block_size)
                                           .on(settings.executor))
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else if (metadata.local_precond == "ilu") {
                if (metadata.my_rank == 0) {
                    std::cout
                        << " Local Ginkgo iterative solve(CG) with ParILU "
                           "preconditioning "
                        << std::endl;
                }
                auto exec = settings.executor;
                auto par_ilu_fact =
                    gko::factorization::ParIlu<ValueType, IndexType>::build()
                        .on(exec);
                auto par_ilu = par_ilu_fact->generate(local_matrix);
                auto ilu_pre_factory =
                    gko::preconditioner::Ilu<
                        gko::solver::LowerTrs<ValueType, IndexType>,
                        gko::solver::UpperTrs<ValueType, IndexType>,
                        false>::build()
                        .on(exec);
                auto ilu_preconditioner =
                    ilu_pre_factory->generate(gko::share(par_ilu));
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_generated_preconditioner(
                                       gko::share(ilu_preconditioner))
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else if (metadata.local_precond == "isai") {
                if (metadata.my_rank == 0) {
                    std::cout << " Local Ginkgo iterative solve(CG) with ISAI"
                                 "preconditioning "
                              << std::endl;
                }
                auto exec = settings.executor;
                using LowerIsai =
                    gko::preconditioner::LowerIsai<ValueType, IndexType>;
                using UpperIsai =
                    gko::preconditioner::UpperIsai<ValueType, IndexType>;
                auto ilu_pre_factory =
                    gko::preconditioner::Ilu<LowerIsai, UpperIsai, false,
                                             IndexType>::build()
                        .on(exec);
                auto ilu_preconditioner =
                    ilu_pre_factory->generate(gko::share(local_matrix));
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .with_generated_preconditioner(
                                       gko::share(ilu_preconditioner))
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else if (metadata.local_precond == "null") {
                if (metadata.my_rank == 0) {
                    std::cout << " Local Ginkgo iterative solve(CG) with no "
                                 "preconditioning "
                              << std::endl;
                }
                this->solver = solver::build()
                                   .with_criteria(this->combined_criterion)
                                   .on(settings.executor)
                                   ->generate(local_matrix);
            } else {
                std::cerr << "Unsupported preconditioner." << std::endl;
            }
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


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::local_solve(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &triangular_factor_l,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &triangular_factor_u,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_perm,
    std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_inv_perm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &init_guess,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
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
        auto temp_sol = gko::matrix::Dense<ValueType>::create(
            settings.executor, local_solution->get_size());
        temp_sol->copy_from(local_solution.get());
        SCHWARZ_ASSERT_NO_UMFPACK_ERRORS(umfpack_di_solve(
            UMFPACK_A, (const int *)local_matrix->get_const_row_ptrs(),
            (const int *)local_matrix->get_const_col_idxs(),
            (const double *)local_matrix->get_const_values(),
            (double *)local_solution->get_values(),
            (const double *)temp_sol->get_const_values(), umfpack.numeric,
            umfpack.control, umfpack.info));
#endif
    } else if (solver_settings ==
               Settings::local_solver_settings::direct_solver_ginkgo) {
        auto vec_size = local_solution->get_size()[0];
        std::shared_ptr<vec> perm_sol =
            vec::create(settings.executor, gko::dim<2>(vec_size, 1),
                        gko::Array<ValueType>::view(settings.executor, vec_size,
                                                    work_vector->get_values()),
                        1);
        local_perm->apply(local_solution.get(), perm_sol.get());
        SolverTools::solve_direct_ginkgo(settings, metadata, this->L_solver,
                                         this->U_solver, work_vector, perm_sol);
        local_inv_perm->apply(perm_sol.get(), local_solution.get());
    } else if (solver_settings ==
               Settings::local_solver_settings::iterative_solver_ginkgo) {
        int new_max_iters = 0;
        if (metadata.updated_max_iters == -1) {
            new_max_iters = local_matrix->get_size()[0];
        } else {
            new_max_iters = metadata.updated_max_iters;
        }
        if (settings.reset_local_crit_iter != -1 &&
            metadata.iter_count > settings.reset_local_crit_iter) {
            this->combined_criterion =
                gko::stop::Combined::build()
                    .with_criteria(
                        gko::stop::Iteration::build()
                            .with_max_iters(new_max_iters)
                            .on(settings.executor),
                        gko::stop::ResidualNormReduction<ValueType>::build()
                            .with_reduction_factor(
                                metadata.local_solver_tolerance)
                            .on(settings.executor))
                    .on(settings.executor);
        }
        if (settings.enable_logging) {
            this->combined_criterion->add_logger(this->record_logger);
        }
        if (settings.non_symmetric_matrix) {
            gko::as<gko::solver::Gmres<ValueType>>(this->solver.get())
                ->set_stop_criterion_factory(this->combined_criterion);
        } else {
            gko::as<gko::solver::Cg<ValueType>>(this->solver.get())
                ->set_stop_criterion_factory(this->combined_criterion);
        }
        SolverTools::solve_iterative_ginkgo(settings, metadata, this->solver,
                                            local_solution, init_guess);
        if (settings.enable_logging) {
            auto res_exec = this->record_logger->get()
                                .criterion_check_completed.back()
                                ->residual.get();
            auto res_vec = gko::as<gko::matrix::Dense<ValueType>>(res_exec);
            auto rnorm_d = vec::create(settings.executor, gko::dim<2>(1, 1));
            res_vec->compute_norm2(rnorm_d.get());
            auto rnorm =
                vec::create(settings.executor->get_master(), gko::dim<2>(1, 1));
            rnorm->copy_from(rnorm_d.get());
            auto conv_iter_count = this->record_logger->get()
                                       .criterion_check_completed.back()
                                       ->num_iterations;
            metadata.post_process_data.local_converged_iter_count.push_back(
                static_cast<IndexType>(conv_iter_count));
            metadata.post_process_data.local_converged_resnorm.push_back(
                rnorm->at(0));
            metadata.post_process_data.local_timestamp.push_back(
                MPI_Wtime() - metadata.init_mpi_wtime);

        } else {
            metadata.post_process_data.local_converged_iter_count.push_back(
                static_cast<IndexType>(0));
            metadata.post_process_data.local_converged_resnorm.push_back(0.0);
        }
        metadata.post_process_data.local_timestamp.push_back(
            MPI_Wtime() - metadata.init_mpi_wtime);
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


template <typename ValueType, typename IndexType, typename MixedValueType>
bool Solve<ValueType, IndexType, MixedValueType>::check_local_convergence(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
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
        auto local_b = vec::create(settings.executor,
                                   gko::dim<2>(metadata.local_size_x, 1),
                                   gko::Array<ValueType>::view(
                                       settings.executor, metadata.local_size_x,
                                       work_vector->get_values()),
                                   1);
        auto local_x = vec::create(
            settings.executor, gko::dim<2>(metadata.local_size_x, 1),
            gko::Array<ValueType>::view(
                settings.executor, metadata.local_size_x,
                work_vector->get_values() + metadata.local_size_x),
            1);
        // extract local parts of b and x (with local matrix (interior+overlap),
        // but only with the interior part)
        local_b->copy_from(local_solution.get());
        SolverTools::extract_local_vector(
            settings, metadata, local_x.get(), global_solution.get(),
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


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::check_global_convergence(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
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
            conv_tools::put_all_local_residual_norms(
                settings, metadata, local_resnorm, this->local_residual_vector,
                this->window_residual_vector);
        } else {
            conv_tools::propagate_all_local_residual_norms<ValueType, IndexType,
                                                           MixedValueType>(
                settings, metadata, comm_struct, local_resnorm,
                this->local_residual_vector, this->window_residual_vector);
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
            (metadata.post_process_data.global_residual_vector_out[j])
                .push_back(l_res_vec[j]);
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
            (metadata.post_process_data.global_residual_vector_out[j])
                .push_back(l_res_vec[j]);
        }
    }
    // check for global convergence (count how many processes have detected the
    // global convergence)
    if (settings.comm_settings.enable_onesided) {
        if (settings.convergence_settings.enable_global_simple_tree) {
            conv_tools::global_convergence_check_onesided_tree(
                settings, metadata, convergence_vector, converged_all_local,
                num_converged_procs, window_convergence);
        } else if (settings.convergence_settings
                       .enable_decentralized_leader_election) {
            conv_tools::global_convergence_decentralized<ValueType, IndexType,
                                                         MixedValueType>(
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
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::check_convergence(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
    ValueType &local_residual_norm, ValueType &local_residual_norm0,
    ValueType &global_residual_norm, ValueType &global_residual_norm0,
    int &num_converged_procs)
{
    int num_converged_p = 0;
    auto tolerance = metadata.tolerance;
    auto iter = metadata.iter_count;
    if (check_local_convergence(settings, metadata, local_solution,
                                global_solution, local_matrix, work_vector,
                                local_residual_norm, local_residual_norm0)) {
        num_converged_p = 1;
    } else {
        num_converged_p = 0;
    }
    metadata.post_process_data.local_residual_vector_out.push_back(
        local_residual_norm);
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


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::update_residual(
    const Settings &settings,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto one = gko::initialize<vec>({1.0}, settings.executor);
    auto neg_one = gko::initialize<vec>({-1.0}, settings.executor);

    local_matrix->apply(neg_one.get(), gko::lend(global_solution), one.get(),
                        gko::lend(solution_vector));
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::compute_residual_norm(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &global_matrix,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
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
                        &global_solution->get_values()[first_row])),
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
    global_solution->copy_from(global_sol.get());
    auto one = gko::initialize<vec>({1.0}, settings.executor);
    auto neg_one = gko::initialize<vec>({-1.0}, settings.executor);

    global_matrix->apply(neg_one.get(), gko::lend(global_solution), one.get(),
                         gko::lend(global_rhs));

    global_rhs->compute_norm2(rnorm.get());
    cpu_resnorm->copy_from(gko::lend(rnorm));
    residual_norm = cpu_resnorm->at(0);
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void Solve<ValueType, IndexType, MixedValueType>::clear(Settings &settings)
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

#define DECLARE_SOLVER(ValueType, IndexType, MixedValueType) \
    class Solve<ValueType, IndexType, MixedValueType>
INSTANTIATE_FOR_EACH_VALUE_MIXEDVALUE_AND_INDEX_TYPE(DECLARE_SOLVER);
#undef DECLARE_SOLVER


}  // namespace schwz
