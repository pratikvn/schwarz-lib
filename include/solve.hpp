
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


#ifndef solve_hpp
#define solve_hpp

#include <memory>
#include <vector>

#include <settings.hpp>

#include <communicate.hpp>
#include <conv_tools.hpp>
#include <solver_tools.hpp>


namespace schwz {


/**
 * The Solver class the provides the solver and the convergence checking
 * methods.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ref solve
 * @ingroup solve
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          typename MixedValueType = gko::default_precision>
class Solve : public Settings {
public:
    using ResidualCriterionFactory =
        typename gko::stop::ResidualNormReduction<ValueType>::Factory;
    using IterationCriterionFactory = typename gko::stop::Iteration::Factory;

    Solve() = default;

    Solve(const Settings &settings);

    friend class Initialize<ValueType, IndexType>;

protected:
    std::shared_ptr<gko::matrix::Dense<ValueType>> local_residual_vector;

    std::shared_ptr<gko::matrix::Dense<ValueType>> residual_vector;

    std::shared_ptr<gko::Array<IndexType>> convergence_vector;

    std::shared_ptr<gko::Array<IndexType>> convergence_sent;

    std::shared_ptr<gko::Array<IndexType>> convergence_local;

    /**
     * The RDMA window used to communicate the residual vectors
     */
    MPI_Win window_residual_vector;

    /**
     * The RDMA window used to communicate the convergence
     */
    MPI_Win window_convergence;

    /**
     * Sets up the local solver from the user settings
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param local_matrix  The local sudomain matrix.
     * @param triangular_factor_l  The lower triangular factor.
     * @param triangular_factor_u  The upper triangular factor.
     * @param local_perm  The local permutation vector in the subdomain.
     * @param local_inv_perm  The local inverse permutation vector in the
     * subdomain.
     * @param local_rhs  The local right hand side vector in the subdomain.
     */
    void setup_local_solver(
        const Settings &settings, Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &triangular_factor_l,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &triangular_factor_u,
        std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_perm,
        std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_inv_perm,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs);


    /**
     * Computes the triangular factors based on the factorization type needed..
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param local_matrix  The local sudomain matrix.
     * subdomain.
     */
    void compute_local_factors(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix);


    /**
     * Computes the local solution.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param triangular_factor_l  The lower triangular factor.
     * @param triangular_factor_u  The upper triangular factor.
     * @param local_perm  The local permutation vector in the subdomain.
     * @param local_inv_perm  The local inverse permutation vector in the
     * @param init_guess  The initial solution for the local iterative solver.
     * @param local_solution The local solution vector in the subdomain.
     */
    void local_solve(
        const Settings &settings, Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &triangular_factor_l,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &triangular_factor_u,
        std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_perm,
        std::shared_ptr<gko::matrix::Permutation<IndexType>> &local_inv_perm,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &init_guess,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution);

    /**
     * Checks how many subdomains have converged.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param comm_struct  The struct containing the communication data.
     * @param convergence_vector  The array containing the convergence data.
     * @param global_old_solution  The global solution vector of the previous
     *                             iteration.
     * @param global_solution  The workspace solution vector.
     * @param local_matrix  The local subdomain matrix.
     * @param local_residual_norm  The local residual norm.
     * @param local_residual_norm0  The initial local residual norm.
     * @param global_residual_norm  The global residual norm.
     * @param global_residual_norm0  The initial global residual norm.
     * @param num_converged_procs  The number of subdomains that have converged.
     */
    void check_convergence(
        const Settings &settings, Metadata<ValueType, IndexType> &metadata,
        struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
            &comm_struct,
        std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
        const std::shared_ptr<gko::matrix::Dense<ValueType>>
            &global_old_solution,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
        ValueType &local_residual_norm, ValueType &local_residual_norm0,
        ValueType &global_residual_norm, ValueType &global_residual_norm0,
        int &num_converged_procs);

    /**
     * Checks if global convergence has been achieved.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param comm_struct  The struct containing the communication data.
     * @param convergence_vector  The array containing the convergence data.
     * @param local_residual_norm  The local residual norm.
     * @param local_residual_norm0  The initial local residual norm.
     * @param global_residual_norm  The global residual norm.
     * @param global_residual_norm0  The initial global residual norm.
     * @param converged_all_local  A flag which is true if all local subdomains
     *                             have converged.
     * @param num_converged_procs  The number of subdomains that have converged.
     */
    void check_global_convergence(
        const Settings &settings, Metadata<ValueType, IndexType> &metadata,
        struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
            &comm_struct,
        std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
        ValueType &local_resnorm, ValueType &local_resnorm0,
        ValueType &global_resnorm, ValueType &global_resnorm0,
        int &converged_all_local, int &num_converged_procs);

    /**
     * Checks if the local subdomain has converged.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param global_solution  The workspace solution vector.
     * @param global_old_solution  The global solution vector of the previous
     *                             iteration.
     * @param local_matrix  The local subdomain matrix.
     * @param local_residual_norm  The local residual norm.
     * @param local_residual_norm0  The initial local residual norm.
     *
     * @return local_convergence  If the local subdomain has converged.
     */
    bool check_local_convergence(
        const Settings &settings, Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
        const std::shared_ptr<gko::matrix::Dense<ValueType>>
            &global_old_solution,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &work_vector,
        ValueType &local_resnorm, ValueType &local_resnorm0);

    /**
     * Updates the residual.
     *
     * @param settings  The settings struct.
     * @param global_solution  The workspace solution vector.
     * @param local_matrix  The local subdomain matrix.
     * @param global_old_solution  The global solution vector of the previous
     *                             iteration.
     */
    void update_residual(
        const Settings &settings,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        const std::shared_ptr<gko::matrix::Dense<ValueType>>
            &global_old_solution);

    /**
     * Compute the final residual norm.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param global_matrix  The global matrix.
     * @param global_solution  The workspace solution vector.
     * @param global_rhs  The global right hand side vector.
     * @param mat_norm  The matrix norm of the global matrix.
     * @param rhs_norm  The rhs norm of the input rhs.
     * @param sol_norm  The sol norm of the solution vector.
     * @param residual_norm  The residual norm.
     */
    void compute_residual_norm(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &global_matrix,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
        ValueType &mat_norm, ValueType &rhs_norm, ValueType &sol_norm,
        ValueType &residual_norm);

    /**
     * Clears the data.
     */
    void clear(Settings &settings);

private:
    /**
     * The local iterative solver from Ginkgo.
     */
    std::shared_ptr<gko::LinOp> solver;

    /**
     * The local iterative solver residual criterion.
     */
    std::shared_ptr<gko::stop::Combined::Factory> combined_criterion;

    // /**
    //  * The local iterative solver iteration criterion.
    //  */
    // std::shared_ptr<IterationCriterionFactory> iteration_criterion;

    /**
     * The local iterative solver iteration criterion.
     */
    std::shared_ptr<gko::log::Record> record_logger;

    /**
     * The local lower triangular solver from Ginkgo.
     */
    std::shared_ptr<gko::solver::LowerTrs<ValueType, IndexType>> L_solver;

    /**
     * The local upper triangular solver from Ginkgo.
     */
    std::shared_ptr<gko::solver::UpperTrs<ValueType, IndexType>> U_solver;

#if SCHW_HAVE_CHOLMOD
    struct cholmod {
        int num_nonzeros;
        int num_rows;

        cholmod_common settings;
        cholmod_sparse *system_matrix;
        cholmod_dense *rhs;
        cholmod_factor *L_factor;
        int sorted;
        int packed;
        int stype;
        int xtype;
    };
    cholmod cholmod;
#endif

#if SCHW_HAVE_UMFPACK
    struct umfpack {
        int factor_l_nnz;
        int factor_u_nnz;
        int nz_udiag;
        int n_row;
        int n_col;
        int do_reciproc;
        void *numeric;
        std::shared_ptr<gko::matrix::Dense<ValueType>> row_scale;
        int status;
        double control[UMFPACK_CONTROL];
        double info[UMFPACK_INFO];
    };
    umfpack umfpack;
#endif

    Settings settings;
    Metadata<ValueType, IndexType> metadata;
};


}  // namespace schwz


#endif  // solve.hpp
