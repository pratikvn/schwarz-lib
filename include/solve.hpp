
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


#include <comm_tools.hpp>


#include <communicate.hpp>
#include <settings.hpp>
#include <solver_tools.hpp>


namespace SchwarzWrappers {


/**
 * The Solver class the provides the solver and the convergence checking
 * methods.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Solve : public Settings {
public:
    Solve() = default;

    Solve(const Settings &settings);

    friend class Initialize<ValueType, IndexType>;

protected:
    std::shared_ptr<gko::matrix::Dense<ValueType>> local_residual_vector;

    std::shared_ptr<gko::matrix::Dense<ValueType>> local_residual_vector_out;

    std::shared_ptr<gko::matrix::Dense<ValueType>> global_residual_vector_out;

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
     * Sets up the local solver from the user settings and computes the
     * triangular factors if needed.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param local_matrix  The local sudomain matrix.
     * @param triangular_factor  The triangular factor.
     * @param local_rhs  The local right hand side vector in the subdomain.
     */
    void setup_local_solver(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &triangular_factor,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs);

    /**
     * Computes the local solution.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param triangular_factor  The triangular factor.
     * @param temp_loc_solution The local solution vector in the subdomain.
     * @param local_solution The local solution vector in the subdomain.
     */
    void local_solve(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &triangular_factor,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &temp_loc_solution,
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
     * @param solution_vector  The workspace solution vector.
     * @param local_matrix  The local subdomain matrix.
     * @param local_residual_norm  The local residual norm.
     * @param local_residual_norm0  The initial local residual norm.
     * @param global_residual_norm  The global residual norm.
     * @param global_residual_norm0  The initial global residual norm.
     * @param num_converged_procs  The number of subdomains that have converged.
     */
    void check_convergence(
        const Settings &settings, Metadata<ValueType, IndexType> &metadata,
        struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
        std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
        const std::shared_ptr<gko::matrix::Dense<ValueType>>
            &global_old_solution,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        ValueType &local_residual_norm, ValueType &local_residual_norm0,
        ValueType &global_residual_norm, ValueType &global_residual_norm0,
        IndexType &num_converged_procs);

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
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
        std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
        ValueType &local_resnorm, ValueType &local_resnorm0,
        ValueType &global_resnorm, ValueType &global_resnorm0,
        int &converged_all_local, int &num_converged_procs);

    /**
     * Checks if the local subdomain has converged.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param solution_vector  The workspace solution vector.
     * @param global_old_solution  The global solution vector of the previous
     *                             iteration.
     * @param local_matrix  The local subdomain matrix.
     * @param local_residual_norm  The local residual norm.
     * @param local_residual_norm0  The initial local residual norm.
     *
     * @return local_convergence  If the local subdomain has converged.
     */
    bool check_local_convergence(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
        const std::shared_ptr<gko::matrix::Dense<ValueType>>
            &global_old_solution,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &local_matrix,
        ValueType &local_resnorm, ValueType &local_resnorm0);

    /**
     * Updates the residual.
     *
     * @param settings  The settings struct.
     * @param solution_vector  The workspace solution vector.
     * @param local_matrix  The local subdomain matrix.
     * @param global_old_solution  The global solution vector of the previous
     *                             iteration.
     */
    void update_residual(
        const Settings &settings,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
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
     * @param solution_vector  The workspace solution vector.
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
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
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

    Settings settings;
    Metadata<ValueType, IndexType> metadata;
};


}  // namespace SchwarzWrappers


#endif  // solve.hpp
