#ifndef solve_hpp
#define solve_hpp

#include <comm_tools.hpp>
#include <communicate.hpp>
#include <solver_tools.hpp>

namespace SchwarzWrappers {
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

  MPI_Win window_residual_vector;

  MPI_Win window_convergence;

  void setup_local_solver(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &local_matrix,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &triangular_factor,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs);

  void
  local_solve(const Settings &settings,
              const Metadata<ValueType, IndexType> &metadata,
              const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
                  &triangular_factor,
              std::shared_ptr<gko::matrix::Dense<ValueType>> &temp_loc_solution,
              std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution);

  void check_convergence(
      const Settings &settings, Metadata<ValueType, IndexType> &metadata,
      struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
      std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &local_matrix,
      ValueType &local_residual_norm, ValueType &local_residual_norm0,
      ValueType &global_residual_norm, ValueType &global_residual_norm0,
      IndexType &num_converged_procs);

  void check_global_convergence(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      struct Communicate<ValueType, IndexType>::comm_struct &comm_struct,
      std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
      ValueType &local_resnorm, ValueType &local_resnorm0,
      ValueType &global_resnorm, ValueType &global_resnorm0,
      int &converged_all_local, int &num_converged_procs);

  bool check_local_convergence(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &local_matrix,
      ValueType &local_resnorm, ValueType &local_resnorm0);

  void update_residual(
      const Settings &settings,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &local_matrix,
      const std::shared_ptr<gko::matrix::Dense<ValueType>>
          &global_old_solution);

  void compute_residual_norm(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &global_matrix,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
      ValueType &mat_norm, ValueType &rhs_norm, ValueType &sol_norm,
      ValueType &residual_norm);
  void clear(Settings &settings);

private:
  std::shared_ptr<gko::LinOp> solver;
  std::shared_ptr<gko::solver::LowerTrs<ValueType, IndexType>> L_solver;
  std::shared_ptr<gko::solver::UpperTrs<ValueType, IndexType>> U_solver;

#if SCHWARZ_USE_CHOLMOD
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

} // namespace SchwarzWrappers

#endif
/*----------------------------   solve.hpp ---------------------------*/
