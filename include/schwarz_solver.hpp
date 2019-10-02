#ifndef schwarz_solver_hpp
#define schwarz_solver_hpp

#include <cmath>
#include <fstream>
#include <memory>

#include <communicate.hpp>
#include <initialization.hpp>
#include <schwarz/config.hpp>
#include <solve.hpp>

#if SCHWARZ_USE_DEALII
#include <deal.II/lac/sparse_matrix.h>
#endif

namespace SchwarzWrappers {

template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class SolverBase : public Initialize<ValueType, IndexType>,
                   public Communicate<ValueType, IndexType>,
                   public Solve<ValueType, IndexType>
// public Preconditioner,
// public TearDown
{
public:
  SolverBase(Settings &settings, Metadata<ValueType, IndexType> &metadata);

#if SCHWARZ_USE_DEALII
  void initialize(const dealii::SparseMatrix<ValueType> &matrix,
                  const dealii::Vector<ValueType> &system_rhs);
#else
  void initialize();
#endif

  /**
   * Solve the linear system <tt>Ax=b</tt>. Depending on the information
   * provided by derived classes one of the linear solvers of Schwarz is
   * chosen.
   */
  void run(std::shared_ptr<gko::matrix::Dense<ValueType>> &solution);

  void print_vector(std::shared_ptr<gko::matrix::Dense<ValueType>> &vector,
                    int rank, std::string name);

  void print_matrix(
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &matrix,
      int rank, std::string name);

  std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> local_matrix;

  std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> triangular_factor;

  std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> interface_matrix;

  std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> global_matrix;

  std::shared_ptr<gko::matrix::Dense<ValueType>> local_rhs;

  std::shared_ptr<gko::matrix::Dense<ValueType>> global_rhs;

  std::shared_ptr<gko::matrix::Dense<ValueType>> local_solution;

  std::shared_ptr<gko::matrix::Dense<ValueType>> global_solution;

  // typename Communicate<ValueType, IndexType>::comm_struct comm_struct;

protected:
  Settings &settings;

  Metadata<ValueType, IndexType> &metadata;
  struct Communicate<ValueType, IndexType>::comm_struct comm_struct;
};

/**
 * An implementation of the solver interface using the RAS solver.
 *
 * @ingroup SchwarzWrappers
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class SolverRAS : public SolverBase<ValueType, IndexType> {
public:
  /**
   * Standardized data struct to pipe additional data to the solver.
   */
  struct AdditionalData {};

  SolverRAS(Settings &settings, Metadata<ValueType, IndexType> &metadata,
            const AdditionalData &data = AdditionalData());
  // protected:
  /**
   * Store a copy of the flags for this particular solver.
   */
  const AdditionalData additional_data;

  void setup_local_matrices(
      Settings &settings, Metadata<ValueType, IndexType> &metadata,
      std::vector<unsigned int> &partition_indices,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &interface_matrix)
      override;

  void setup_comm_buffers() override;

  void setup_windows(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer) override;

  void exchange_boundary(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector) override;

  void update_boundary(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
      const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &interface_matrix) override;
};

// #define DECLARE_SOLVER_BASE(ValueType, IndexType) class
// SolverBase<ValueType, IndexType>
//   INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SOLVER_BASE);
// #undef DECLARE_SOLVER_BASE

// explicit instantiations for SchwarzWrappers
// #define DECLARE_SOLVER_RAS(ValueType, IndexType) class SolverRAS<ValueType,
// IndexType>
//   INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_SOLVER_RAS);
// #undef DECLARE_SOLVER_RAS

} // namespace SchwarzWrappers

#endif
/*----------------------------   schwarz_solver.hpp
 * ---------------------------*/
