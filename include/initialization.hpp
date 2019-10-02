#ifndef initialization_hpp
#define initialization_hpp

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <schwarz/config.hpp>

#if SCHWARZ_USE_DEALII
#include <deal.II/lac/sparse_matrix.h>
#endif

#include <partition_tools.hpp>

namespace SchwarzWrappers {
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Initialize : public Settings, public Metadata<ValueType, IndexType> {
public:
  Initialize(Settings &settings, Metadata<ValueType, IndexType> &metadata);
  // Initialize();
  // virtual ~Initialize() = default;

  std::vector<unsigned int> partition_indices;

  std::vector<unsigned int> cell_weights;

#if SCHWARZ_USE_DEALII
  void setup_global_matrix(
      const dealii::SparseMatrix<ValueType> &matrix,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix);
#endif

  void generate_rhs(std::vector<ValueType> &rhs);

  void setup_global_matrix_laplacian(
      const IndexType &global_size,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix);

  void partition(const Settings &settings,
                 const Metadata<ValueType, IndexType> &metadata,
                 const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
                     &global_matrix,
                 std::vector<unsigned int> &partition_indices);

  void setup_vectors(
      const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
      std::vector<ValueType> &rhs,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
      std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution);

  virtual void setup_local_matrices(
      Settings &settings, Metadata<ValueType, IndexType> &metadata,
      std::vector<unsigned int> &partition_indices,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
      std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
          &interface_matrix) = 0;

private:
  Settings &settings;

  Metadata<ValueType, IndexType> &metadata;
};

} // namespace SchwarzWrappers

#endif
/*----------------------------   initialization.hpp
 * ---------------------------*/
