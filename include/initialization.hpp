#ifndef initialization_hpp
#define initialization_hpp


#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>


#include <schwarz/config.hpp>


#if SCHW_HAVE_DEALII
#include <deal.II/lac/sparse_matrix.h>
#endif


#include <partition_tools.hpp>
#include <settings.hpp>


namespace SchwarzWrappers {


/**
 * The initialization class that provides methods for initialization of the solver.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Initialize : public Settings, public Metadata<ValueType, IndexType> {
public:
    Initialize(Settings &settings, Metadata<ValueType, IndexType> &metadata);

    /**
     * The partition indices containing the subdomains to which each row(vertex)
     * of the matrix(graph) belongs to.
     */
    std::vector<unsigned int> partition_indices;

    /**
     * The cell weights for the partition algorithm.
     */
    std::vector<unsigned int> cell_weights;

    /**
     * Generates the right hand side vector
     *
     * @param rhs  The rhs vector.
     */
    void generate_rhs(std::vector<ValueType> &rhs);

#if SCHW_HAVE_DEALII
    void setup_global_matrix(
        const dealii::SparseMatrix<ValueType> &matrix,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix);
#endif

    /**
     * Generates the 2D global laplacian matrix.
     *
     * @param global_size  The size of the matrix.
     * @param global_matrix  The global matrix.
     */
    void setup_global_matrix_laplacian(
        const IndexType &global_size,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix);

    /**
     * The partitioning function. Allows the partition of the global matrix
     * depending with METIS and a naive 1D decomposition.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param global_matrix  The global matrix.
     * @param partition_indices  The partition indices [OUTPUT].
     */
    void partition(const Settings &settings,
                   const Metadata<ValueType, IndexType> &metadata,
                   const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
                       &global_matrix,
                   std::vector<unsigned int> &partition_indices);

    /**
     * Setup the vectors with default values and allocate mameory if not
     * allocated.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param local_rhs  The local right hand side vector in the subdomain.
     * @param global_rhs  The global right hand side vector.
     * @param local_solution  The local solution vector in the subdomain.
     * @param global_solution  The global solution vector.
     */
    void setup_vectors(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        std::vector<ValueType> &rhs,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution);

    /**
     * Sets up the local and the interface matrices from the global matrix and
     * the partition indices.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param partition_indices  The array containing the partition indices.
     * @param global_matrix The global system matrix.
     * @param local_matrix The local system matrix.
     * @param interface_matrix The interface matrix containing the interface and
     *                         the overlap data mainly used for exchanging
     *                         values between different sub-domains.
     */
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

}  // namespace SchwarzWrappers

#endif  // initialization.hpp
