
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


namespace schwz {


/**
 * The initialization class that provides methods for initialization of the
 * solver.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ref init
 * @ingroup init
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

//CHANGED
    /**
     * Generates a random right hand side vector
     *
     * @param rhs The rhs vector.
     */
    void generate_random_rhs(std::vector<ValueType> &rhs);

    /**
     * Generates a dipole right hand side vector
     *
     * @param rhs The rhs vector.
     */
    void generate_dipole_rhs(std::vector<ValueType> &rhs);

    /**
     * Generates a sinusoidal right hand side vector
     *
     * @param rhs The rhs vector.
     */
    void generate_sin_rhs(std::vector<ValueType> &rhs); 
//END CHANGED

    /**
     * Generates the 2D global laplacian matrix.
     *
     * @param oned_laplacian_size  The size of the one d laplacian grid.
     * @param global_matrix  The global matrix.
     */
#if SCHW_HAVE_DEALII
    void setup_global_matrix(
        const std::string &filename, const gko::size_type &oned_laplacian_size,
        const dealii::SparseMatrix<ValueType> &matrix,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix);
#else
    void setup_global_matrix(
        const std::string &filename, const gko::size_type &oned_laplacian_size,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix);
#endif

    /**
     * The partitioning function. Allows the partition of the global matrix
     * depending with METIS and a regular 1D decomposition.
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
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_last_solution,
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
     * @param local_perm  The local permutation, obtained through RCM or METIS.
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

}  // namespace schwz

#endif  // initialization.hpp
