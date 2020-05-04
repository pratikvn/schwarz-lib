
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


#ifndef schwarz_base_hpp
#define schwarz_base_hpp

#include <cmath>
#include <fstream>
#include <memory>

#include <omp.h>

#include <schwarz/config.hpp>

#include <communicate.hpp>
#include <initialization.hpp>
#include <solve.hpp>


#if SCHW_HAVE_DEALII
#include <deal.II/lac/sparse_matrix.h>
#endif

/**
 * The Schwarz wrappers namespace
 *
 * @ingroup schwarz_wrappers
 */
namespace schwz {

/**
 * The Base solver class is meant to be the class implementing the common
 * implementations for all the schwarz methods. It derives from the
 * Initialization class, the Communication class and the Solve class all of
 * which are templated.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ingroup schwarz_class
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class SchwarzBase : public Initialize<ValueType, IndexType>,
                    public Communicate<ValueType, IndexType>,
                    public Solve<ValueType, IndexType> {
public:
    /**
     * The constructor that takes in the user settings and a metadata struct
     * containing the solver metadata.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     */
    SchwarzBase(Settings &settings, Metadata<ValueType, IndexType> &metadata);

#if SCHW_HAVE_DEALII
    /**
     * Initialize the matrix and vectors obtained from the deal.ii.
     *
     * @param matrix  The system matrix.
     * @param system_rhs  The right hand side vector.
     */
    void initialize(const dealii::SparseMatrix<ValueType> &matrix,
                    const dealii::Vector<ValueType> &system_rhs);
#endif

    /**
     * Initialize the matrix and vectors.
     */
    void initialize();

    /**
     * The function that runs the actual solver and obtains the final solution.
     *
     * @param solution  The solution vector.
     */
    void run(std::shared_ptr<gko::matrix::Dense<ValueType>> &solution);

    /**
     * The auxiliary function that prints a passed in vector.
     *
     * @param vector  The vector to be printed.
     * @param subd  The subdomain on which the vector exists.
     * @param name  The name of the vector as a string.
     */
    void print_vector(
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &vector, int subd,
        std::string name);

    /**
     * The auxiliary function that prints a passed in CSR matrix.
     *
     * @param matrix  The matrix to be printed.
     * @param subd  The subdomain on which the vector exists.
     * @param name  The name of the matrix as a string.
     */
    void print_matrix(
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &matrix,
        int rank, std::string name);

    /**
     * The local subdomain matrix.
     */
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> local_matrix;

    /**
     * The local subdomain permutation matrix/array.
     */
    std::shared_ptr<gko::matrix::Permutation<IndexType>> local_perm;

    /**
     * The local subdomain inverse permutation matrix/array.
     */
    std::shared_ptr<gko::matrix::Permutation<IndexType>> local_inv_perm;

    /**
     * The local lower triangular factor used for the triangular solves.
     */
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> triangular_factor_l;

    /**
     * The local upper triangular factor used for the triangular solves.
     */
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> triangular_factor_u;

    /**
     * The local interface matrix.
     */
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> interface_matrix;

    /**
     * The global matrix.
     */
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> global_matrix;

    /**
     * The local right hand side.
     */
    std::shared_ptr<gko::matrix::Dense<ValueType>> local_rhs;

    /**
     * The global right hand side.
     */
    std::shared_ptr<gko::matrix::Dense<ValueType>> global_rhs;

    /**
     * A work vector on the device.
     */
    std::shared_ptr<gko::matrix::Dense<ValueType>> work_vector;

    /**
     * The work vector on the CPU.
     */
    std::shared_ptr<gko::matrix::Dense<ValueType>> cpu_work_vector;

    /**
     * The local solution vector.
     */
    std::shared_ptr<gko::matrix::Dense<ValueType>> local_solution;

    /**
     * The global solution vector.
     */
    std::shared_ptr<gko::matrix::Dense<ValueType>> global_solution;

    /**
     * The global residual vector.
     */
    std::vector<ValueType> local_residual_vector_out;

    /**
     * The local residual vector.
     */
    std::vector<std::vector<ValueType>> global_residual_vector_out;

protected:
    /**
     * The settings struct used to store the solver and other auxiliary
     * settings.
     */
    Settings &settings;

    /**
     * The metadata struct used to store the solver metadata.
     */
    Metadata<ValueType, IndexType> &metadata;

    /**
     * The communication struct used to store the metadata and arrays needed for
     * the communication bewtween subdomains.
     */
    struct Communicate<ValueType, IndexType>::comm_struct comm_struct;
};


}  // namespace schwz


#endif  // schwarz_base.hpp
