
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


#ifndef restricted_schwarz_hpp
#define restricted_schwarz_hpp

#include <cmath>
#include <fstream>
#include <memory>

#include <schwarz/config.hpp>

#include <schwarz_base.hpp>

/**
 * The Schwarz wrappers namespace
 *
 * @ingroup schwarz_wrappers
 */
namespace schwz {

/**
 * An implementation of the solver interface using the RAS solver.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ingroup schwarz_class
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32,
          typename MixedValueType = gko::default_precision>
class SolverRAS : public SchwarzBase<ValueType, IndexType, MixedValueType> {
public:
    /**
     * The constructor that takes in the user settings and a metadata struct
     * containing the solver metadata.
     *
     * @param settings  The settings struct.
     * @param metadata  The metadata struct.
     * @param data  The additional data struct.
     */
    SolverRAS(Settings &settings, Metadata<ValueType, IndexType> &metadata);

    void setup_local_matrices(
        Settings &settings, Metadata<ValueType, IndexType> &metadata,
        std::vector<unsigned int> &partition_indices,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
        std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &interface_matrix) override;

    void setup_comm_buffers() override;

    void setup_windows(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer) override;

    void exchange_boundary(const Settings &settings,
                           const Metadata<ValueType, IndexType> &metadata,
                           const std::shared_ptr<gko::matrix::Dense<ValueType>>
                               &prev_global_solution,
                           std::shared_ptr<gko::matrix::Dense<ValueType>>
                               &global_solution) override;

    void update_boundary(
        const Settings &settings,
        const Metadata<ValueType, IndexType> &metadata,
        std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution,
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
            &interface_matrix) override;

    ValueType get_threshold(const Settings &settings,
                            const Metadata<ValueType, IndexType> &metadata);
};


}  // namespace schwz


#endif  // restricted_schwarz.hpp
