
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


#include <communicate.hpp>
#include <exception_helpers.hpp>

namespace SchwarzWrappers {
template <typename ValueType, typename IndexType>
void Communicate<ValueType, IndexType>::setup_comm_buffers()
    SCHWARZ_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void Communicate<ValueType, IndexType>::setup_windows(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &main_buffer)
    SCHWARZ_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void Communicate<ValueType, IndexType>::exchange_boundary(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &last_solution_vector,
    std::ofstream &fp, std::ofstream &fpr)
    SCHWARZ_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void Communicate<ValueType, IndexType>::update_boundary(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &solution_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_old_solution,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &interface_matrix) SCHWARZ_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void Communicate<ValueType, IndexType>::local_to_global_vector(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_vector) {
  using vec = gko::matrix::Dense<ValueType>;
  auto alpha =
      gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, settings.executor);
  auto temp_vector = vec::create(
      settings.executor, gko::dim<2>(metadata.local_size, 1),
      (gko::Array<ValueType>::view(
          settings.executor, metadata.local_size,
          &global_vector->get_values()[metadata.first_row
                                           ->get_data()[metadata.my_rank]])),
      1);

  auto temp_vector2 = vec::create(
      settings.executor, gko::dim<2>(metadata.local_size, 1),
      (gko::Array<ValueType>::view(settings.executor, metadata.local_size,
                                   &local_vector->get_values()[0])),
      1);
  if (settings.convergence_settings.convergence_crit ==
      Settings::convergence_settings::local_convergence_crit::residual_based) {
    local_vector->add_scaled(alpha.get(), temp_vector.get());
    temp_vector->add_scaled(alpha.get(), local_vector.get());
  } else {
    // TODO GPU: DONE
    temp_vector->copy_from(temp_vector2.get());
  }
}

template <typename ValueType, typename IndexType>
void Communicate<ValueType, IndexType>::clear(Settings &settings) {
  if (settings.comm_settings.enable_onesided) {
    //MPI_Win_unlock_all(comm_struct.window_buffer);
    //MPI_Win_unlock_all(comm_struct.window_x);

    MPI_Win_free(&comm_struct.window_x);
  }
}

#define DECLARE_COMMUNICATE(ValueType, IndexType)                              \
  class Communicate<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_COMMUNICATE);
#undef DECLARE_COMMUNICATE

} // namespace SchwarzWrappers
