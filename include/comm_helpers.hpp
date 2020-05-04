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


#ifndef comm_helpers_hpp
#define comm_helpers_hpp


#include <memory>
#include <vector>


#include <communicate.hpp>
#include <schwarz/config.hpp>
#include <settings.hpp>


namespace schwz {
/**
 * @brief The CommHelper namespace .
 * @ref comm_helpers
 * @ingroup comm
 */
namespace CommHelpers {


template <typename ValueType, typename IndexType>
void transfer_one_by_one(const Settings &settings, MPI_Win &window,
                         ValueType *buffer, IndexType **offset,
                         const int num_neighbors, const IndexType *neighbors)
{
    ValueType dummy = 1.0;
    auto mpi_vtype = boost::mpi::get_mpi_datatype(dummy);
    for (auto p = 0; p < num_neighbors; p++) {
        if ((offset[p])[0] > 0) {
            if (settings.comm_settings.enable_put) {
                for (auto i = 0; i < (offset[p])[0]; i++) {
                    MPI_Put(&buffer[(offset[p])[i + 1]], 1, mpi_vtype,
                            neighbors[p], (offset[p])[i + 1], 1, mpi_vtype,
                            window);
                }
            } else if (settings.comm_settings.enable_get) {
                for (auto i = 0; i < (offset[p])[0]; i++) {
                    MPI_Get(&buffer[(offset[p])[i + 1]], 1, mpi_vtype,
                            neighbors[p], (offset[p])[i + 1], 1, mpi_vtype,
                            window);
                }
            }
            if (settings.comm_settings.enable_flush_all) {
                MPI_Win_flush(neighbors[p], window);
            } else if (settings.comm_settings.enable_flush_local) {
                MPI_Win_flush_local(neighbors[p], window);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void pack_buffer(const Settings &settings, ValueType *buffer,
                 ValueType *send_buffer, ValueType *work_vector,
                 IndexType **num_send_elems, const int offset,
                 const int send_subd, std::vector<bool> &is_local_neighbor)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using arr = gko::Array<IndexType>;
    using varr = gko::Array<ValueType>;
    if (settings.executor_string == "cuda") {
        auto tmp_send_buf = work_vector;
        auto tmp_idx_s = arr(settings.executor,
                             arr::view(settings.executor->get_master(),
                                       ((num_send_elems)[send_subd])[0],
                                       &((num_send_elems[send_subd])[1])));
        settings.executor->run(GatherScatter<ValueType, IndexType>(
            true, (num_send_elems[send_subd])[0], tmp_idx_s.get_data(), buffer,
            tmp_send_buf));
#if SCHW_HAVE_CUDA
        SCHWARZ_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpy(&(send_buffer[offset]), tmp_send_buf,
                       ((num_send_elems)[send_subd])[0] * sizeof(ValueType),
                       cudaMemcpyDeviceToDevice));
#endif
    } else {
        for (auto i = 0; i < (num_send_elems[send_subd])[0]; i++) {
            send_buffer[offset + i] =
                buffer[(num_send_elems[send_subd])[i + 1]];
        }
    }
}


template <typename ValueType, typename IndexType>
void transfer_buffer(const Settings &settings, MPI_Win &window,
                     ValueType *target_buffer, IndexType **num_elems,
                     const int offset, const int target_subd,
                     const IndexType *neighbors, const IndexType *displacements,
                     std::vector<bool> &is_local_neighbor)
{
    ValueType dummy = 1.0;
    auto mpi_vtype = boost::mpi::get_mpi_datatype(dummy);
    if (settings.comm_settings.enable_lock_local) {
        MPI_Win_lock(MPI_LOCK_SHARED, neighbors[target_subd], 0, window);
    }
    if (settings.comm_settings.enable_put) {
        MPI_Put(&target_buffer[offset], (num_elems[target_subd])[0], mpi_vtype,
                neighbors[target_subd], displacements[neighbors[target_subd]],
                (num_elems[target_subd])[0], mpi_vtype, window);
    } else if (settings.comm_settings.enable_get) {
        MPI_Get(&target_buffer[offset], (num_elems[target_subd])[0], mpi_vtype,
                neighbors[target_subd], displacements[neighbors[target_subd]],
                (num_elems[target_subd])[0], mpi_vtype, window);
    }
    if (settings.comm_settings.enable_flush_all) {
        MPI_Win_flush(neighbors[target_subd], window);
    } else if (settings.comm_settings.enable_flush_local) {
        MPI_Win_flush_local(neighbors[target_subd], window);
    }
    if (settings.comm_settings.enable_lock_local) {
        MPI_Win_unlock(neighbors[target_subd], window);
    }
}


template <typename ValueType, typename IndexType>
void unpack_buffer(const Settings &settings, ValueType *buffer,
                   ValueType *recv_buffer, ValueType *work_vector,
                   IndexType **num_recv_elems, const int offset,
                   const int recv_subd, std::vector<bool> &is_local_neighbor)
{
    using vec_vtype = gko::matrix::Dense<ValueType>;
    using arr = gko::Array<IndexType>;
    using varr = gko::Array<ValueType>;
    if (settings.executor_string == "cuda") {
        auto tmp_recv_buf = work_vector;
#if SCHW_HAVE_CUDA
        SCHWARZ_ASSERT_NO_CUDA_ERRORS(
            cudaMemcpy(tmp_recv_buf, &(recv_buffer[offset]),
                       ((num_recv_elems)[recv_subd])[0] * sizeof(ValueType),
                       cudaMemcpyDeviceToDevice));
#endif
        auto tmp_idx_r = arr(settings.executor,
                             arr::view(settings.executor->get_master(),
                                       ((num_recv_elems)[recv_subd])[0],
                                       &((num_recv_elems[recv_subd])[1])));
        settings.executor->run(GatherScatter<ValueType, IndexType>(
            false, (num_recv_elems[recv_subd])[0], tmp_idx_r.get_data(),
            tmp_recv_buf, buffer));
    } else {
        for (auto i = 0; i < (num_recv_elems[recv_subd])[0]; i++) {
            buffer[(num_recv_elems[recv_subd])[i + 1]] =
                recv_buffer[offset + i];
        }
    }
}

}  // namespace CommHelpers
}  // namespace schwz


#endif  // comm_helpers.hpp
