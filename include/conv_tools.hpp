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


#ifndef conv_tools_hpp
#define conv_tools_hpp

#include <algorithm>
#include <functional>
#include <memory>


#include <communicate.hpp>
#include <settings.hpp>


namespace schwz {
/**
 * @brief The conv_tools namespace .
 * @ref conv_tools
 * @ingroup solve
 */
namespace conv_tools {


template <typename ValueType, typename IndexType>
void put_all_local_residual_norms(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    ValueType &local_resnorm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_residual_vector,
    MPI_Win &window_residual_vector)
{
    auto num_subdomains = metadata.num_subdomains;
    auto my_rank = metadata.my_rank;
    auto l_res_vec = local_residual_vector->get_values();
    auto iter = metadata.iter_count;
    auto mpi_vtype = schwz::mpi::get_mpi_datatype(l_res_vec[my_rank]);

    l_res_vec[my_rank] = std::min(l_res_vec[my_rank], local_resnorm);
    for (auto j = 0; j < num_subdomains; j++) {
        auto gres =
            metadata.post_process_data.global_residual_vector_out[my_rank];
        if (j != my_rank && iter > 0 && l_res_vec[my_rank] != gres[iter - 1]) {
            MPI_Put(&l_res_vec[my_rank], 1, mpi_vtype, j, my_rank, 1, mpi_vtype,
                    window_residual_vector);
            if (settings.comm_settings.enable_flush_all) {
                MPI_Win_flush(j, window_residual_vector);
            } else if (settings.comm_settings.enable_flush_local) {
                MPI_Win_flush_local(j, window_residual_vector);
            }
        }
    }
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void propagate_all_local_residual_norms(
    const Settings &settings, Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_s,
    ValueType &local_resnorm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_residual_vector,
    MPI_Win &window_residual_vector)
{
    auto num_subdomains = metadata.num_subdomains;
    auto my_rank = metadata.my_rank;
    auto l_res_vec = local_residual_vector->get_values();
    auto iter = metadata.iter_count;
    auto global_put = comm_s.global_put->get_data();
    auto neighbors_out = comm_s.neighbors_out->get_data();
    auto max_valtype = std::numeric_limits<ValueType>::max();
    auto mpi_vtype = schwz::mpi::get_mpi_datatype(l_res_vec[my_rank]);

    l_res_vec[my_rank] = std::min(l_res_vec[my_rank], local_resnorm);
    auto gres = metadata.post_process_data.global_residual_vector_out[my_rank];
    for (auto i = 0; i < comm_s.num_neighbors_out; i++) {
        if ((global_put[i])[0] > 0) {
            auto p = neighbors_out[i];
            int flag = 0;
            if (iter == 0 || l_res_vec[my_rank] != gres[iter - 1]) flag = 1;
            if (flag == 0) {
                for (auto j = 0; j < num_subdomains; j++) {
                    if (j != p && iter > 0 && l_res_vec[j] != max_valtype &&
                        l_res_vec[j] !=
                            (metadata.post_process_data
                                 .global_residual_vector_out[j])[iter - 1]) {
                        flag++;
                    }
                }
            }
            if (flag > 0) {
                for (auto j = 0; j < num_subdomains; j++) {
                    if ((j == my_rank &&
                         (iter == 0 || l_res_vec[my_rank] != gres[iter - 1])) ||
                        (j != p && iter > 0 && l_res_vec[j] != max_valtype &&
                         l_res_vec[j] !=
                             (metadata.post_process_data
                                  .global_residual_vector_out[j])[iter - 1])) {
                        // double result;
                        MPI_Accumulate(&l_res_vec[j], 1, mpi_vtype, p, j, 1,
                                       mpi_vtype, MPI_MIN,
                                       window_residual_vector);
                    }
                }
                if (settings.comm_settings.enable_flush_all) {
                    MPI_Win_flush(p, window_residual_vector);
                } else if (settings.comm_settings.enable_flush_local) {
                    MPI_Win_flush_local(p, window_residual_vector);
                }
            }
        }
    }
}

// This implementation is from Yamazaki et.al 2019
// (https://doi.org/10.1016/j.parco.2019.05.004)
template <typename ValueType, typename IndexType>
void global_convergence_check_onesided_tree(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    int &converged_all_local, int &num_converged_procs,
    MPI_Win &window_convergence)
{
    int ione = 1;
    auto num_subdomains = metadata.num_subdomains;
    auto my_rank = metadata.my_rank;
    auto conv_vector = convergence_vector->get_data();

    // if the child locally converged and if first time local convergence
    // detected, push up
    if (((conv_vector[0] == 1 &&
          conv_vector[1] == 1) ||  // both children converged
         (conv_vector[0] == 1 &&
          my_rank == num_subdomains / 2 - 1) ||  // only one child
         (my_rank >= num_subdomains / 2 && conv_vector[0] != 2)) &&  // leaf
        converged_all_local > 0)  // locally deteced global convergence
    {
        if (my_rank == 0) {
            // on the top, start going down
            conv_vector[2] = 1;
        } else {
            // push to parent
            int p = (my_rank - 1) / 2;
            int id = (my_rank % 2 == 0 ? 1 : 0);
            MPI_Put(&ione, 1, MPI_INT, p, id, 1, MPI_INT, window_convergence);
            if (settings.comm_settings.enable_flush_all) {
                MPI_Win_flush(p, window_convergence);
            } else if (settings.comm_settings.enable_flush_local) {
                MPI_Win_flush_local(p, window_convergence);
            }
        }
        conv_vector[0] = 2;  // to push up only once
    }

    // if first time global convergence detected, push down
    if (conv_vector[2] == 1) {
        int p = 2 * my_rank + 1;
        if (p < num_subdomains) {
            MPI_Put(&ione, 1, MPI_INT, p, 2, 1, MPI_INT, window_convergence);
            if (settings.comm_settings.enable_flush_all) {
                MPI_Win_flush(p, window_convergence);
            } else if (settings.comm_settings.enable_flush_local) {
                MPI_Win_flush_local(p, window_convergence);
            }
        }
        p++;
        if (p < num_subdomains) {
            MPI_Put(&ione, 1, MPI_INT, p, 2, 1, MPI_INT, window_convergence);
            if (settings.comm_settings.enable_flush_all) {
                MPI_Win_flush(p, window_convergence);
            } else if (settings.comm_settings.enable_flush_local) {
                MPI_Win_flush_local(p, window_convergence);
            }
        }
        conv_vector[1]++;
        num_converged_procs = num_subdomains;
    } else {
        num_converged_procs = 0;
    }
}


template <typename ValueType, typename IndexType, typename MixedValueType>
void global_convergence_decentralized(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_s,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    std::shared_ptr<gko::Array<IndexType>> &convergence_sent,
    std::shared_ptr<gko::Array<IndexType>> &convergence_local,
    int &converged_all_local, int &num_converged_procs,
    MPI_Win &window_convergence)
{
    auto num_subdomains = metadata.num_subdomains;
    auto my_rank = metadata.my_rank;
    auto conv_vector = convergence_vector->get_data();
    auto conv_sent = convergence_sent->get_data();
    auto conv_local = convergence_local->get_data();
    auto global_put = comm_s.global_put->get_data();
    auto neighbors_out = comm_s.neighbors_out->get_data();
    if (settings.convergence_settings.enable_accumulate) {
        if (converged_all_local == 1) {
            for (auto j = 0; j < num_subdomains; j++) {
                if (j != my_rank) {
                    int ione = 1;
                    MPI_Accumulate(&ione, 1, MPI_INT, j, 0, 1, MPI_INT, MPI_SUM,
                                   window_convergence);
                    if (settings.comm_settings.enable_flush_all) {
                        MPI_Win_flush(j, window_convergence);
                    } else if (settings.comm_settings.enable_flush_local) {
                        MPI_Win_flush_local(j, window_convergence);
                    }
                } else {
                    conv_vector[0]++;
                }
            }
        }
        num_converged_procs = conv_vector[0];
    } else {
        if (converged_all_local == 1) {
            conv_vector[my_rank] = 1;
        }
        num_converged_procs = 0;
        std::copy(conv_vector, conv_vector + num_subdomains, conv_local);
        num_converged_procs =
            std::accumulate(conv_vector, conv_vector + num_subdomains, 0);
        for (auto i = 0; i < comm_s.num_neighbors_out; i++) {
            if ((global_put[i])[0] > 0) {
                auto p = neighbors_out[i];
                int ione = 1;
                for (auto j = 0; j < num_subdomains; j++) {
                    if (conv_sent[j] == 0 && conv_local[j] == 1) {
                        MPI_Put(&ione, 1, MPI_INT, p, j, 1, MPI_INT,
                                window_convergence);
                    }
                }
                if (settings.comm_settings.enable_flush_all) {
                    MPI_Win_flush(p, window_convergence);
                } else if (settings.comm_settings.enable_flush_local) {
                    MPI_Win_flush_local(p, window_convergence);
                }
            }
        }
        std::copy(conv_local, conv_local + num_subdomains, conv_sent);
    }
}

/*
// Explicit Instantiations
#define DECLARE_FUNCTION(ValueType, IndexType)                           \
    void put_all_local_residual_norms(                                   \
        const Settings &, Metadata<ValueType, IndexType> &, ValueType &, \
        std::shared_ptr<gko::matrix::Dense<ValueType>> &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_FUNCTION2(ValueType, IndexType, MixedValueType)               \
    void propagate_all_local_residual_norms(                                  \
        const Settings &, Metadata<ValueType, IndexType> &,                   \
        struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct \
            &,                                                                \
        ValueType &, std::shared_ptr<gko::matrix::Dense<ValueType>> &,        \
        MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_MIXEDVALUE_AND_INDEX_TYPE(DECLARE_FUNCTION2);
#undef DECLARE_FUNCTION2

#define DECLARE_FUNCTION3(ValueType, IndexType)                   \
    void global_convergence_check_onesided_tree(                  \
        const Settings &, const Metadata<ValueType, IndexType> &, \
        std::shared_ptr<gko::Array<IndexType>> &, int &, int &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION3);
#undef DECLARE_FUNCTION3

#define DECLARE_FUNCTION4(ValueType, IndexType, MixedValueType)               \
    void global_convergence_decentralized(                                    \
        const Settings &, const Metadata<ValueType, IndexType> &,             \
        struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct \
            &,                                                                \
        std::shared_ptr<gko::Array<IndexType>> &,                             \
        std::shared_ptr<gko::Array<IndexType>> &,                             \
        std::shared_ptr<gko::Array<IndexType>> &, int &, int &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_MIXEDVALUE_AND_INDEX_TYPE(DECLARE_FUNCTION4);
#undef DECLARE_FUNCTION4
*/

}  // namespace conv_tools
}  // namespace schwz


#endif  // conv_tools.hpp
