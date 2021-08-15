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


#ifndef event_helpers_hpp
#define event_helpers_hpp


#include <memory>
#include <vector>

#include <schwarz/config.hpp>

#include <communicate.hpp>
#include <gather.hpp>
#include <scatter.hpp>
#include <settings.hpp>


namespace schwz {
/**
 * @brief The EventHelper namespace .
 * @ref event_helpers
 * @ingroup comm
 */
namespace EventHelpers {

template <typename ValueType, typename IndexType, typename MixedValueType>
ValueType compute_nonadaptive_threshold(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata)
{
    return metadata.constant * std::pow(metadata.gamma, metadata.iter_count);
}

template <typename ValueType, typename IndexType, typename MixedValueType>
ValueType compute_adaptive_threshold(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
    int target_subd, int iter_diff)
{
    return comm_struct.thres->get_values()[target_subd] *
           std::pow(metadata.decay_param, iter_diff);
}

template <typename ValueType, typename IndexType, typename MixedValueType>
void compute_sender_slopes(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
    int target_subd, int iter_diff, ValueType value_diff)
{
    auto num_neighbors_out = comm_struct.num_neighbors_out;
    auto temp = 0.0;
    int i;

    for (i = 0; i < metadata.sent_history - 1; i++) {
        comm_struct.last_sent_slopes_avg
            ->get_values()[i * num_neighbors_out + target_subd] =
            comm_struct.last_sent_slopes_avg
                ->get_values()[(i + 1) * num_neighbors_out + target_subd];
        temp += comm_struct.last_sent_slopes_avg
                    ->get_values()[i * num_neighbors_out + target_subd];
    }
    if (iter_diff != 0) {
        comm_struct.last_sent_slopes_avg
            ->get_values()[i * num_neighbors_out + target_subd] =
            value_diff / iter_diff;
    }
    temp += comm_struct.last_sent_slopes_avg
                ->get_values()[i * num_neighbors_out + target_subd];
    temp = temp / metadata.sent_history;
    comm_struct.thres->get_values()[target_subd] = temp * metadata.horizon;
}

template <typename ValueType, typename IndexType, typename MixedValueType>
void compute_receiver_slopes(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
    int target_subd, int iter_diff, int num_get)
{
    auto global_get = comm_struct.global_get->get_data();
    auto num_recv = comm_struct.num_recv;

    for (auto i = 0; i < (global_get[target_subd])[0]; i++) {
        // shift old slopes
        int j;
        for (j = 0; j < metadata.recv_history - 1; j++) {
            comm_struct.last_recv_slopes
                ->get_values()[j * num_recv + (num_get + i)] =
                comm_struct.last_recv_slopes
                    ->get_values()[(j + 1) * num_recv + (num_get + i)];
        }

        // calculate new last_recv_slope using received
        // values
        // ITER DIFF SHOULD BE CALCULATED OUTSIDE BOTH NEW RECV AND EXTRAPOLATE
        if (iter_diff != 0) {
            comm_struct.last_recv_slopes
                ->get_values()[j * num_recv + (num_get + i)] =
                (comm_struct.recv_buffer->get_values()[num_get + i] -
                 comm_struct.last_recv_bdy->get_values()[num_get + i]) /
                iter_diff;
        } else {
            comm_struct.last_recv_slopes
                ->get_values()[j * num_recv + (num_get + i)] = 0.0;
        }

        // assign new last_recv_bdy
        comm_struct.last_recv_bdy->get_values()[num_get + i] =
            comm_struct.recv_buffer->get_values()[num_get + i];

    }  // end for
}

template <typename ValueType, typename IndexType, typename MixedValueType>
ValueType generate_extrapolated_buffer(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType, MixedValueType>::comm_struct
        &comm_struct,
    int target_subd, int iter_diff, int num_get)
{
    auto global_get = comm_struct.global_get->get_data();
    auto num_recv = comm_struct.num_recv;

    auto temp_avg = 0.0;

    for (auto i = 0; i < (global_get[target_subd])[0]; i++) {
        // calculate the avg slope for extrapolation
        auto slope_avg = 0.0;
        for (auto j = 0; j < metadata.recv_history; j++) {
            slope_avg += comm_struct.last_recv_slopes
                             ->get_values()[j * num_recv + (num_get + i)];
        }
        slope_avg = slope_avg / metadata.recv_history;

        comm_struct.extra_buffer->get_values()[num_get + i] =
            comm_struct.last_recv_bdy->get_values()[num_get + i] +
            slope_avg * iter_diff;
        temp_avg += comm_struct.extra_buffer->get_values()[num_get + i];
    }
    temp_avg = temp_avg / (global_get[target_subd])[0];
    return temp_avg;
}

}  // namespace EventHelpers
}  // namespace schwz


#endif  // event_helpers.hpp
