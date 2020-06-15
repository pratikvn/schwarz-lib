
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

#include <functional>

#include <cuda_runtime.h>


#define BLOCK_SIZE 512

namespace schwz {

template <typename ValueType, typename IndexType, typename AdditionalOperation>
__global__ void gather_kernel(const IndexType num_elems,
                              const IndexType *indices,
                              const ValueType *gather_from,
                              ValueType *gather_into, AdditionalOperation op)
// std::function<ValueType(ValueType &, ValueType &)> op)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_elems) {
        gather_into[row] = op(gather_into[row], gather_from[indices[row]]);
    }
}


template <typename ValueType, typename IndexType>
void gather_values(
    const IndexType num_elems, const IndexType *indices,
    const ValueType *gather_from, ValueType *gather_into
    // std::function<ValueType __device__(const ValueType &, const ValueType &)>
    // std::function<__device__ ValueType(const ValueType &, const ValueType &)>
    //     op)
)
{
    dim3 grid((num_elems + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    auto op = [] __device__(const ValueType &x, const ValueType &y) {
        return y;
    };
    gather_kernel<<<grid, BLOCK_SIZE, 0, 0>>>(num_elems, indices, gather_from,
                                              gather_into, op);
}

#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int);                          \
    template _macro(double, int);                         \
    template _macro(int, int);                            \
    template _macro(long int, int);                       \
    template _macro(float, long int);                     \
    template _macro(double, long int);                    \
    template _macro(int, long int);                       \
    template _macro(long int, long int);


#define DECLARE_GATHER(ValueType, IndexType)                                  \
    void gather_values(const IndexType, const IndexType *, const ValueType *, \
                       ValueType *)
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_GATHER);
#undef DECLARE_GATHER

}  // namespace schwz