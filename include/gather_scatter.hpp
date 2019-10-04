
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

#ifndef gather_scatter_hpp
#define gather_scatter_hpp


#include <omp.h>
#include <ginkgo/ginkgo.hpp>


template <typename ValueType, typename IndexType>
extern void gather_values(const IndexType num_elems, const IndexType *indices,
                          const ValueType *from_array, ValueType *into_array);


template <typename ValueType, typename IndexType>
extern void scatter_values(const IndexType num_elems, const IndexType *indices,
                           const ValueType *from_array, ValueType *into_array);


template <typename ValueType, typename IndexType>
struct GatherScatter : public gko::Operation {
    GatherScatter(const bool flag, const IndexType num_elems,
                  const IndexType *indices, const ValueType *from_array,
                  ValueType *into_array)
        : flag{flag},
          num_elems{num_elems},
          indices{indices},
          from_array{from_array},
          into_array{into_array}
    {}

    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        if (flag)  // gather if true: TODO: improve this
        {
#pragma omp parallel for
            for (auto i = 0; i < num_elems; ++i) {
                into_array[i] = from_array[indices[i]];
            }
        } else  // scatter if false
        {
#pragma omp parallel for
            for (auto i = 0; i < num_elems; ++i) {
                into_array[indices[i]] = from_array[i];
            }
        }
    }

    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        if (flag) {
            gather_values(num_elems, indices, from_array, into_array);
        } else {
            scatter_values(num_elems, indices, from_array, into_array);
        }
    }
    const bool flag;
    const IndexType num_elems;
    const IndexType *indices;
    const ValueType *from_array;
    ValueType *into_array;
};


#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, gko::int32);                   \
    template _macro(double, gko::int32);                  \
    template _macro(float, gko::int64);                   \
    template _macro(double, gko::int64);

#define DECLARE_GATHERSCATTER(ValueType, IndexType) \
    struct GatherScatter<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_GATHERSCATTER);
#undef DECLARE_GATHERSCATTER


#endif  // gather_scatter.hpp
