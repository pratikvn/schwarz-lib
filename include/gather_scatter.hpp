#ifndef gather_scatter_hpp
#define gather_scatter_hpp

#include <ginkgo/ginkgo.hpp>
#include <omp.h>

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
      : flag{flag}, num_elems{num_elems}, indices{indices},
        from_array{from_array}, into_array{into_array} {}

  void run(std::shared_ptr<const gko::OmpExecutor>) const override {
    if (flag) // gather if true: TODO: improve this
    {
#pragma omp parallel for
      for (auto i = 0; i < num_elems; ++i) {
        into_array[i] = from_array[indices[i]];
      }
    } else // scatter if false
    {
#pragma omp parallel for
      for (auto i = 0; i < num_elems; ++i) {
        into_array[indices[i]] = from_array[i];
      }
    }
  }

  void run(std::shared_ptr<const gko::CudaExecutor>) const override {
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

#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro)                      \
  template _macro(float, gko::int32);                                          \
  template _macro(double, gko::int32);                                         \
  template _macro(float, gko::int64);                                          \
  template _macro(double, gko::int64);

#define DECLARE_GATHERSCATTER(ValueType, IndexType)                            \
  struct GatherScatter<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_GATHERSCATTER);
#undef DECLARE_GATHERSCATTER

#endif
