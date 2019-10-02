#ifndef settings_hpp
#define settings_hpp

#include <boost/mpi/datatype.hpp>
#include <ginkgo/ginkgo.hpp>
#include <mpi.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <exception_helpers.hpp>
#include <gather_scatter.hpp>

#define MINIMAL_OVERLAP 2

namespace SchwarzWrappers {
struct Settings {
  std::string executor_string;

  std::shared_ptr<gko::Executor> executor = gko::ReferenceExecutor::create();

  enum partition_settings {
    partition_naive = 0x0,
    partition_metis = 0x1,
    partition_auto = 0x2,
    partition_zoltan = 0x3,
    partition_custom = 0x4
  };

  partition_settings partition = partition_settings::partition_naive;

  gko::int32 overlap = MINIMAL_OVERLAP;

  bool explicit_laplacian = false;

  bool enable_random_rhs = false;

  bool print_matrices = false;

  enum local_solver_settings {
    direct_solver_cholmod = 0x0,
    direct_solver_ginkgo = 0x1,
    iterative_solver_ginkgo = 0x2,
    iterative_solver_dealii = 0x3,
    solver_custom = 0x4
  };

  local_solver_settings local_solver =
      local_solver_settings::direct_solver_cholmod;

  bool naturally_ordered_factor = false;

  struct comm_settings {
    bool enable_onesided = false;
    bool enable_overlap = false;
    bool enable_push = true;
    bool enable_push_one_by_one = false;
    bool enable_flush_local = false;
    bool enable_flush_all = true;
  };

  comm_settings comm_settings;

  struct convergence_settings {
    bool put_all_local_residual_norms = true;
    bool enable_global_simple_tree = true;
    bool enable_global_check = true;
    bool enable_accumulate = false;

    enum local_convergence_crit { residual_based = 0x0, solution_based = 0x1 };

    local_convergence_crit convergence_crit =
        local_convergence_crit::solution_based;
  };

  convergence_settings convergence_settings;

  Settings(std::string executor_string = "reference")
      : executor_string(executor_string) {}
};

template <typename ValueType, typename IndexType> struct Metadata {
  MPI_Comm mpi_communicator;

  IndexType global_size = 0;

  IndexType local_size = 0;

  IndexType local_size_x = 0;

  IndexType local_size_o = 0;

  IndexType overlap_size = 0;

  IndexType num_subdomains = 1;

  int my_rank;

  int comm_size;

  int num_threads;

  IndexType iter_count = 10;

  ValueType tolerance = 1e-6;

  ValueType local_solver_tolerance = 1e-12;

  IndexType max_iters;

  ValueType current_residual_norm = -1.0;

  ValueType min_residual_norm = -1.0;

  std::vector<std::tuple<int, int, int, std::string, std::vector<ValueType>>>
      time_struct;

  std::shared_ptr<gko::Array<IndexType>> global_to_local;

  std::shared_ptr<gko::Array<IndexType>> local_to_global;

  std::shared_ptr<gko::Array<IndexType>> overlap_row;

  std::shared_ptr<gko::Array<IndexType>> first_row;

  std::shared_ptr<gko::Array<IndexType>> permutation;

  std::shared_ptr<gko::Array<IndexType>> i_permutation;
};

#define MEASURE_ELAPSED_FUNC_TIME(_func, _id, _rank, _name, _iter)             \
  {                                                                            \
    auto start_time = std::chrono::steady_clock::now();                        \
    _func;                                                                     \
    auto elapsed_time = std::chrono::duration<ValueType>(                      \
        std::chrono::steady_clock::now() - start_time);                        \
    if (_iter == 0) {                                                          \
      std::vector<ValueType> temp_vec(1, elapsed_time.count());                \
      metadata.time_struct.push_back(                                          \
          std::make_tuple(_id, _rank, _iter, #_name, temp_vec));               \
    } else {                                                                   \
      std::get<2>(metadata.time_struct[_id]) = _iter;                          \
      (std::get<4>(metadata.time_struct[_id]))                                 \
          .push_back(elapsed_time.count());                                    \
    }                                                                          \
  }
// if (!settings.comm_settings.enable_onesided)                    \
  //   {                                                                \
  // }                                                                  \
  // else                                                               \
  //   {                                                                \
  //     _func;                                                         \
  //     if (_iter == 0)                                                \
  //       {                                                            \
  //         std::vector<ValueType> temp_vec(1, -1.0);                  \
  //         metadata.time_struct.push_back(                            \
  //           std::make_tuple(_id, _rank, _iter, #_name, temp_vec));   \
  //       }                                                            \
  //   }                                                                \


#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro)                      \
  template _macro(float, gko::int32);                                          \
  template _macro(double, gko::int32);                                         \
  template _macro(float, gko::int64);                                          \
  template _macro(double, gko::int64);

// #define INSTANTIATE_FOR_EACH_TYPE(_macro)      \
//   template _macro(float, gko::int32);          \
//   template _macro(double, gko::int32);         \
//   template _macro(float, gko::int64);          \
//   template _macro(double, gko::int64);
// template _macro(gko::int32, gko::int64);     \
  // template _macro(gko::int64, gko::int64);     \
  // template _macro(gko::int64, gko::int32);     \
  // template _macro(gko::int32, gko::int32);

// explicit instantiations for SchwarzWrappers
#define DECLARE_METADATA(ValueType, IndexType)                                 \
  struct Metadata<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_METADATA);
#undef DECLARE_METADATA

} // namespace SchwarzWrappers

#endif
/*----------------------------   settings.hpp ---------------------------*/
