#ifndef comm_tools_hpp
#define comm_tools_hpp

#include <settings.hpp>
#include <communicate.hpp>

namespace SchwarzWrappers {

namespace ConvergenceTools {
template <typename ValueType, typename IndexType>
void put_all_local_residual_norms(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    ValueType &local_resnorm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_residual_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_residual_vector_out,
    MPI_Win &window_residual_vector) {
  auto num_subdomains = metadata.num_subdomains;
  auto my_rank = metadata.my_rank;
  auto l_res_vec = local_residual_vector->get_values();
  auto iter = metadata.iter_count;
  auto mpi_vtype = boost::mpi::get_mpi_datatype(l_res_vec[my_rank]);
  // TODO: Is this necessary ?
  // l_res_vec[my_rank] = std::min(l_res_vec[my_rank], local_resnorm);
  l_res_vec[my_rank] = local_resnorm;
  for (auto j = 0; j < num_subdomains; j++) {
    if (j != my_rank && iter > 0 &&
        l_res_vec[my_rank] !=
            global_residual_vector_out->at(iter - 1, my_rank)) {
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

template <typename ValueType, typename IndexType>
void propagate_all_local_residual_norms(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_s,
    ValueType &local_resnorm,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_residual_vector,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_residual_vector_out,
    MPI_Win &window_residual_vector) {
  auto num_subdomains = metadata.num_subdomains;
  auto my_rank = metadata.my_rank;
  auto l_res_vec = local_residual_vector->get_values();
  auto iter = metadata.iter_count;
  auto global_put = comm_s.global_put->get_data();
  auto neighbors_out = comm_s.neighbors_out->get_data();
  auto max_valtype = std::numeric_limits<ValueType>::max();
  auto mpi_vtype = boost::mpi::get_mpi_datatype(l_res_vec[my_rank]);

  // TODO: Is this necessary ?
  // l_res_vec[my_rank] = std::min(l_res_vec[my_rank], local_resnorm);
  l_res_vec[my_rank] = local_resnorm;
  for (auto i = 0; i < comm_s.num_neighbors_out; i++) {
    if ((global_put[i])[0] > 0) {
      auto p = neighbors_out[i];
      int flag = 0;
      if (iter == 0 || l_res_vec[my_rank] !=
                           global_residual_vector_out->at(iter - 1, my_rank))
        flag = 1;
      if (flag == 0) {
        for (auto j = 0; j < num_subdomains; j++) {
          if (j != p && iter > 0 && l_res_vec[j] != max_valtype &&
              l_res_vec[j] != global_residual_vector_out->at(iter - 1, j)) {
            flag++;
          }
        }
      }
      if (flag > 0) {
        for (auto j = 0; j < num_subdomains; j++) {
          if ((j == my_rank && (iter == 0 || l_res_vec[my_rank] !=
                                                 global_residual_vector_out->at(
                                                     iter - 1, my_rank))) ||
              (j != p && iter > 0 && l_res_vec[j] != max_valtype &&
               l_res_vec[j] != global_residual_vector_out->at(iter - 1, j))) {
            // double result;
            MPI_Accumulate(&l_res_vec[j], 1, mpi_vtype, p, j, 1, mpi_vtype,
                           MPI_MIN, window_residual_vector);
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

template <typename ValueType, typename IndexType>
void global_convergence_check_onesided_tree(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    int &converged_all_local, int &num_converged_procs,
    MPI_Win &window_convergence) {
  int ione = 1;
  auto num_subdomains = metadata.num_subdomains;
  auto my_rank = metadata.my_rank;
  auto conv_vector = convergence_vector->get_data();

  // if the child locally converged and if first time local convergence
  // detected, push up
  if (((conv_vector[0] == 1 &&
        conv_vector[1] == 1) || // both children converged
       (conv_vector[0] == 1 &&
        my_rank == num_subdomains / 2 - 1) || // only one child
       (my_rank >= num_subdomains / 2 && conv_vector[0] != 2)) && // leaf
      converged_all_local > 0) // locally deteced global convergence
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
    conv_vector[0] = 2; // to push up only once
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

template <typename ValueType, typename IndexType>
void global_convergence_check_onesided_propagate(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Communicate<ValueType, IndexType>::comm_struct &comm_s,
    std::shared_ptr<gko::Array<IndexType>> &convergence_vector,
    std::shared_ptr<gko::Array<IndexType>> &convergence_sent,
    std::shared_ptr<gko::Array<IndexType>> &convergence_local,
    int &converged_all_local, int &num_converged_procs,
    MPI_Win &window_convergence) {
  auto num_subdomains = metadata.num_subdomains;
  auto my_rank = metadata.my_rank;
  auto conv_vector = convergence_vector->get_data();
  auto conv_sent = convergence_sent->get_data();
  auto conv_local = convergence_local->get_data();
  auto global_put = comm_s.global_put->get_data();
  auto neighbors_out = comm_s.neighbors_out->get_data();
  // count how many processes have locally detected the global convergence
  if (settings.convergence_settings.enable_accumulate) {
    // > if this process has detected the global convergence
    // > let everyone know (by incrementing the counter)
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
    // > read (from the window) how many processed have locally detected
    // the global convergence
    num_converged_procs = conv_vector[0];
  } else {
    // > if this process has detected the global convergence
    // > put a check at my slot in the window
    if (converged_all_local == 1) {
      conv_vector[my_rank] = 1;
    }
    // > go through all the slots in the window
    // > and count how many processes have locally detected the global
    // convergence
    num_converged_procs = 0;
    for (auto j = 0; j < num_subdomains; j++) {
      conv_local[j] = conv_vector[j];
      num_converged_procs += conv_vector[j];
    }
    // > let the neighbors know who have detected the global convergence
    for (auto i = 0; i < comm_s.num_neighbors_out; i++) {
      if ((global_put[i])[0] > 0) {
        auto p = neighbors_out[i];
        int ione = 1;
        for (auto j = 0; j < num_subdomains; j++) {
          // only if not sent, yet
          if (conv_sent[j] == 0 && conv_local[j] == 1) {
            MPI_Put(&ione, 1, MPI_INT, p, j, 1, MPI_INT, window_convergence);
          }
        }
        if (settings.comm_settings.enable_flush_all) {
          MPI_Win_flush(p, window_convergence);
        } else if (settings.comm_settings.enable_flush_local) {
          MPI_Win_flush_local(p, window_convergence);
        }
      }
    }
    for (auto j = 0; j < num_subdomains; j++) {
      conv_sent[j] = conv_local[j];
    }
  }
}
} // namespace ConvergenceTools

// Explicit Instantiations
#define DECLARE_FUNCTION(ValueType, IndexType)                                 \
  void ConvergenceTools::put_all_local_residual_norms(                         \
      const Settings &, const Metadata<ValueType, IndexType> &, ValueType &,   \
      std::shared_ptr<gko::matrix::Dense<ValueType>> &,                        \
      std::shared_ptr<gko::matrix::Dense<ValueType>> &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_FUNCTION2(ValueType, IndexType)                                \
  void ConvergenceTools::propagate_all_local_residual_norms(                   \
      const Settings &, const Metadata<ValueType, IndexType> &,                \
      struct Communicate<ValueType, IndexType>::comm_struct &, ValueType &,    \
      std::shared_ptr<gko::matrix::Dense<ValueType>> &,                        \
      std::shared_ptr<gko::matrix::Dense<ValueType>> &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION2);
#undef DECLARE_FUNCTION2

#define DECLARE_FUNCTION3(ValueType, IndexType)                                \
  void ConvergenceTools::global_convergence_check_onesided_tree(               \
      const Settings &, const Metadata<ValueType, IndexType> &,                \
      std::shared_ptr<gko::Array<IndexType>> &, int &, int &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION3);
#undef DECLARE_FUNCTION3

#define DECLARE_FUNCTION4(ValueType, IndexType)                                \
  void ConvergenceTools::global_convergence_check_onesided_propagate(          \
      const Settings &, const Metadata<ValueType, IndexType> &,                \
      struct Communicate<ValueType, IndexType>::comm_struct &,                 \
      std::shared_ptr<gko::Array<IndexType>> &,                                \
      std::shared_ptr<gko::Array<IndexType>> &,                                \
      std::shared_ptr<gko::Array<IndexType>> &, int &, int &, MPI_Win &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION4);
#undef DECLARE_FUNCTION4

} // namespace SchwarzWrappers

#endif
