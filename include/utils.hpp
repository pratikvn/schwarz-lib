#ifndef utils_hpp
#define utils_hpp

#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <settings.hpp>
#include <string>

namespace SchwarzWrappers {
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
struct Utils {
  Utils() = default;

  static int get_local_rank(MPI_Comm mpi_communicator);

  static int get_local_num_procs(MPI_Comm mpi_communicator);

  static void print_matrix(const gko::matrix::Csr<ValueType, IndexType> *matrix,
                           int rank, std::string name);

  static void print_vector(const gko::matrix::Dense<ValueType> *vector,
                           int rank, std::string name);
};
} // namespace SchwarzWrappers

#endif
/*----------------------------   utils.hpp ---------------------------*/
