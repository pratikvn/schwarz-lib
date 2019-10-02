

// #include <exception_helpers.hpp>
#include <initialization.hpp>
#include <solver_tools.hpp>

namespace SchwarzWrappers {
template <typename ValueType, typename IndexType>
Initialize<ValueType, IndexType>::Initialize(
    Settings &settings, Metadata<ValueType, IndexType> &metadata)
    : settings(settings), metadata(metadata) {
  MPI_Comm_rank(metadata.mpi_communicator, &metadata.my_rank);
  MPI_Comm_size(metadata.mpi_communicator, &metadata.comm_size);
  metadata.num_subdomains = metadata.comm_size;
}

template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::setup_local_matrices(
    Settings &settings, Metadata<ValueType, IndexType> &metadata,
    std::vector<unsigned int> &partition_indices,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &local_matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &interface_matrix)
    SCHWARZ_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::generate_rhs(
    std::vector<ValueType> &rhs) {
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::default_random_engine engine;
  for (gko::size_type i = 0; i < rhs.size(); ++i) {
    rhs[i] = unif(engine);
  }
}

#if SCHWARZ_USE_DEALII

template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::setup_global_matrix(
    const dealii::SparseMatrix<ValueType> &matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix) {
  using index_type = IndexType;
  using value_type = ValueType;
  using mtx = gko::matrix::Csr<value_type, index_type>;
  auto metadata = this->metadata;
  int N = 0;
  int numnnz = 0;
  if (metadata.my_rank == 0) {
    Assert(matrix.m() == matrix.n(), dealii::ExcNotQuadratic());
    N = matrix.m();
    numnnz = matrix.n_nonzero_elements();
  }
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&numnnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  global_matrix =
      mtx::create(settings.executor->get_master(), gko::dim<2>(N), numnnz);
  std::shared_ptr<mtx> global_matrix_compute;
  global_matrix_compute =
      mtx::create(settings.executor->get_master(), gko::dim<2>(N), numnnz);
  if (metadata.my_rank == 0) {
    // Needs to be a square matrix

    // TODO: Templatize using the the matrix type.
    // TODO: Maybe can make it a friend class of the deal.ii Matrix class to
    // avoid copies ?
    value_type *mat_values = global_matrix_compute->get_values();
    index_type *mat_row_ptrs = global_matrix_compute->get_row_ptrs();
    index_type *mat_col_idxs = global_matrix_compute->get_col_idxs();

    // copy over the data from the matrix to the data structures Schwarz
    // needs.
    //
    // final note: if the matrix has entries in the sparsity pattern that
    // are actually occupied by entries that have a zero numerical value,
    // then we keep them anyway. people are supposed to provide accurate
    // sparsity patterns.

    // first fill row lengths array
    mat_row_ptrs[0] = 0;
    for (index_type row = 1; row <= N; ++row) {
      mat_row_ptrs[row] =
          mat_row_ptrs[row - 1] + matrix.get_row_length(row - 1);
    }
    // then copy over matrix elements. note that for sparse matrices,
    // iterators are sorted so that they traverse each row from start to end
    // before moving on to the next row. however, this isn't true for block
    // matrices, so we have to do a bit of book keeping
    {
      // have an array that for each row points to the first entry not yet
      // written to
      std::vector<index_type> row_pointers(N + 1);
      std::copy(global_matrix_compute->get_row_ptrs(),
                global_matrix_compute->get_row_ptrs() + N + 1,
                row_pointers.begin());

      // loop over the elements of the matrix row by row, as suggested in
      // the documentation of the sparse matrix iterator class
      for (index_type row = 0; row < N; ++row) {
        for (typename dealii::SparseMatrix<ValueType>::const_iterator p =
                 matrix.begin(row);
             p != matrix.end(row); ++p) {
          // write entry into the first free one for this row
          mat_col_idxs[row_pointers[row]] = p->column();
          mat_values[row_pointers[row]] = p->value();

          // then move pointer ahead
          ++row_pointers[row];
        }
      }

      // at the end, we should have written all rows completely
      for (index_type i = 0; i < N - 1; ++i) {
        Assert(row_pointers[i] == mat_row_ptrs[i + 1],
               dealii::ExcInternalError());
      }
    }
  }
  auto mpi_itype =
      boost::mpi::get_mpi_datatype(global_matrix_compute->get_row_ptrs()[0]);
  auto mpi_vtype =
      boost::mpi::get_mpi_datatype(global_matrix_compute->get_values()[0]);
  MPI_Bcast(global_matrix_compute->get_row_ptrs(), N + 1, mpi_itype, 0,
            MPI_COMM_WORLD);
  MPI_Bcast(global_matrix_compute->get_col_idxs(), numnnz, mpi_itype, 0,
            MPI_COMM_WORLD);
  MPI_Bcast(global_matrix_compute->get_values(), numnnz, mpi_vtype, 0,
            MPI_COMM_WORLD);
  global_matrix->copy_from(global_matrix_compute.get());
}
#endif

template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::setup_global_matrix_laplacian(
    const IndexType &global_size,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix) {
  using index_type = IndexType;
  using value_type = ValueType;
  using mtx = gko::matrix::Csr<value_type, index_type>;
  auto metadata = this->metadata;

  global_matrix = mtx::create(settings.executor->get_master(),
                              gko::dim<2>(global_size), 5 * global_size);
  std::shared_ptr<mtx> global_matrix_compute;
  global_matrix_compute =
      mtx::create(settings.executor->get_master(), gko::dim<2>(global_size),
                  5 * global_size);
  value_type *val = global_matrix_compute->get_values();
  index_type *row = global_matrix_compute->get_row_ptrs();
  index_type *col = global_matrix_compute->get_col_idxs();

  index_type nx = std::sqrt(global_size);
  auto ny = nx;
  auto nnz = 0;
  row[0] = 0;
  for (auto i = 0; i < nx; i++) {
    /* diagonal */
    val[nnz] = 4.0;
    col[nnz] = i * ny;
    nnz++;
    /* next-neighbor */
    val[nnz] = -1.0;
    col[nnz] = i * ny + 1;
    nnz++;
    /* down-neighbor */
    if (i < nx - 1) {
      val[nnz] = -1.0;
      col[nnz] = i * ny + ny;
      nnz++;
    }
    /* up-neighbor */
    if (i > 0) {
      val[nnz] = -1.0;
      col[nnz] = i * ny - ny;
      nnz++;
    }
    /* update ptr */
    row[i * ny + 1] = nnz;
    for (auto j = 1; j < ny - 1; j++) {
      /* diagonal */
      val[nnz] = 4.0;
      col[nnz] = i * ny + j;
      nnz++;
      /* prev-neighbor */
      val[nnz] = -1.0;
      col[nnz] = i * ny + j - 1;
      nnz++;
      /* next-neighbor */
      val[nnz] = -1.0;
      col[nnz] = i * ny + j + 1;
      nnz++;
      /* down-neighbor */
      if (i < nx - 1) {
        val[nnz] = -1.0;
        col[nnz] = i * ny + j + ny;
        nnz++;
      }
      /* up-neighbor */
      if (i > 0) {
        val[nnz] = -1.0;
        col[nnz] = i * ny + j - ny;
        nnz++;
      }
      /* update ptr */
      row[i * ny + j + 1] = nnz;
    }
    /* diagonal */
    val[nnz] = 4.0;
    col[nnz] = i * ny + ny - 1;
    nnz++;
    /* prev-neighbor */
    val[nnz] = -1.0;
    col[nnz] = i * ny + ny - 2;
    nnz++;
    /* down-neighbor */
    if (i < nx - 1) {
      val[nnz] = -1.0;
      col[nnz] = i * ny + ny + ny - 1;
      nnz++;
    }
    /* up-neighbor */
    if (i > 0) {
      val[nnz] = -1.0;
      col[nnz] = i * ny - 1;
      nnz++;
    }
    /* update ptr */
    row[i * ny + ny] = nnz;
  }
  global_matrix->copy_from(global_matrix_compute.get());
}

template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::partition(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &global_matrix,
    std::vector<unsigned int> &partition_indices)

{
  partition_indices.resize(metadata.global_size);
  if (metadata.my_rank == 0) {
    auto partition_settings = (Settings::partition_settings::partition_zoltan |
                               Settings::partition_settings::partition_metis |
                               Settings::partition_settings::partition_naive |
                               Settings::partition_settings::partition_custom |
                               Settings::partition_settings::partition_auto) &
                              settings.partition;

    if (partition_settings == Settings::partition_settings::partition_zoltan) {
      // #  ifndef DEAL_II_TRILINOS_WITH_ZOLTAN
      //           AssertThrow(false,
      //                       dealii::ExcMessage(
      //                                  "Choosing 'partition_zoltan'
      //                                  requires the library " "to be
      //                                  compiled with support for
      //                                  Zoltan! " "Instead, you might
      //                                  use 'partition_auto' to select
      //                                  " "a partitioning algorithm
      //                                  that is supported " "by your
      //                                  current configuration."));
      // #  else
      //           PartitionTools::partition_zoltan(this->global_matrix,
      //           this->cell_weights,
      //                                         this->num_subdomains,
      //                                         this->partition_indices);
      // #  endif
    } else if (partition_settings ==
               Settings::partition_settings::partition_metis) {
      PartitionTools::PartitionMetis(global_matrix, this->cell_weights,
                                     metadata.num_subdomains,
                                     partition_indices);

      // #  endif
    } else if (partition_settings ==
               Settings::partition_settings::partition_naive) {
      PartitionTools::PartitionNaive(global_matrix, metadata.num_subdomains,
                                     partition_indices);
    } else if (partition_settings ==
               Settings::partition_settings::partition_custom) {
      // User partitions mesh manually
    } else {
    }
  }
}

template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::setup_vectors(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::vector<ValueType> &rhs,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_rhs,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution) {
  using vec = gko::matrix::Dense<ValueType>;
  auto my_rank = metadata.my_rank;
  auto first_row = metadata.first_row->get_data()[my_rank];
  auto global_size = metadata.global_size;
  auto local_size_x = metadata.local_size_x;

  gko::Array<ValueType> temp_rhs{settings.executor->get_master(), rhs.begin(),
                                 rhs.end()};
  global_rhs =
      vec::create(settings.executor, gko::dim<2>{global_size, 1}, temp_rhs, 1);
  global_solution =
      vec::create(settings.executor->get_master(), gko::dim<2>(global_size, 1));

  local_rhs = vec::create(settings.executor, gko::dim<2>(local_size_x, 1));

  SolverTools::extract_local_vector(settings, metadata, local_rhs, global_rhs,
                                    first_row);

  local_solution = vec::create(settings.executor, gko::dim<2>(local_size_x, 1));
}

#define DECLARE_INITIALIZE(ValueType, IndexType)                               \
  class Initialize<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_INITIALIZE);
#undef DECLARE_INITIALIZE

} // namespace SchwarzWrappers
