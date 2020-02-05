

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


#include <schwarz/config.hpp>


#include <map>
#include <vector>


#include <mpi.h>


#include <exception_helpers.hpp>
#include <initialization.hpp>
#include <solver_tools.hpp>


namespace SchwarzWrappers {


template <typename ValueType, typename IndexType>
Initialize<ValueType, IndexType>::Initialize(
    Settings &settings, Metadata<ValueType, IndexType> &metadata)
    : settings(settings), metadata(metadata)
{
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
void Initialize<ValueType, IndexType>::generate_rhs(std::vector<ValueType> &rhs)
{
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::default_random_engine engine;
    for (gko::size_type i = 0; i < rhs.size(); ++i) {
        rhs[i] = unif(engine);
    }
}


#if SCHW_HAVE_DEALII


template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::setup_global_matrix(
    const dealii::SparseMatrix<ValueType> &matrix,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix)
{
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
                for (typename dealii::SparseMatrix<ValueType>::const_iterator
                         p = matrix.begin(row);
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


inline gko::size_type linearize_index(const gko::size_type row,
                                      const gko::size_type col,
                                      const gko::size_type num_rows)
{
    return (row)*num_rows + col;
}


template <typename ValueType, typename IndexType>
void Initialize<ValueType, IndexType>::setup_global_matrix_laplacian(
    const gko::size_type &oned_laplacian_size,
    std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &global_matrix)
{
    using index_type = IndexType;
    using value_type = ValueType;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    gko::size_type global_size = oned_laplacian_size * oned_laplacian_size;

    global_matrix = mtx::create(settings.executor->get_master(),
                                gko::dim<2>(global_size), 5 * global_size);
    value_type *values = global_matrix->get_values();
    index_type *row_ptrs = global_matrix->get_row_ptrs();
    index_type *col_idxs = global_matrix->get_col_idxs();

    std::vector<gko::size_type> exclusion_set;

    std::map<IndexType, ValueType> stencil_map = {
        {-oned_laplacian_size, -1}, {-1, -1}, {0, 4}, {1, -1},
        {oned_laplacian_size, -1},
    };
    for (auto i = 2; i < global_size; ++i) {
        gko::size_type index = (i - 1) * oned_laplacian_size;
        if (index * index < global_size * global_size) {
            exclusion_set.push_back(
                linearize_index(index, index - 1, global_size));
            exclusion_set.push_back(
                linearize_index(index - 1, index, global_size));
        }
    }

    std::sort(exclusion_set.begin(),
              exclusion_set.begin() + exclusion_set.size());

    IndexType pos = 0;
    IndexType col_idx = 0;
    row_ptrs[0] = pos;
    gko::size_type cur_idx = 0;
    for (IndexType i = 0; i < global_size; ++i) {
        for (auto ofs : stencil_map) {
            auto in_exclusion_flag =
                (exclusion_set[cur_idx] ==
                 linearize_index(i, i + ofs.first, global_size));
            if (0 <= i + ofs.first && i + ofs.first < global_size &&
                !in_exclusion_flag) {
                values[pos] = ofs.second;
                col_idxs[pos] = i + ofs.first;
                ++pos;
            }
            if (in_exclusion_flag) {
                cur_idx++;
            }
            col_idx = row_ptrs[i + 1] - pos;
        }
        row_ptrs[i + 1] = pos;
    }
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
        auto partition_settings =
            (Settings::partition_settings::partition_zoltan |
             Settings::partition_settings::partition_metis |
             Settings::partition_settings::partition_regular |
             Settings::partition_settings::partition_regular2d |
             Settings::partition_settings::partition_custom) &
            settings.partition;

        if (partition_settings ==
            Settings::partition_settings::partition_zoltan) {
            SCHWARZ_NOT_IMPLEMENTED;
        } else if (partition_settings ==
                   Settings::partition_settings::partition_metis) {
            if (metadata.my_rank == 0) {
                std::cout << " METIS partition" << std::endl;
            }
            PartitionTools::PartitionMetis(
                settings, global_matrix, this->cell_weights,
                metadata.num_subdomains, partition_indices);
        } else if (partition_settings ==
                   Settings::partition_settings::partition_regular) {
            if (metadata.my_rank == 0) {
                std::cout << " Regular 1D partition" << std::endl;
            }
            PartitionTools::PartitionRegular(
                global_matrix, metadata.num_subdomains, partition_indices);
        } else if (partition_settings ==
                   Settings::partition_settings::partition_regular2d) {
            if (metadata.my_rank == 0) {
                std::cout << " Regular 2D partition" << std::endl;
            }
            PartitionTools::PartitionRegular2D(
                global_matrix, settings.write_debug_out,
                metadata.num_subdomains, partition_indices);
        } else if (partition_settings ==
                   Settings::partition_settings::partition_custom) {
            // User partitions mesh manually
            SCHWARZ_NOT_IMPLEMENTED;
        } else {
            SCHWARZ_NOT_IMPLEMENTED;
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
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_last_solution,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &global_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto my_rank = metadata.my_rank;
    auto first_row = metadata.first_row->get_data()[my_rank];

    // Copy the global rhs vector to the required executor.
    gko::Array<ValueType> temp_rhs{settings.executor->get_master(), rhs.begin(),
                                   rhs.end()};
    global_rhs = vec::create(settings.executor,
                             gko::dim<2>{metadata.global_size, 1}, temp_rhs, 1);
    global_solution = vec::create(settings.executor->get_master(),
                                  gko::dim<2>(metadata.global_size, 1));

    local_rhs =
        vec::create(settings.executor, gko::dim<2>(metadata.local_size_x, 1));
    // Extract the local rhs from the global rhs. Also takes into account the
    // overlap.
    SolverTools::extract_local_vector(settings, metadata, local_rhs, global_rhs,
                                      first_row);

    local_solution =
        vec::create(settings.executor, gko::dim<2>(metadata.local_size_x, 1));
    
    //contains the solution at the last event of communication
    local_last_solution =
        vec::create(settings.executor, gko::dim<2>(metadata.local_size_x, 1));
}


#define DECLARE_INITIALIZE(ValueType, IndexType) \
    class Initialize<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_INITIALIZE);
#undef DECLARE_INITIALIZE


}  // namespace SchwarzWrappers
