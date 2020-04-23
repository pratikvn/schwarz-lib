
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


#include <utils.hpp>


namespace schwz {

template <typename ValueType, typename IndexType>
int Utils<ValueType, IndexType>::get_local_rank(MPI_Comm mpi_communicator)
{
    MPI_Comm local_comm;
    int rank;
    MPI_Comm_split_type(mpi_communicator, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &rank);
    return rank;
}


template <typename ValueType, typename IndexType>
bool Utils<ValueType, IndexType>::check_subd_locality(MPI_Comm mpi_communicator,
                                                      int neighbor_rank,
                                                      int my_rank)
{
    MPI_Comm local_comm;
    int local_nprocs = 0;
    MPI_Comm_split_type(mpi_communicator, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_size(local_comm, &local_nprocs);
    if (std::abs((neighbor_rank - my_rank)) < local_nprocs)
        return true;
    else
        return false;
}


template <typename ValueType, typename IndexType>
int Utils<ValueType, IndexType>::get_local_num_procs(MPI_Comm mpi_communicator)
{
    MPI_Comm local_comm;
    int local_num_procs;
    MPI_Comm_split_type(mpi_communicator, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_size(local_comm, &local_num_procs);
    return local_num_procs;
}


template <typename ValueType, typename IndexType>
void Utils<ValueType, IndexType>::print_vector(
    const gko::matrix::Dense<ValueType> *vector, int rank, std::string name)
{
    std::cout << "from rank" << rank << " , " << name << " :[ \n" << std::endl;
    for (auto i = 0; i < vector->get_size()[0]; ++i) {
        std::cout << vector->get_const_values()[i] << "\n";
    }
    std::cout << "]" << std::endl;
}


template <typename ValueType, typename IndexType>
void Utils<ValueType, IndexType>::print_matrix(
    const gko::matrix::Csr<ValueType, IndexType> *matrix, int rank,
    std::string name)
{
    std::ofstream file;
    file.open(name + "_" + std::to_string(rank) + ".csv");
    for (auto row = 0; row < matrix->get_size()[0]; row++) {
        for (auto col = matrix->get_const_row_ptrs()[row];
             col < matrix->get_const_row_ptrs()[row + 1]; col++) {
            file << row + 1 << "," << matrix->get_const_col_idxs()[col] + 1
                 << "," << matrix->get_const_values()[col] << "\n";
        }
    }
    file.close();
}


template <typename ValueType, typename IndexType>
int Utils<ValueType, IndexType>::find_duplicates(IndexType val,
                                                 std::size_t index,
                                                 const IndexType *data,
                                                 std::size_t length)
{
    auto count = 0;
    for (auto i = 0; i < length; ++i) {
        if (i != index && val == data[i]) {
            count++;
        }
    }
    return count;
}


template <typename ValueType, typename IndexType>
bool Utils<ValueType, IndexType>::assert_correct_permutation(
    const gko::matrix::Permutation<IndexType> *input_perm)
{
    auto perm_data = input_perm->get_const_permutation();
    auto perm_size = input_perm->get_permutation_size();

    for (auto i = 0; i < perm_size; ++i) {
        if (perm_data[i] >= perm_size) {
            std::cout << "Here " << __LINE__ << ", perm[i] " << perm_data[i]
                      << " at " << i << std::endl;
            return false;
        }
        if (perm_data[i] < 0) {
            std::cout << "Here " << __LINE__ << std::endl;
            return false;
        }
        auto duplicates = Utils<ValueType, IndexType>::find_duplicates(
            perm_data[i], i, perm_data, perm_size);
        if (duplicates > 0) {
            std::cout << "Here " << __LINE__ << " " << duplicates << std::endl;
            return false;
        }
    }
    return true;
}


template <typename ValueType, typename IndexType>
bool Utils<ValueType, IndexType>::assert_correct_cuda_devices(int num_devices,
                                                              int my_rank)
{
    if (num_devices > 0) {
        if (my_rank == 0) {
            std::cout << " Number of available devices: " << num_devices
                      << std::endl;
        }
    } else {
        std::cout << " No CUDA devices available for rank " << my_rank
                  << std::endl;
        std::exit(-1);
    }
}


#define DECLARE_UTILS(ValueType, IndexType) struct Utils<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_UTILS);
#undef DECLARE_UTILS


}  // namespace schwz
