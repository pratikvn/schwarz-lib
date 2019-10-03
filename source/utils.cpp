
#include <utils.hpp>


namespace SchwarzWrappers {

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


#define DECLARE_UTILS(ValueType, IndexType) struct Utils<ValueType, IndexType>
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_UTILS);
#undef DECLARE_UTILS


}  // namespace SchwarzWrappers
