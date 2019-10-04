#ifndef partition_tools_hpp
#define partition_tools_hpp


#include <memory>
#include <vector>


#if SCHW_HAVE_CHOLMOD
#include <cholmod.h>
#endif


#if SCHW_HAVE_METIS
#include <metis.h>
#endif


#include <settings.hpp>
#include <ginkgo/ginkgo.hpp>


namespace SchwarzWrappers {
namespace PartitionTools {


template <typename ValueType, typename IndexType>
void PartitionNaive(
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &global_matrix,
    const unsigned int &n_partitions,
    std::vector<unsigned int> &partition_indices)
{
    // TODO: Move the naive partitioning here from initialization
}


template <typename ValueType, typename IndexType>
void PartitionMetis(
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &global_matrix,
    const std::vector<unsigned int> &cell_weights,
    const unsigned int &n_partitions,
    std::vector<unsigned int> &partition_indices)
{
#if SCHW_HAVE_METIS
    // generate the data structures for
    // METIS. Note that this is particularly
    // simple, since METIS wants exactly our
    // compressed row storage format. we only
    // have to set up a few auxiliary arrays
    idx_t n = static_cast<signed int>(global_matrix->get_size()[0]),
          ncon = 1,  // number of balancing constraints (should be >0)
        nparts =
            static_cast<int>(n_partitions),  // number of subdomains to create
        dummy;                               // the numbers of edges cut by the
    // resulting partition

    // We can not partition n items into more than n parts. METIS will
    // generate non-sensical output (everything is owned by a single process)
    // and complain with a message (but won't return an error code!):
    // ***Cannot bisect a graph with 0 vertices!
    // ***You are trying to partition a graph into too many parts!
    nparts = std::min(n, nparts);

    // use default options for METIS
    idx_t options[METIS_NOPTIONS];
    SCHWARZ_ASSERT_NO_METIS_ERRORS(METIS_SetDefaultOptions(options));
    // options[METIS_OPTION_SEED]      = 0;
    // options[METIS_OPTION_NUMBERING] = 0;

    // // one more nuisance: we have to copy our own data to arrays that store
    // // signed integers :-(
    std::vector<idx_t> int_rowstart(n + 1);
    std::vector<idx_t> int_colnums(global_matrix->get_num_stored_elements());
    // TODO: need to remove this ? OPT
    for (auto i = 0; i < n + 1; ++i) {
        int_rowstart[i] = static_cast<idx_t>(global_matrix->get_row_ptrs()[i]);
    }
    for (gko::size_type i = 0; i < global_matrix->get_num_stored_elements();
         ++i) {
        int_colnums[i] = static_cast<idx_t>(global_matrix->get_col_idxs()[i]);
    }

    std::vector<idx_t> int_partition_indices(global_matrix->get_size()[0]);

    // Setup cell weighting option
    std::vector<idx_t> int_cell_weights;
    if (cell_weights.size() > 0) {
        SCHWARZ_ASSERT_EQ(cell_weights.size(), global_matrix->get_size()[0]);
        int_cell_weights.resize(cell_weights.size());
        std::copy(cell_weights.begin(), cell_weights.end(),
                  int_cell_weights.begin());
    }
    // Set a pointer to the optional cell weighting information.
    // METIS expects a null pointer if there are no weights to be considered.
    idx_t *const p_int_cell_weights =
        (cell_weights.size() > 0 ? int_cell_weights.data() : nullptr);


    // Use recursive if the number of partitions is less than or equal to 8
    if (nparts <= 8) {
        SCHWARZ_ASSERT_NO_METIS_ERRORS(METIS_PartGraphRecursive(
            &n, &ncon, int_rowstart.data(), int_colnums.data(),
            p_int_cell_weights, nullptr, nullptr, &nparts, nullptr, nullptr,
            options, &dummy, int_partition_indices.data()));
    }
    // Otherwise use kway
    else {
        SCHWARZ_ASSERT_NO_METIS_ERRORS(METIS_PartGraphKway(
            &n, &ncon, int_rowstart.data(), int_colnums.data(),
            p_int_cell_weights, nullptr, nullptr, &nparts, nullptr, nullptr,
            options, &dummy, int_partition_indices.data()));
    }

    // now copy back generated indices into the output array
    std::copy(int_partition_indices.begin(), int_partition_indices.end(),
              partition_indices.begin());
#endif
}


#define DECLARE_FUNCTION(ValueType, IndexType)                           \
    void PartitionMetis(                                                 \
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &, \
        const std::vector<unsigned int> &, const unsigned int &,         \
        std::vector<unsigned int> &)
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_FUNCTION(ValueType, IndexType)                           \
    void PartitionNaive(                                                 \
        const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &, \
        const unsigned int &, std::vector<unsigned int> &)
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

}  // namespace PartitionTools

}  // namespace SchwarzWrappers

#endif
