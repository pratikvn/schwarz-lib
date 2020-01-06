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


#include <ginkgo/ginkgo.hpp>
#include <settings.hpp>


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
void PartitionNaive2D(
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &global_matrix,
    bool write_debug_out, const unsigned int &n_partitions,
    std::vector<unsigned int> &partition_indices)
{
    auto n = global_matrix->get_size()[0];
    int sq_n = static_cast<int>(std::sqrt(n));
    int sq_partn = static_cast<int>(std::sqrt(n_partitions));
    auto offset1 = 0;
    auto offset2 = 0;
    int subd = 0;
    for (auto j1 = 0; j1 < sq_partn; ++j1) {
        offset2 = j1 * sq_partn * std::pow(sq_n / sq_partn, 2);
        for (auto j2 = 0; j2 < sq_partn; ++j2) {
            auto my_id = sq_partn * j1 + j2;
            for (auto i1 = 0; i1 < sq_n / sq_partn; ++i1) {
                offset1 = (j2)*sq_n / sq_partn;
                for (auto i2 = 0; i2 < sq_n / sq_partn; ++i2) {
                    partition_indices[offset2 + offset1 +
                                      (2 * i1 * sq_n / sq_partn) + i2] = my_id;
                }
            }
        }
    }

    if (write_debug_out) {
        std::ofstream file;
        std::string filename = "part_indices.csv";
        file.open(filename);
        file << "idx,subd\n";
        for (auto i = 0; i < partition_indices.size(); ++i) {
            file << i << "," << partition_indices[i] << "\n";
        }
        file.close();
    }
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
    // generate non-sensical output (everything is owned by a single
    // process) and complain with a message (but won't return an error
    // code!):
    // ***Cannot bisect a graph with 0 vertices!
    // ***You are trying to partition a graph into too many parts!
    nparts = std::min(n, nparts);

    // use default options for METIS
    idx_t options[METIS_NOPTIONS];
    SCHWARZ_ASSERT_NO_METIS_ERRORS(METIS_SetDefaultOptions(options));
    // options[METIS_OPTION_SEED]      = 0;
    // options[METIS_OPTION_NUMBERING] = 0;

    // // one more nuisance: we have to copy our own data to arrays that
    // store
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
    // METIS expects a null pointer if there are no weights to be
    // considered.
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
