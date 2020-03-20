
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


#ifndef utils_hpp
#define utils_hpp


#include <fstream>
#include <iostream>
#include <memory>
#include <string>


#include <mpi.h>
#include <ginkgo/ginkgo.hpp>


#include <settings.hpp>


namespace SchwarzWrappers {


/**
 * The utilities class which provides some checks and basic utilities.
 *
 * @tparam ValueType  The type of the floating point values.
 * @tparam IndexType  The type of the index type values.
 *
 * @ref utils
 * @ingroup utils
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
struct Utils {
    Utils() = default;

    static int get_local_rank(MPI_Comm mpi_communicator);

    static int get_local_num_procs(MPI_Comm mpi_communicator);

    static bool check_subd_locality(MPI_Comm mpi_communicator,
                                    int neighbor_rank, int my_rank);

    static void print_matrix(
        const gko::matrix::Csr<ValueType, IndexType> *matrix, int rank,
        std::string name);

    static void print_vector(const gko::matrix::Dense<ValueType> *vector,
                             int rank, std::string name);

    static int find_duplicates(IndexType val, std::size_t index,
                               const IndexType *data, std::size_t length);

    static bool assert_correct_permutation(
        const gko::matrix::Permutation<IndexType> *input_perm);
};


}  // namespace SchwarzWrappers


#endif  // utils.hpp
