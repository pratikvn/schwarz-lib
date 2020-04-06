
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

// This is a modified version of the topology example given in
// the hwloc documentation
// https://www.open-mpi.org/projects/hwloc/doc/v2.0.4/a00312.php#interface

#ifndef process_topology_hpp
#define process_topology_hpp


#include <mpi.h>
#include <omp.h>


#include <ginkgo/ginkgo.hpp>
#include <schwarz/config.hpp>


#if SCHW_HAVE_CUDA
#include <cuda_runtime.h>
#endif


#include "device_guard.hpp"


/**
 * @brief The ProcessTopology namespace .
 * @ref proc_topo
 * @ingroup init
 * @ingroup comm
 */
namespace ProcessTopology {


static void bind_gpus_to_process(
    std::shared_ptr<schwz::device_guard> &dev_guard, int &local_rank,
    int &local_num_procs, int &num_threads)
{
    // SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaSetDevice(local_rank));
    // SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaGetLastError());
}


}  // namespace ProcessTopology


#endif  // process_topology.hpp
