
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


#if SCHW_HAVE_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif


#if SCHW_HAVE_METIS
#include <metis.h>
#endif


#include <exception.hpp>


std::string CudaError::get_error(int error_code)
{
#if SCHW_HAVE_CUDA
    std::string name = cudaGetErrorName(static_cast<cudaError>(error_code));
    std::string message =
        cudaGetErrorString(static_cast<cudaError>(error_code));
    return name + ": " + message;
#else
    return "CUDA not being built.";
#endif
}


std::string CusparseError::get_error(int error_code)
{
#if SCHW_HAVE_CUDA
#define SCHWARZ_REGISTER_CUSPARSE_ERROR(error_name) \
    if (error_code == int(error_name)) {            \
        return #error_name;                         \
    }
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_SUCCESS);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_NOT_INITIALIZED);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_ALLOC_FAILED);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_INVALID_VALUE);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_ARCH_MISMATCH);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_MAPPING_ERROR);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_EXECUTION_FAILED);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_INTERNAL_ERROR);
    SCHWARZ_REGISTER_CUSPARSE_ERROR(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    return "Unknown error";
#else
    return "CUDA not being built.";
#endif
}


std::string MetisError::get_error(int error_code)
{
#if SCHW_HAVE_METIS
#define GKO_REGISTER_METIS_ERROR(error_name)          \
    if (error_code == static_cast<int>(error_name)) { \
        return #error_name;                           \
    }
    GKO_REGISTER_METIS_ERROR(METIS_ERROR_INPUT);
    GKO_REGISTER_METIS_ERROR(METIS_ERROR_MEMORY);
    GKO_REGISTER_METIS_ERROR(METIS_ERROR);
    return "Unknown error";
#else
    return "Metis is not linked/built";
#endif
}
