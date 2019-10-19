
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

#ifndef exception_helpers_hpp
#define exception_helpers_hpp


#include <exception.hpp>

/**
 * Asserts that a module has not yet been implemented.
 *
 * @param _cuda_call  a library call expression
 */
#define SCHWARZ_MODULE_NOT_IMPLEMENTED(module_)                              \
    {                                                                        \
        throw ::ModuleNotImplemented(__FILE__, __LINE__, module_, __func__); \
    }

/**
 * Asserts that a feature has not yet been implemented.
 *
 * @param _cuda_call  a library call expression
 */
#define SCHWARZ_NOT_IMPLEMENTED                               \
    {                                                         \
        throw ::NotImplemented(__FILE__, __LINE__, __func__); \
    }

/**
 *Asserts that _val1 and _val2 are equal.
 *
 *@throw BadDimension if _val1 is different from _val2.
 */
#define SCHWARZ_ASSERT_EQ(_val1, _val2)                                      \
    if (_val1 != _val2) {                                                    \
        throw ::BadDimension(__FILE__, __LINE__, __func__, " Value ", _val1, \
                             _val2, "expected equal values");                \
    }

/**
 * Asserts that a cuSPARSE library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(_cusparse_call)                  \
    do {                                                                   \
        auto _errcode = _cusparse_call;                                    \
        if (_errcode != CUSPARSE_STATUS_SUCCESS) {                         \
            throw ::CusparseError(__FILE__, __LINE__, __func__, _errcode); \
        }                                                                  \
    } while (false)


/**
 * Asserts that a CUDA library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define SCHWARZ_ASSERT_NO_CUDA_ERRORS(_cuda_call)                      \
    do {                                                               \
        auto _errcode = _cuda_call;                                    \
        if (_errcode != cudaSuccess) {                                 \
            throw ::CudaError(__FILE__, __LINE__, __func__, _errcode); \
        }                                                              \
    } while (false)


/**
 * Asserts that a METIS library call completed without errors.
 *
 * @param _metis_call  a library call expression
 */
#define SCHWARZ_ASSERT_NO_METIS_ERRORS(_metis_call)                     \
    do {                                                                \
        auto _errcode = _metis_call;                                    \
        if (_errcode != METIS_OK) {                                     \
            throw ::MetisError(__FILE__, __LINE__, __func__, _errcode); \
        }                                                               \
    } while (false)


#endif  // exception_helpers.hpp
