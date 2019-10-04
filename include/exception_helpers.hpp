#ifndef exception_helpers_hpp
#define exception_helpers_hpp


#include <exception.hpp>


/**
 * Asserts that a feature or module has not yet been imeplemented.
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
#define SCHWARZ_ASSERT_EQ(_val1, _val2)                                     \
  if (_val1 != _val2) {                                                 \
    throw ::BadDimension(__FILE__, __LINE__, __func__, " Value ",  \
                              _val1, _val2, "expected equal values");   \
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
