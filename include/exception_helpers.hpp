#ifndef exception_helpers_hpp
#define exception_helpers_hpp

#include <exception.hpp>

#define SCHWARZ_NOT_IMPLEMENTED                                                \
  { throw ::NotImplemented(__FILE__, __LINE__, __func__); }

/**
 * Asserts that a cuSPARSE library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(_cusparse_call)                      \
  do {                                                                         \
    auto _errcode = _cusparse_call;                                            \
    if (_errcode != CUSPARSE_STATUS_SUCCESS) {                                 \
      throw ::CusparseError(__FILE__, __LINE__, __func__, _errcode);           \
    }                                                                          \
  } while (false)

/**
 * Asserts that a CUDA library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define SCHWARZ_ASSERT_NO_CUDA_ERRORS(_cuda_call)                              \
  do {                                                                         \
    auto _errcode = _cuda_call;                                                \
    if (_errcode != cudaSuccess) {                                             \
      throw ::CudaError(__FILE__, __LINE__, __func__, _errcode);               \
    }                                                                          \
  } while (false)

#endif
/*----------------------------   exception_helpers.hpp
 * ---------------------------*/
