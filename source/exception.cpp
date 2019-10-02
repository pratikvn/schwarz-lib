#include <exception.hpp>

#include <schwarz/config.hpp>

#if SCHWARZ_BUILD_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#if SCHWARZ_BUILD_CUDA

std::string CudaError::get_error(int error_code) {
  std::string name = cudaGetErrorName(static_cast<cudaError>(error_code));
  std::string message = cudaGetErrorString(static_cast<cudaError>(error_code));
  return name + ": " + message;
}

std::string CusparseError::get_error(int error_code) {
#define SCHWARZ_REGISTER_CUSPARSE_ERROR(error_name)                            \
  if (error_code == int(error_name)) {                                         \
    return #error_name;                                                        \
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
}

#else

std::string CudaError::get_error(int error_code) {
  return "CUDA not being built.";
}

std::string CusparseError::get_error(int error_code) {
  return "CUDA not being built.";
}
#endif
