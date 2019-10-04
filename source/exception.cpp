
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
