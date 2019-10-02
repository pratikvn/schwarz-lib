#ifndef cusparse_helpers_hpp
#define cusparse_helpers_hpp

#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <exception_helpers.hpp>
#include <settings.hpp>
#include <solve.hpp>

namespace SchwarzWrappers {
namespace CusparseWrappers {
template <typename ValueType, typename IndexType>
void initialize(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    struct Solve<ValueType, IndexType>::cusparse &cusparse,
    const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
        &triangular_factor,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution) {
  auto handle = (static_cast<gko::CudaExecutor *>(settings.executor.get()))
                    ->get_cusparse_handle();
  auto num_rows = triangular_factor->get_size()[0];
  auto sol_size = local_solution->get_size()[0];
  auto num_rhs = local_solution->get_size()[1];
  auto factor_nnz =
      triangular_factor
          ->get_num_stored_elements(); // Check if this is actually equal to nnz
  auto row_ptrs = triangular_factor->get_const_row_ptrs();
  auto col_idxs = triangular_factor->get_const_col_idxs();
  auto factor_values = triangular_factor->get_const_values();
  auto sol_values = local_solution->get_values();
  auto one = 1.0;

  cusparse.policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  cusparse.algo = 0;
  cusparse.info = NULL;
  cusparse.L_factor_descr = NULL;
  cusparse.L_factor_info = NULL;
  cusparse.L_factor_work_size = 0;
  cusparse.U_factor_descr = NULL;
  cusparse.U_factor_info = NULL;
  cusparse.U_factor_work_size = 0;
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle));
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseCreateSolveAnalysisInfo(&cusparse.info));

  /*  configuration of matrices */
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseCreateCsrsm2Info(&cusparse.L_factor_info));
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseCreateMatDescr(&cusparse.L_factor_descr));
  cusparseSetMatIndexBase(cusparse.L_factor_descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(cusparse.L_factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(cusparse.L_factor_descr, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(cusparse.L_factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseCreateCsrsm2Info(&cusparse.U_factor_info));
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseCreateMatDescr(&cusparse.U_factor_descr));
  cusparseSetMatIndexBase(cusparse.U_factor_descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(cusparse.U_factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(cusparse.U_factor_descr, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(cusparse.U_factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrsm2_bufferSizeExt(
      handle, cusparse.algo, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rhs, factor_nnz, &one,
      cusparse.U_factor_descr, // descriptor
      factor_values, row_ptrs, col_idxs, sol_values, sol_size,
      cusparse.U_factor_info, cusparse.policy, &cusparse.U_factor_work_size));
  // > L solve
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrsm2_bufferSizeExt(
      handle, cusparse.algo, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rhs, factor_nnz, &one,
      cusparse.L_factor_descr, // descriptor
      factor_values, row_ptrs, col_idxs, sol_values, sol_size,
      cusparse.L_factor_info, cusparse.policy, &cusparse.L_factor_work_size));

  // size_t lwork = (gpu_struct->lwork_L > gpu_struct->lwork_U ?
  // gpu_struct->lwork_L : gpu_struct->lwork_U); gpu_struct->lwork_L =
  // lwork; gpu_struct->lwork_U = lwork;

  // allocate workspace
  if (cusparse.L_factor_work_vec != nullptr) {
    cudaFree(cusparse.L_factor_work_vec);
  }
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaMalloc((void **)&cusparse.L_factor_work_vec,
                                           cusparse.L_factor_work_size));

  if (cusparse.U_factor_work_vec != nullptr) {
    cudaFree(cusparse.U_factor_work_vec);
  }
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaMalloc((void **)&cusparse.U_factor_work_vec,
                                           cusparse.U_factor_work_size));

  // Analyze U solve.
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrsm2_analysis(
      handle, cusparse.algo, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rhs, factor_nnz, &one,
      cusparse.U_factor_descr, // descriptor
      factor_values, row_ptrs, col_idxs, sol_values, sol_size,
      cusparse.U_factor_info, cusparse.policy, cusparse.U_factor_work_vec));
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());

  // Analyze L solve.
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrsm2_analysis(
      handle, cusparse.algo, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rhs, factor_nnz, &one,
      cusparse.L_factor_descr, // descriptor
      factor_values, row_ptrs, col_idxs, sol_values, sol_size,
      cusparse.L_factor_info, cusparse.policy, cusparse.L_factor_work_vec));
  SCHWARZ_ASSERT_NO_CUDA_ERRORS(cudaDeviceSynchronize());
}

template <typename ValueType, typename IndexType>
void solve(const Settings &settings,
           const Metadata<ValueType, IndexType> &metadata,
           struct Solve<ValueType, IndexType>::cusparse &cusparse,
           const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>
               &triangular_factor,
           std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution) {
  auto handle = (static_cast<gko::CudaExecutor *>(settings.executor.get()))
                    ->get_cusparse_handle();
  // cusparseOperation_t t_flag;

  // if (transpose_flag == "upper")
  //   {
  //     t_flag = CUSPARSE_OPERATION_NON_TRANSPOSE;

  //   }
  // else if (transpose_flag == "lower")
  //   {
  //     t_flag = CUSPARSE_OPERATION_TRANSPOSE;
  //   }
  // else
  //   {
  //     std::cout
  //       << " transpose flag needs to be non-transpose or transpose, Check
  //       the calling function"
  //       << std::endl;
  //   }
  auto num_rows = triangular_factor->get_size()[0];
  auto sol_size = local_solution->get_size()[0];
  auto num_rhs = local_solution->get_size()[1];
  auto factor_nnz =
      triangular_factor
          ->get_num_stored_elements(); // Check if this is actually equal to nnz
  auto row_ptrs = triangular_factor->get_const_row_ptrs();
  auto col_idxs = triangular_factor->get_const_col_idxs();
  auto factor_values = triangular_factor->get_const_values();
  auto sol_values = local_solution->get_values();
  auto one = 1.0;
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrsm2_solve(
      handle, cusparse.algo, CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rhs, factor_nnz, &one,
      cusparse.L_factor_descr, factor_values, row_ptrs, col_idxs, sol_values,
      sol_size, cusparse.L_factor_info, cusparse.policy,
      cusparse.L_factor_work_vec));

  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrsm2_solve(
      handle, cusparse.algo, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rhs, factor_nnz, &one,
      cusparse.U_factor_descr, factor_values, row_ptrs, col_idxs, sol_values,
      sol_size, cusparse.U_factor_info, cusparse.policy,
      cusparse.U_factor_work_vec));
}

// Weird bug with template parameter fro cusparse struct
// being asked for class template instead of a basic typename.
template <typename ValueType, typename IndexType>
void clear(const Settings &settings,
           const Metadata<ValueType, IndexType> &metadata,
           struct Solve<ValueType, IndexType>::cusparse &cusparse) {
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsrsm2Info(cusparse.info));
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseDestroyMatDescr(cusparse.L_factor_descr));
  SCHWARZ_ASSERT_NO_CUSPARSE_ERRORS(
      cusparseDestroyMatDescr(cusparse.U_factor_descr));
  cusparse.algo = 0;
  cusparse.L_factor_info = NULL;
  cusparse.L_factor_work_size = 0;
  cusparse.U_factor_info = NULL;
  cusparse.U_factor_work_size = 0;
}

} // namespace CusparseWrappers

// Explicit Instantiations
#define DECLARE_FUNCTION(ValueType, IndexType)                                 \
  void CusparseWrappers::initialize(                                           \
      const Settings &, const Metadata<ValueType, IndexType> &,                \
      struct Solve<ValueType, IndexType>::cusparse &,                          \
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &,         \
      std::shared_ptr<gko::matrix::Dense<ValueType>> &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_FUNCTION(ValueType, IndexType)                                 \
  void CusparseWrappers::solve(                                                \
      const Settings &, const Metadata<ValueType, IndexType> &,                \
      struct Solve<ValueType, IndexType>::cusparse &,                          \
      const std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> &,         \
      std::shared_ptr<gko::matrix::Dense<ValueType>> &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_FUNCTION(ValueType, IndexType)                                 \
  void CusparseWrappers::clear(                                                \
      const Settings &, const Metadata<ValueType, IndexType> &,                \
      struct Solve<ValueType, IndexType>::cusparse &);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION
} // namespace SchwarzWrappers

#endif
