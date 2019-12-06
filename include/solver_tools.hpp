#ifndef solver_tools_hpp
#define solver_tools_hpp


#include <memory>
#include <vector>


#include <schwarz/config.hpp>
#include <settings.hpp>


#if SCHWARZ_BUILD_CHOLMOD
#include <cholmod.h>
#endif


namespace SchwarzWrappers {
namespace SolverTools {


#if SCHW_HAVE_CHOLMOD
template <typename ValueType, typename IndexType>
void solve_direct_cholmod(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    cholmod_common &ch_settings, cholmod_factor *L_factor, cholmod_dense *rhs,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto local_n = metadata.local_size_x;
    rhs->x = local_solution->get_values();

    cholmod_dense *sol = cholmod_solve(CHOLMOD_A, L_factor, rhs, &ch_settings);
    auto temp =
        vec::create(settings.executor->get_master(),
                    gko::dim<2>(local_solution->get_size()[0], 1),
                    gko::Array<ValueType>::view(settings.executor->get_master(),
                                                local_solution->get_size()[0],
                                                (ValueType *)(sol->x)),
                    1);

    local_solution->copy_from(gko::lend(temp));
    cholmod_free_dense(&sol, &ch_settings);
}
#endif

template <typename ValueType, typename IndexType>
void solve_direct_ginkgo(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::solver::LowerTrs<ValueType, IndexType>>
        &L_solver,
    const std::shared_ptr<gko::solver::UpperTrs<ValueType, IndexType>>
        &U_solver,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto temp_rhs = vec::create(settings.executor, local_solution->get_size());
    L_solver->apply(gko::lend(local_solution), gko::lend(temp_rhs));

    U_solver->apply(gko::lend(temp_rhs), gko::lend(local_solution));
}

template <typename ValueType, typename IndexType>
void solve_iterative_ginkgo(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    const std::shared_ptr<gko::LinOp> &solver,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &local_rhs,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &local_solution)
{
    solver->apply(gko::lend(local_rhs), gko::lend(local_solution));
}

template <typename ValueType, typename IndexType>
void extract_local_vector(
    const Settings &settings, const Metadata<ValueType, IndexType> &metadata,
    std::shared_ptr<gko::matrix::Dense<ValueType>> &sub_vector,
    const std::shared_ptr<gko::matrix::Dense<ValueType>> &vector,
    const IndexType &vec_index)
{
    using vec = gko::matrix::Dense<ValueType>;
    auto local_size = metadata.local_size;
    auto tmp = vec::create(
        settings.executor, gko::dim<2>(metadata.local_size, 1),
        gko::Array<ValueType>::view(settings.executor, metadata.local_size,
                                    &(vector->at(vec_index))),
        1);
    auto tmp2 = vec::create(
        settings.executor, gko::dim<2>(metadata.local_size, 1),
        gko::Array<ValueType>::view(settings.executor, metadata.local_size,
                                    &(sub_vector->at(0))),
        1);
    // TODO: GPU (DONE)
    tmp2->copy_from(gko::lend(tmp));
    settings.executor->run(GatherScatter<ValueType, IndexType>(
        true, metadata.overlap_size, metadata.overlap_row->get_data(),
        vector->get_values(), &(sub_vector->get_values()[local_size])));
}

}  // namespace SolverTools


// Explicit Instantiations
#if SCHW_HAVE_CHOLMOD
#define DECLARE_FUNCTION(ValueType, IndexType)                    \
    void SolverTools::solve_direct_cholmod(                       \
        const Settings &, const Metadata<ValueType, IndexType> &, \
        cholmod_common &, cholmod_factor *, cholmod_dense *,      \
        std::shared_ptr<gko::matrix::Dense<ValueType>> &)
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION
#endif

#define DECLARE_FUNCTION(ValueType, IndexType)                    \
    void SolverTools::extract_local_vector(                       \
        const Settings &, const Metadata<ValueType, IndexType> &, \
        std::shared_ptr<gko::matrix::Dense<ValueType>> &,         \
        const std::shared_ptr<gko::matrix::Dense<ValueType>> &,   \
        const IndexType &)
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION

#define DECLARE_FUNCTION(ValueType, IndexType)                    \
    void SolverTools::solve_iterative_ginkgo(                     \
        const Settings &, const Metadata<ValueType, IndexType> &, \
        const std::shared_ptr<gko::LinOp> &,                      \
        std::shared_ptr<gko::matrix::Dense<ValueType>> &,         \
        std::shared_ptr<gko::matrix::Dense<ValueType>> &)         \
        INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_FUNCTION);
#undef DECLARE_FUNCTION


}  // namespace SchwarzWrappers


#endif  // solver_tools.hpp
