/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 * Modified version.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>


#include <bench_base.hpp>


DEFINE_uint32(
    num_refine_cycles, 1,
    "Number of refinement cycles for the adaptive refinement within deal.ii");
DEFINE_uint32(init_refine_level, 4,
              "Initial level for the refinement of the mesh.");
DEFINE_bool(dealii_orig, false, "Solve with dealii iterative CG");
DEFINE_bool(vis_sol, false, "Print the solution for visualization");

#define CHECK_HERE std::cout << "Here " << __LINE__ << std::endl;


using namespace dealii;
template <int dim, typename ValueType = double, typename IndexType = int>
class BenchDealiiLaplace : public BenchBase<ValueType, IndexType> {
public:
    BenchDealiiLaplace();
    void run();
    void run(MPI_Comm mpi_communicator);

private:
    void setup_system();
    void assemble_system();
    void solve();
    void solve(MPI_Comm mpi_communicator);
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    AffineConstraints<double> constraints;
    SparseMatrix<double> system_matrix;
    SparsityPattern sparsity_pattern;
    Vector<double> solution;
    Vector<double> system_rhs;
};


template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p.square() < 0.5 * 0.5)
        return 20;
    else
        return 1;
}


template <int dim, typename ValueType, typename IndexType>
BenchDealiiLaplace<dim, ValueType, IndexType>::BenchDealiiLaplace()
    : fe(2), dof_handler(triangulation)
{}


template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
}


template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::assemble_system()
{
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            const double current_coefficient =
                coefficient<dim>(fe_values.quadrature_point(q_index));
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) +=
                        (current_coefficient *               // a(x_q)
                         fe_values.shape_grad(i, q_index) *  // grad phi_i(x_q)
                         fe_values.shape_grad(j, q_index) *  // grad phi_j(x_q)
                         fe_values.JxW(q_index));            // dx
                cell_rhs(i) +=
                    (1.0 *                                // f(x)
                     fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                     fe_values.JxW(q_index));             // dx
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                               local_dof_indices, system_matrix,
                                               system_rhs);
    }
}


template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::solve()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    auto start_time = std::chrono::steady_clock::now();
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    auto elapsed_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time);
    std::cout << "Time for solve only: " << elapsed_time.count() << std::endl;
    constraints.distribute(solution);
}


template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::solve(
    MPI_Comm mpi_communicator)
{
    schwz::Metadata<ValueType, IndexType> metadata;
    schwz::Settings settings(FLAGS_executor);

    // Set solver metadata from command line args.
    metadata.mpi_communicator = mpi_communicator;
    MPI_Comm_rank(metadata.mpi_communicator, &metadata.my_rank);
    MPI_Comm_size(metadata.mpi_communicator, &metadata.comm_size);
    metadata.tolerance = FLAGS_set_tol;
    metadata.max_iters = FLAGS_num_iters;
    metadata.num_subdomains = metadata.comm_size;
    metadata.num_threads = FLAGS_num_threads;
    metadata.oned_laplacian_size = FLAGS_set_1d_laplacian_size;

    // Generic settings
    settings.write_debug_out = FLAGS_enable_debug_write;
    settings.write_perm_data = FLAGS_write_perm_data;
    settings.write_iters_and_residuals = FLAGS_write_iters_and_residuals;
    settings.print_matrices = FLAGS_print_matrices;
    settings.shifted_iter = FLAGS_shifted_iter;

    // Set solver settings from command line args.
    // Comm settings
    settings.comm_settings.enable_onesided = FLAGS_enable_onesided;
    if (FLAGS_remote_comm_type == "put") {
        settings.comm_settings.enable_put = true;
        settings.comm_settings.enable_get = false;
    } else if (FLAGS_remote_comm_type == "get") {
        settings.comm_settings.enable_put = false;
        settings.comm_settings.enable_get = true;
    }
    settings.comm_settings.enable_one_by_one = FLAGS_enable_one_by_one;
    settings.comm_settings.enable_overlap = FLAGS_enable_comm_overlap;
    if (FLAGS_flush_type == "flush-all") {
        settings.comm_settings.enable_flush_all = true;
    } else if (FLAGS_flush_type == "flush-local") {
        settings.comm_settings.enable_flush_all = false;
        settings.comm_settings.enable_flush_local = true;
    }
    if (FLAGS_lock_type == "lock-all") {
        settings.comm_settings.enable_lock_all = true;
    } else if (FLAGS_lock_type == "lock-local") {
        settings.comm_settings.enable_lock_all = false;
        settings.comm_settings.enable_lock_local = true;
    }

    // Convergence settings
    settings.convergence_settings.put_all_local_residual_norms =
        FLAGS_enable_put_all_local_residual_norms;
    settings.convergence_settings.enable_global_check_iter_offset =
        FLAGS_enable_global_check_iter_offset;
    settings.convergence_settings.enable_global_check =
        FLAGS_enable_global_check;
    if (FLAGS_global_convergence_type == "centralized-tree") {
        settings.convergence_settings.enable_global_simple_tree = true;
    } else if (FLAGS_global_convergence_type == "decentralized") {
        settings.convergence_settings.enable_decentralized_leader_election =
            true;
        settings.convergence_settings.enable_accumulate =
            FLAGS_enable_decentralized_accumulate;
    }

    // General solver settings
    metadata.local_solver_tolerance = FLAGS_local_tol;
    metadata.local_precond = FLAGS_local_precond;
    metadata.local_max_iters = FLAGS_local_max_iters;
    settings.non_symmetric_matrix = FLAGS_non_symmetric_matrix;
    settings.restart_iter = FLAGS_restart_iter;
    metadata.precond_max_block_size = FLAGS_precond_max_block_size;
    metadata.precond_max_block_size = FLAGS_precond_max_block_size;
    settings.matrix_filename = FLAGS_matrix_filename;
    settings.explicit_laplacian = FLAGS_explicit_laplacian;
    settings.enable_random_rhs = FLAGS_enable_random_rhs;
    settings.overlap = FLAGS_overlap;
    settings.naturally_ordered_factor = FLAGS_factor_ordering_natural;
    settings.reorder = FLAGS_local_reordering;
    settings.factorization = FLAGS_local_factorization;
    if (FLAGS_partition == "metis") {
        settings.partition =
            schwz::Settings::partition_settings::partition_metis;
        settings.metis_objtype = FLAGS_metis_objtype;
    } else if (FLAGS_partition == "regular") {
        settings.partition =
            schwz::Settings::partition_settings::partition_regular;
    } else if (FLAGS_partition == "regular2d") {
        settings.partition =
            schwz::Settings::partition_settings::partition_regular2d;
    }
    if (FLAGS_local_solver == "iterative-ginkgo") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::iterative_solver_ginkgo;
    } else if (FLAGS_local_solver == "direct-cholmod") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::direct_solver_cholmod;
    } else if (FLAGS_local_solver == "direct-umfpack") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::direct_solver_umfpack;
    } else if (FLAGS_local_solver == "direct-ginkgo") {
        settings.local_solver =
            schwz::Settings::local_solver_settings::direct_solver_ginkgo;
    }
    settings.debug_print = FLAGS_debug;
    int gsize = 0;
    if (metadata.my_rank == 0) {
        metadata.global_size = system_matrix.m();
        std::cout << " Running on the " << FLAGS_executor << " executor on "
                  << metadata.num_subdomains << " ranks with "
                  << FLAGS_num_threads << " threads" << std::endl;
        std::cout << " Problem Size: " << metadata.global_size
                  << " Number of non-zeros: "
                  << system_matrix.n_nonzero_elements() << std::endl;
        gsize = metadata.global_size;
    }
    MPI_Bcast(&gsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    metadata.global_size = gsize;
    if (FLAGS_print_config) {
        if (metadata.my_rank == 0) {
            this->print_config();
        }
    }
    using vec_vtype = gko::matrix::Dense<ValueType>;
    std::shared_ptr<vec_vtype> solution_vector;
    schwz::SolverRAS<ValueType, IndexType> solver(settings, metadata);
    solver.initialize(system_matrix, system_rhs);
    auto start_time = std::chrono::steady_clock::now();
    solver.run(solution_vector);
    auto elapsed_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time);
    if (metadata.my_rank == 0) {
        std::cout << "Time for solve only: " << elapsed_time.count()
                  << std::endl;
    }
    if (FLAGS_timings_file != "null") {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename = FLAGS_timings_file + "_" + rank_string + ".csv";
        this->write_timings(metadata.time_struct, filename,
                            settings.comm_settings.enable_onesided);
    }
    if (FLAGS_write_comm_data) {
        std::string rank_string = std::to_string(metadata.my_rank);
        if (metadata.my_rank < 10) {
            rank_string = "0" + std::to_string(metadata.my_rank);
        }
        std::string filename_send = "num_send_" + rank_string + ".csv";
        std::string filename_recv = "num_recv_" + rank_string + ".csv";
        this->write_comm_data(metadata.num_subdomains, metadata.my_rank,
                              metadata.comm_data_struct, filename_send,
                              filename_recv);
    }

    if (metadata.my_rank == 0) {
        std::copy(solution_vector->get_values(),
                  solution_vector->get_values() + metadata.global_size,
                  solution.begin());
        constraints.distribute(solution);
    }
}


template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::refine_grid()
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
        dof_handler, QGauss<dim - 1>(fe.degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(), solution,
        estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
}


template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::output_results(
    const unsigned int cycle) const
{
    {
        GridOut grid_out;
        std::ofstream output("grid-" + std::to_string(cycle) + ".gnuplot");
        GridOutFlags::Gnuplot gnuplot_flags(false, 5);
        grid_out.set_flags(gnuplot_flags);
        MappingQGeneric<dim> mapping(3);
        grid_out.write_gnuplot(triangulation, output, &mapping);
    }
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
        data_out.write_vtu(output);
    }
}

template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::run()
{
    int num_cycles = FLAGS_num_refine_cycles;

    for (unsigned int cycle = 0; cycle < num_cycles; ++cycle) {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0) {
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(FLAGS_init_refine_level);
        } else
            refine_grid();
        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;
        setup_system();
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;
        assemble_system();
        this->solve();
        if (FLAGS_vis_sol) {
            output_results(cycle);
        }
    }
}

template <int dim, typename ValueType, typename IndexType>
void BenchDealiiLaplace<dim, ValueType, IndexType>::run(
    MPI_Comm mpi_communicator)
{
    int num_cycles = FLAGS_num_refine_cycles;
    int mpi_size, mpi_rank;
    MPI_Comm_size(mpi_communicator, &mpi_size);
    MPI_Comm_rank(mpi_communicator, &mpi_rank);

    for (unsigned int cycle = 0; cycle < num_cycles; ++cycle) {
        if (mpi_rank == 0) {
            std::cout << "Cycle " << cycle << ':' << std::endl;

            if (cycle == 0) {
                GridGenerator::hyper_cube(triangulation);
                triangulation.refine_global(FLAGS_init_refine_level);
            } else
                refine_grid();
            std::cout << "   Number of active cells:       "
                      << triangulation.n_active_cells() << std::endl;
            setup_system();
            std::cout << "   Number of degrees of freedom: "
                      << dof_handler.n_dofs() << std::endl;
            assemble_system();
        }
        this->solve(MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            if (FLAGS_vis_sol) {
                output_results(cycle);
            }
        }
    }
}


int main(int argc, char **argv)
{
    try {
        initialize_argument_parsing(&argc, &argv);
        BenchDealiiLaplace<3, double, int> laplace_problem;
        if (FLAGS_num_threads > 1) {
            int req_thread_support = MPI_THREAD_MULTIPLE;
            int prov_thread_support = MPI_THREAD_MULTIPLE;

            MPI_Init_thread(&argc, &argv, req_thread_support,
                            &prov_thread_support);
            if (prov_thread_support != req_thread_support) {
                std::cout << "Required thread support is " << req_thread_support
                          << " but provided thread support is only "
                          << prov_thread_support << std::endl;
            }
        } else {
            MPI_Init(&argc, &argv);
        }

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (FLAGS_dealii_orig) {
            if (rank == 0) {
                auto start_time = std::chrono::steady_clock::now();
                laplace_problem.run();
                auto elapsed_time = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start_time);
                std::cout << "Total Time for setup+solve: "
                          << elapsed_time.count() << std::endl;
            }
        } else {
            auto start_time = std::chrono::steady_clock::now();
            laplace_problem.run(MPI_COMM_WORLD);
            auto elapsed_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time);
            if (rank == 0) {
                std::cout << "Total Time for setup+solve: "
                          << elapsed_time.count() << std::endl;
            }
        }
        MPI_Finalize();
    } catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    } catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}
