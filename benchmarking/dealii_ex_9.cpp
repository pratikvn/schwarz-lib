/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2018 by the deal.II authors
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
 */
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>
namespace Step9 {
using namespace dealii;
template <int dim>
class AdvectionField : public TensorFunction<1, dim> {
public:
    virtual Tensor<1, dim> value(const Point<dim> &p) const override;
    DeclException2(ExcDimensionMismatch, unsigned int, unsigned int,
                   << "The vector has size " << arg1 << " but should have "
                   << arg2 << " elements.");
};
template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
{
    Point<dim> value;
    value[0] = 2;
    for (unsigned int i = 1; i < dim; ++i)
        value[i] = 1 + 0.8 * std::sin(8. * numbers::PI * p[0]);
    return value;
}
template <int dim>
class RightHandSide : public Function<dim> {
public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;

private:
    static const Point<dim> center_point;
};
template <>
const Point<1> RightHandSide<1>::center_point = Point<1>(-0.75);
template <>
const Point<2> RightHandSide<2>::center_point = Point<2>(-0.75, -0.75);
template <>
const Point<3> RightHandSide<3>::center_point = Point<3>(-0.75, -0.75, -0.75);
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const
{
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    const double diameter = 0.1;
    return ((p - center_point).norm_square() < diameter * diameter
                ? 0.1 / std::pow(diameter, dim)
                : 0.0);
}
template <int dim>
class BoundaryValues : public Function<dim> {
public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;
};
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const
{
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    const double sine_term = std::sin(16. * numbers::PI * p.norm_square());
    const double weight = std::exp(5. * (1. - p.norm_square()));
    return weight * sine_term;
}
template <int dim>
class AdvectionProblem {
public:
    AdvectionProblem();
    void run();

private:
    void setup_system();
    struct AssemblyScratchData {
        AssemblyScratchData(const FiniteElement<dim> &fe);
        AssemblyScratchData(const AssemblyScratchData &scratch_data);
        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_face_values;
        std::vector<double> rhs_values;
        std::vector<Tensor<1, dim>> advection_directions;
        std::vector<double> face_boundary_values;
        std::vector<Tensor<1, dim>> face_advection_directions;
        AdvectionField<dim> advection_field;
        RightHandSide<dim> right_hand_side;
        BoundaryValues<dim> boundary_values;
    };
    struct AssemblyCopyData {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
    };
    void assemble_system();
    void local_assemble_system(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        AssemblyScratchData &scratch, AssemblyCopyData &copy_data);
    void copy_local_to_global(const AssemblyCopyData &copy_data);
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FE_Q<dim> fe;
    AffineConstraints<double> hanging_node_constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
};
class GradientEstimation {
public:
    template <int dim>
    static void estimate(const DoFHandler<dim> &dof,
                         const Vector<double> &solution,
                         Vector<float> &error_per_cell);
    DeclException2(ExcInvalidVectorLength, int, int,
                   << "Vector has length " << arg1 << ", but should have "
                   << arg2);
    DeclException0(ExcInsufficientDirections);

private:
    template <int dim>
    struct EstimateScratchData {
        EstimateScratchData(const FiniteElement<dim> &fe,
                            const Vector<double> &solution,
                            Vector<float> &error_per_cell);
        EstimateScratchData(const EstimateScratchData &data);
        FEValues<dim> fe_midpoint_value;
        std::vector<typename DoFHandler<dim>::active_cell_iterator>
            active_neighbors;
        const Vector<double> &solution;
        Vector<float> &error_per_cell;
        std::vector<double> cell_midpoint_value;
        std::vector<double> neighbor_midpoint_value;
    };
    struct EstimateCopyData {};
    template <int dim>
    static void estimate_cell(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        EstimateScratchData<dim> &scratch_data,
        const EstimateCopyData &copy_data);
};
template <int dim>
AdvectionProblem<dim>::AdvectionProblem() : dof_handler(triangulation), fe(5)
{}
template <int dim>
void AdvectionProblem<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, hanging_node_constraints,
                                    /*keep_constrained_dofs =*/false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}
template <int dim>
void AdvectionProblem<dim>::assemble_system()
{
    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), *this,
                    &AdvectionProblem::local_assemble_system,
                    &AdvectionProblem::copy_local_to_global,
                    AssemblyScratchData(fe), AssemblyCopyData());
}
template <int dim>
AdvectionProblem<dim>::AssemblyScratchData::AssemblyScratchData(
    const FiniteElement<dim> &fe)
    : fe_values(fe, QGauss<dim>(fe.degree + 1),
                update_values | update_gradients | update_quadrature_points |
                    update_JxW_values),
      fe_face_values(fe, QGauss<dim - 1>(fe.degree + 1),
                     update_values | update_quadrature_points |
                         update_JxW_values | update_normal_vectors),
      rhs_values(fe_values.get_quadrature().size()),
      advection_directions(fe_values.get_quadrature().size()),
      face_boundary_values(fe_face_values.get_quadrature().size()),
      face_advection_directions(fe_face_values.get_quadrature().size())
{}
template <int dim>
AdvectionProblem<dim>::AssemblyScratchData::AssemblyScratchData(
    const AssemblyScratchData &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                update_values | update_gradients | update_quadrature_points |
                    update_JxW_values),
      fe_face_values(scratch_data.fe_face_values.get_fe(),
                     scratch_data.fe_face_values.get_quadrature(),
                     update_values | update_quadrature_points |
                         update_JxW_values | update_normal_vectors),
      rhs_values(scratch_data.rhs_values.size()),
      advection_directions(scratch_data.advection_directions.size()),
      face_boundary_values(scratch_data.face_boundary_values.size()),
      face_advection_directions(scratch_data.face_advection_directions.size())
{}
template <int dim>
void AdvectionProblem<dim>::local_assemble_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data)
{
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points =
        scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points =
        scratch_data.fe_face_values.get_quadrature().size();
    copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    copy_data.cell_rhs.reinit(dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);
    scratch_data.fe_values.reinit(cell);
    scratch_data.advection_field.value_list(
        scratch_data.fe_values.get_quadrature_points(),
        scratch_data.advection_directions);
    scratch_data.right_hand_side.value_list(
        scratch_data.fe_values.get_quadrature_points(),
        scratch_data.rhs_values);
    const double delta = 0.1 * cell->diameter();
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const auto &sd = scratch_data;
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                copy_data.cell_matrix(i, j) +=
                    ((sd.fe_values.shape_value(i, q_point) +       // (phi_i +
                      delta * (sd.advection_directions[q_point] *  // delta beta
                               sd.fe_values.shape_grad(
                                   i, q_point))) *           // grad phi_i)
                     sd.advection_directions[q_point] *      // beta
                     sd.fe_values.shape_grad(j, q_point)) *  // grad phi_j
                    sd.fe_values.JxW(q_point);               // dx
            copy_data.cell_rhs(i) +=
                (sd.fe_values.shape_value(i, q_point) +  // (phi_i +
                 delta *
                     (sd.advection_directions[q_point] *       // delta beta
                      sd.fe_values.shape_grad(i, q_point))) *  // grad phi_i)
                sd.rhs_values[q_point] *                       // f
                sd.fe_values.JxW(q_point);                     // dx
        }
    for (const auto &face : cell->face_iterators())
        if (face->at_boundary()) {
            scratch_data.fe_face_values.reinit(cell, face);
            scratch_data.boundary_values.value_list(
                scratch_data.fe_face_values.get_quadrature_points(),
                scratch_data.face_boundary_values);
            scratch_data.advection_field.value_list(
                scratch_data.fe_face_values.get_quadrature_points(),
                scratch_data.face_advection_directions);
            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
                if (scratch_data.fe_face_values.normal_vector(q_point) *
                        scratch_data.face_advection_directions[q_point] <
                    0.)
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            copy_data.cell_matrix(i, j) -=
                                (scratch_data
                                     .face_advection_directions[q_point] *
                                 scratch_data.fe_face_values.normal_vector(
                                     q_point) *
                                 scratch_data.fe_face_values.shape_value(
                                     i, q_point) *
                                 scratch_data.fe_face_values.shape_value(
                                     j, q_point) *
                                 scratch_data.fe_face_values.JxW(q_point));
                        copy_data.cell_rhs(i) -=
                            (scratch_data.face_advection_directions[q_point] *
                             scratch_data.fe_face_values.normal_vector(
                                 q_point) *
                             scratch_data.face_boundary_values[q_point] *
                             scratch_data.fe_face_values.shape_value(i,
                                                                     q_point) *
                             scratch_data.fe_face_values.JxW(q_point));
                    }
        }
    cell->get_dof_indices(copy_data.local_dof_indices);
}
template <int dim>
void AdvectionProblem<dim>::copy_local_to_global(
    const AssemblyCopyData &copy_data)
{
    hanging_node_constraints.distribute_local_to_global(
        copy_data.cell_matrix, copy_data.cell_rhs, copy_data.local_dof_indices,
        system_matrix, system_rhs);
}
template <int dim>
void AdvectionProblem<dim>::solve()
{
    SolverControl solver_control(
        std::max<std::size_t>(1000, system_rhs.size() / 10),
        1e-10 * system_rhs.l2_norm());
    SolverGMRES<> solver(solver_control);
    PreconditionJacobi<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    Vector<double> residual(dof_handler.n_dofs());
    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    std::cout << "   Iterations required for convergence: "
              << solver_control.last_step() << '\n'
              << "   Max norm of residual:                "
              << residual.linfty_norm() << '\n';
    hanging_node_constraints.distribute(solution);
}
template <int dim>
void AdvectionProblem<dim>::refine_grid()
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    GradientEstimation::estimate(dof_handler, solution,
                                 estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
}
template <int dim>
void AdvectionProblem<dim>::output_results(const unsigned int cycle) const
{
    {
        GridOut grid_out;
        std::ofstream output("grid-" + std::to_string(cycle) + ".vtu");
        grid_out.write_vtu(triangulation, output);
    }
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches(8);
        DataOutBase::VtkFlags vtk_flags;
        vtk_flags.compression_level =
            DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
        data_out.set_flags(vtk_flags);
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
        data_out.write_vtu(output);
    }
}
template <int dim>
void AdvectionProblem<dim>::run()
{
    for (unsigned int cycle = 0; cycle < 10; ++cycle) {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0) {
            GridGenerator::hyper_cube(triangulation, -1, 1);
            triangulation.refine_global(3);
        } else {
            refine_grid();
        }
        std::cout << "   Number of active cells:              "
                  << triangulation.n_active_cells() << std::endl;
        setup_system();
        std::cout << "   Number of degrees of freedom:        "
                  << dof_handler.n_dofs() << std::endl;
        assemble_system();
        solve();
        output_results(cycle);
    }
}
template <int dim>
GradientEstimation::EstimateScratchData<dim>::EstimateScratchData(
    const FiniteElement<dim> &fe, const Vector<double> &solution,
    Vector<float> &error_per_cell)
    : fe_midpoint_value(fe, QMidpoint<dim>(),
                        update_values | update_quadrature_points),
      solution(solution),
      error_per_cell(error_per_cell),
      cell_midpoint_value(1),
      neighbor_midpoint_value(1)
{
    active_neighbors.reserve(GeometryInfo<dim>::faces_per_cell *
                             GeometryInfo<dim>::max_children_per_face);
}
template <int dim>
GradientEstimation::EstimateScratchData<dim>::EstimateScratchData(
    const EstimateScratchData &scratch_data)
    : fe_midpoint_value(scratch_data.fe_midpoint_value.get_fe(),
                        scratch_data.fe_midpoint_value.get_quadrature(),
                        update_values | update_quadrature_points),
      solution(scratch_data.solution),
      error_per_cell(scratch_data.error_per_cell),
      cell_midpoint_value(1),
      neighbor_midpoint_value(1)
{}
template <int dim>
void GradientEstimation::estimate(const DoFHandler<dim> &dof_handler,
                                  const Vector<double> &solution,
                                  Vector<float> &error_per_cell)
{
    Assert(error_per_cell.size() ==
               dof_handler.get_triangulation().n_active_cells(),
           ExcInvalidVectorLength(
               error_per_cell.size(),
               dof_handler.get_triangulation().n_active_cells()));
    WorkStream::run(dof_handler.begin_active(), dof_handler.end(),
                    &GradientEstimation::template estimate_cell<dim>,
                    std::function<void(const EstimateCopyData &)>(),
                    EstimateScratchData<dim>(dof_handler.get_fe(), solution,
                                             error_per_cell),
                    EstimateCopyData());
}
template <int dim>
void GradientEstimation::estimate_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    EstimateScratchData<dim> &scratch_data, const EstimateCopyData &)
{
    Tensor<2, dim> Y;
    scratch_data.fe_midpoint_value.reinit(cell);
    scratch_data.active_neighbors.clear();
    for (unsigned int face_n : GeometryInfo<dim>::face_indices())
        if (!cell->at_boundary(face_n)) {
            const auto face = cell->face(face_n);
            const auto neighbor = cell->neighbor(face_n);
            if (neighbor->is_active())
                scratch_data.active_neighbors.push_back(neighbor);
            else {
                if (dim == 1) {
                    auto neighbor_child = neighbor;
                    while (neighbor_child->has_children())
                        neighbor_child =
                            neighbor_child->child(face_n == 0 ? 1 : 0);
                    Assert(
                        neighbor_child->neighbor(face_n == 0 ? 1 : 0) == cell,
                        ExcInternalError());
                    scratch_data.active_neighbors.push_back(neighbor_child);
                } else
                    for (unsigned int subface_n = 0;
                         subface_n < face->n_children(); ++subface_n)
                        scratch_data.active_neighbors.push_back(
                            cell->neighbor_child_on_subface(face_n, subface_n));
            }
        }
    const Point<dim> this_center =
        scratch_data.fe_midpoint_value.quadrature_point(0);
    scratch_data.fe_midpoint_value.get_function_values(
        scratch_data.solution, scratch_data.cell_midpoint_value);
    Tensor<1, dim> projected_gradient;
    for (const auto &neighbor : scratch_data.active_neighbors) {
        scratch_data.fe_midpoint_value.reinit(neighbor);
        const Point<dim> neighbor_center =
            scratch_data.fe_midpoint_value.quadrature_point(0);
        scratch_data.fe_midpoint_value.get_function_values(
            scratch_data.solution, scratch_data.neighbor_midpoint_value);
        Tensor<1, dim> y = neighbor_center - this_center;
        const double distance = y.norm();
        y /= distance;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j) Y[i][j] += y[i] * y[j];
        projected_gradient += (scratch_data.neighbor_midpoint_value[0] -
                               scratch_data.cell_midpoint_value[0]) /
                              distance * y;
    }
    AssertThrow(determinant(Y) != 0, ExcInsufficientDirections());
    const Tensor<2, dim> Y_inverse = invert(Y);
    const Tensor<1, dim> gradient = Y_inverse * projected_gradient;
    scratch_data.error_per_cell(cell->active_cell_index()) =
        (std::pow(cell->diameter(), 1 + 1.0 * dim / 2) * gradient.norm());
}
}  // namespace Step9
int main()
{
    using namespace dealii;
    try {
        MultithreadInfo::set_thread_limit();
        Step9::AdvectionProblem<2> advection_problem_2d;
        advection_problem_2d.run();
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
