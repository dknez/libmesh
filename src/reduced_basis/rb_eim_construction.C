// rbOOmit: An implementation of the Certified Reduced Basis method.
// Copyright (C) 2009, 2010 David J. Knezevic

// This file is part of rbOOmit.

// rbOOmit is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// rbOOmit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

// C++ includes
#include <fstream>
#include <sstream>

// LibMesh includes
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/equation_systems.h"
#include "libmesh/parallel.h"
#include "libmesh/parallel_algebra.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature.h"
#include "libmesh/utility.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_compute_data.h"
#include "libmesh/getpot.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/fem_context.h"
#include "libmesh/elem.h"
#include "libmesh/int_range.h"
#include "libmesh/auto_ptr.h"

// rbOOmit includes
#include "libmesh/rb_eim_construction.h"
#include "libmesh/rb_eim_evaluation.h"

namespace libMesh
{

RBEIMConstruction::RBEIMConstruction (EquationSystems & es,
                                      const std::string & name_in,
                                      const unsigned int number_in)
  : RBConstructionBase(es, name_in, number_in),
    best_fit_type_flag(PROJECTION_BEST_FIT),
    _parametrized_functions_in_training_set_initialized(false)
{
}

RBEIMConstruction::~RBEIMConstruction ()
{
}

void RBEIMConstruction::clear()
{
  RBConstructionBase::clear();

  // clear the eim assembly vector
  _rb_eim_assembly_objects.clear();

  // clear the parametrized functions from the training set
  _local_parametrized_functions_for_training.clear();
}

void RBEIMConstruction::set_rb_eim_evaluation(RBEIMEvaluation & rb_eim_eval_in)
{
  _rb_eim_eval = &rb_eim_eval_in;
}

RBEIMEvaluation & RBEIMConstruction::get_rb_eim_evaluation()
{
  if (!_rb_eim_eval)
    libmesh_error_msg("Error: RBEIMEvaluation object hasn't been initialized yet");

  return *_rb_eim_eval;
}

void RBEIMConstruction::process_parameters_file (const std::string & parameters_filename)
{
  Parent::process_parameters_file(parameters_filename);

  GetPot infile(parameters_filename);

  std::string best_fit_type_string = infile("best_fit_type","projection");
  set_best_fit_type_flag(best_fit_type_string);
}

void RBEIMConstruction::set_best_fit_type_flag (const std::string & best_fit_type_string)
{
  if (best_fit_type_string == "projection")
    {
      best_fit_type_flag = PROJECTION_BEST_FIT;
    }
  else
    if (best_fit_type_string == "eim")
      {
        best_fit_type_flag = EIM_BEST_FIT;
      }
    else
      libmesh_error_msg("Error: invalid best_fit_type in input file");
}

void RBEIMConstruction::print_info()
{
  RBConstructionBase::print_info();

  // Print out setup info
  libMesh::out << std::endl << "RBEIMConstruction parameters:" << std::endl;
  if (best_fit_type_flag == PROJECTION_BEST_FIT)
    {
      libMesh::out << "best fit type: projection" << std::endl;
    }
  else
    if (best_fit_type_flag == EIM_BEST_FIT)
      {
        libMesh::out << "best fit type: eim" << std::endl;
      }
  libMesh::out << std::endl;
}

void RBEIMConstruction::initialize_eim_construction()
{
  initialize_quad_point_data();
  initialize_parametrized_functions_in_training_set();

  _eim_projection_matrix.resize(Nmax,Nmax);
}

Real RBEIMConstruction::train_eim_approximation()
{
}

void RBEIMConstruction::initialize_eim_assembly_objects()
{
  _rb_eim_assembly_objects.clear();
  for (unsigned int i=0; i<get_rb_evaluation().get_n_basis_functions(); i++)
    _rb_eim_assembly_objects.push_back(build_eim_assembly(i));
}

std::vector<std::unique_ptr<ElemAssembly>> & RBEIMConstruction::get_eim_assembly_objects()
{
  return _rb_eim_assembly_objects;
}

Real RBEIMConstruction::compute_best_fit_error()
{
  LOG_SCOPE("compute_best_fit_error()", "RBEIMConstruction");

  const unsigned int RB_size = get_rb_evaluation().get_n_basis_functions();

  // load up the parametrized function for the current parameters
  truth_solve(-1);

  switch(best_fit_type_flag)
    {
      // Perform an L2 projection in order to find an approximation to solution (from truth_solve above)
    case(PROJECTION_BEST_FIT):
      {
        // We have pre-stored inner_product_matrix * basis_function[i] for each i
        // so we can just evaluate the dot product here.
        DenseVector<Number> best_fit_rhs(RB_size);
        for (unsigned int i=0; i<RB_size; i++)
          {
            best_fit_rhs(i) = get_explicit_system().solution->dot(*_matrix_times_bfs[i]);
          }

        // Now compute the best fit by an LU solve
        get_rb_evaluation().RB_solution.resize(RB_size);
        DenseMatrix<Number> RB_inner_product_matrix_N(RB_size);
        _eim_projection_matrix.get_principal_submatrix(RB_size, RB_inner_product_matrix_N);

        RB_inner_product_matrix_N.lu_solve(best_fit_rhs, get_rb_evaluation().RB_solution);
        break;
      }
      // Perform EIM solve in order to find the approximation to solution
      // (rb_solve provides the EIM basis function coefficients used below)
    case(EIM_BEST_FIT):
      {
        // Turn off error estimation for this rb_solve, we use the linfty norm instead
        get_rb_evaluation().evaluate_RB_error_bound = false;
        get_rb_evaluation().set_parameters( get_parameters() );
        get_rb_evaluation().rb_solve(RB_size);
        get_rb_evaluation().evaluate_RB_error_bound = true;
        break;
      }
    default:
      libmesh_error_msg("Should not reach here");
    }

  // load the error into solution
  for (unsigned int i=0; i<get_rb_evaluation().get_n_basis_functions(); i++)
    get_explicit_system().solution->add(-get_rb_evaluation().RB_solution(i),
                                        get_rb_evaluation().get_basis_function(i));

  Real best_fit_error = get_explicit_system().solution->linfty_norm();

  return best_fit_error;
}

void RBEIMConstruction::init_context(FEMContext & c)
{
  // Pre-request FE data for all element dimensions present in the
  // mesh.  Note: we currently pre-request FE data for all variables
  // in the current system but in some cases that may be overkill, for
  // example if only variable 0 is used.
  const System & sys = c.get_system();
  const MeshBase & mesh = sys.get_mesh();

  for (unsigned int dim=1; dim<=3; ++dim)
    if (mesh.elem_dimensions().count(dim))
      for (unsigned int var=0; var<sys.n_vars(); ++var)
      {
        auto fe = c.get_element_fe(var, dim);
        fe->get_JxW();
        fe->get_phi();
        fe->get_xyz();

        auto side_fe = c.get_side_fe(var, dim);
        side_fe->get_JxW();
        side_fe->get_phi();
        side_fe->get_xyz();
      }
}

void RBEIMConstruction::enrich_eim_approximation()
{
  LOG_SCOPE("enrich_eim_approximation()", "RBEIMConstruction");

  // put solution in _ghosted_meshfunction_vector so we can access it from the mesh function
  // this allows us to compute EIM_rhs appropriately
  get_explicit_system().solution->localize(*_ghosted_meshfunction_vector,
                                           get_explicit_system().get_dof_map().get_send_list());

  RBEIMEvaluation & eim_eval = get_rb_eim_evaluation();

  // If we have at least one basis function we need to use
  // rb_solve to find the EIM interpolation error, otherwise just use solution as is
  if (get_rb_evaluation().get_n_basis_functions() > 0)
    {
      // get the right-hand side vector for the EIM approximation
      // by sampling the parametrized function (stored in solution)
      // at the interpolation points
      unsigned int RB_size = get_rb_evaluation().get_n_basis_functions();
      DenseVector<Number> EIM_rhs(RB_size);
      for (unsigned int i=0; i<RB_size; i++)
        {
          EIM_rhs(i) = evaluate_mesh_function( eim_eval.interpolation_points_var[i],
                                               eim_eval.interpolation_points[i] );
        }

      eim_eval.set_parameters( get_parameters() );
      eim_eval.rb_solve(EIM_rhs);

      // Load the "EIM residual" into solution by subtracting
      // the EIM approximation
      for (unsigned int i=0; i<get_rb_evaluation().get_n_basis_functions(); i++)
        {
          get_explicit_system().solution->add(-eim_eval.RB_solution(i), get_rb_evaluation().get_basis_function(i));
        }
    }

  // need to update since context uses current_local_solution
  get_explicit_system().update();

  // Find the quadrature point at which solution (which now stores
  // the "EIM residual") has maximum absolute value
  // by looping over the mesh
  Point optimal_point;
  Number optimal_value = 0.;
  unsigned int optimal_var = 0;
  dof_id_type optimal_elem_id = DofObject::invalid_id;

  // Initialize largest_abs_value to be negative so that it definitely gets updated.
  Real largest_abs_value = -1.;

  // Compute truth representation via projection
  MeshBase & mesh = this->get_mesh();

  std::unique_ptr<DGFEMContext> explicit_c = libmesh_make_unique<DGFEMContext>(get_explicit_system());
  DGFEMContext & explicit_context = cast_ref<DGFEMContext &>(*explicit_c);

  // Pre-request required data
  init_context_with_sys(explicit_context, get_explicit_system());

  // Get local reference to xyz data for variable 0. This is needed in
  // the loop below, but we cannot call elem_fe->get_xyz() once
  // calculations have already started. Note: We don't need a separate
  // "xyz" for each var, since all vars should be using the same
  // quadrature rule.
  FEBase * elem_fe = nullptr;
  explicit_context.get_element_fe( 0, elem_fe );
  const std::vector<Point> & xyz = elem_fe->get_xyz();

  for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      explicit_context.pre_fe_reinit(get_explicit_system(), elem);
      explicit_context.elem_fe_reinit();

      for (unsigned int var=0; var<get_explicit_system().n_vars(); var++)
        {
          unsigned int n_qpoints = explicit_context.get_element_qrule().n_points();

          for (unsigned int qp=0; qp<n_qpoints; qp++)
            {
              Number value = explicit_context.interior_value(var, qp);
              Real abs_value = std::abs(value);

              if (abs_value > largest_abs_value)
                {
                  optimal_value = value;
                  largest_abs_value = abs_value;
                  optimal_var = var;
                  optimal_elem_id = elem->id();
                  optimal_point = xyz[qp];
                }
            }
        }
    }

  // Find out which processor has the largest of the abs values
  unsigned int proc_ID_index;
  this->comm().maxloc(largest_abs_value, proc_ID_index);

  // Broadcast the optimal point from proc_ID_index
  this->comm().broadcast(optimal_point, proc_ID_index);

  // Also broadcast the corresponding optimal_var, optimal_value, and optimal_elem_id
  this->comm().broadcast(optimal_var, proc_ID_index);
  this->comm().broadcast(optimal_value, proc_ID_index);
  this->comm().broadcast(optimal_elem_id, proc_ID_index);

  // In debug mode, assert that we found an optimal_elem_id
  libmesh_assert_not_equal_to(optimal_elem_id, DofObject::invalid_id);

  // Scale the solution
  get_explicit_system().solution->scale(1./optimal_value);

  // Store optimal point in interpolation_points
  eim_eval.interpolation_points.push_back(optimal_point);
  eim_eval.interpolation_points_var.push_back(optimal_var);
  Elem * elem_ptr = mesh.elem_ptr(optimal_elem_id);
  eim_eval.interpolation_points_elem.push_back( elem_ptr );

  {
    auto new_bf = NumericVector<Number>::build(this->comm());
    new_bf->init (get_explicit_system().n_dofs(), get_explicit_system().n_local_dofs(), false, PARALLEL);
    *new_bf = *get_explicit_system().solution;
    get_rb_evaluation().basis_functions.emplace_back( std::move(new_bf) );
  }

  if (best_fit_type_flag == PROJECTION_BEST_FIT)
    {
      // In order to speed up dot products, we store the product
      // of the basis function and the inner product matrix

      std::unique_ptr<NumericVector<Number>> implicit_sys_temp1 = this->solution->zero_clone();
      std::unique_ptr<NumericVector<Number>> implicit_sys_temp2 = this->solution->zero_clone();
      auto matrix_times_new_bf = get_explicit_system().solution->zero_clone();

      // We must localize new_bf before calling get_explicit_sys_subvector
      std::unique_ptr<NumericVector<Number>> localized_new_bf =
        NumericVector<Number>::build(this->comm());
      localized_new_bf->init(get_explicit_system().n_dofs(), false, SERIAL);
      get_rb_evaluation().basis_functions.back()->localize(*localized_new_bf);

      for (unsigned int var=0; var<get_explicit_system().n_vars(); var++)
        {
          get_explicit_sys_subvector(*implicit_sys_temp1,
                                     var,
                                     *localized_new_bf);

          inner_product_matrix->vector_mult(*implicit_sys_temp2, *implicit_sys_temp1);

          set_explicit_sys_subvector(*matrix_times_new_bf,
                                     var,
                                     *implicit_sys_temp2);
        }

      _matrix_times_bfs.emplace_back(std::move(matrix_times_new_bf));
    }
}

void RBEIMConstruction::initialize_parametrized_functions_in_training_set()
{
  LOG_SCOPE("initialize_parametrized_functions_in_training_set()", "RBEIMConstruction");

  if (!serial_training_set)
    libmesh_error_msg("Error: We must have serial_training_set==true in " \
                      << "RBEIMConstruction::initialize_parametrized_functions_in_training_set");

  libMesh::out << "Initializing parametrized functions in training set..." << std::endl;

  // Store the locations of all quadrature points
  initialize_qp_data();

  get_rb_evaluation().initialize_parameters(*this);

  _local_parametrized_functions_for_training.resize( get_n_training_samples() );
  for (unsigned int i=0; i<get_n_training_samples(); i++)
    {
      set_params_from_training_set(i);
      eim_eval.get_parametrized_function().preevaluate_parametrized_function(get_parameters(),
                                                                             _local_quad_point_locations,
                                                                             _local_quad_point_subdomain_ids);

      for (unsigned int comp : index_range(eim_eval().get_n_parametrized_functions().get_n_components())
        for (const auto & [elem_id,xyz_vector] : _local_quad_point_locations)
          for (unsigned int qp : index_range(xyz_vector.size()))
            {
              _local_parametrized_functions_for_training[training_index][elem_id][var] =
                eim_eval.evaluate_parametrized_function(comp, elem_id, qp);
            }

      libMesh::out << "Initialized data for training sample " << (i+1) << " of " << get_n_training_samples() << std::endl;
    }

  libMesh::out << "Parametrized functions in training set initialized" << std::endl << std::endl;
}

void RBEIMConstruction::initialize_qp_data()
{
  LOG_SCOPE("initialize_qp_data()", "RBEIMConstruction");

  // Compute truth representation via L2 projection
  const MeshBase & mesh = this->get_mesh();

  FEMContext context(*this);
  init_context(context);

  FEBase * elem_fe = nullptr;
  context.get_element_fe( 0, elem_fe );
  const std::vector<Real> & JxW = elem_fe->get_JxW();
  const std::vector<Point> & xyz = elem_fe->get_xyz();

  _local_quad_point_locations.clear();
  _local_quad_point_subdomain_ids.clear();
  _local_quad_point_JxW.clear();

  for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      dof_id_type elem_id = elem->id();

      context.pre_fe_reinit(*this, elem);
      context.elem_fe_reinit();

      _local_quad_point_locations[elem_id] = xyz;
      _local_quad_point_JxW[elem_id] = JxW;
      _local_quad_point_subdomain_ids[elem_id] = elem->subdomain_id();
    }
}

void RBEIMConstruction::update_eim_matrices()
{
  LOG_SCOPE("update_eim_matrices()", "RBEIMConstruction");

  // First, update the inner product matrix
  {
    unsigned int RB_size = get_rb_evaluation().get_n_basis_functions();

    std::unique_ptr<NumericVector<Number>> explicit_sys_temp =
      get_explicit_system().solution->zero_clone();

    std::unique_ptr<NumericVector<Number>> temp1 = this->solution->zero_clone();
    std::unique_ptr<NumericVector<Number>> temp2 = this->solution->zero_clone();

    for (unsigned int i=(RB_size-1); i<RB_size; i++)
      {
        for (unsigned int j=0; j<RB_size; j++)
          {
            // We must localize get_rb_evaluation().get_basis_function(j) before calling
            // get_explicit_sys_subvector
            std::unique_ptr<NumericVector<Number>> localized_basis_function =
              NumericVector<Number>::build(this->comm());
            localized_basis_function->init(get_explicit_system().n_dofs(), false, SERIAL);
            get_rb_evaluation().get_basis_function(j).localize(*localized_basis_function);

            // Compute reduced inner_product_matrix via a series of matvecs
            for (unsigned int var=0; var<get_explicit_system().n_vars(); var++)
              {
                get_explicit_sys_subvector(*temp1, var, *localized_basis_function);
                inner_product_matrix->vector_mult(*temp2, *temp1);
                set_explicit_sys_subvector(*explicit_sys_temp, var, *temp2);
              }

            Number value = explicit_sys_temp->dot( get_rb_evaluation().get_basis_function(i) );
            _eim_projection_matrix(i,j) = value;
            if (i!=j)
              {
                // The inner product matrix is assumed
                // to be hermitian
                _eim_projection_matrix(j,i) = libmesh_conj(value);
              }
          }
      }
  }

  unsigned int RB_size = get_rb_evaluation().get_n_basis_functions();

  RBEIMEvaluation & eim_eval = get_rb_eim_evaluation();

  // update the EIM interpolation matrix
  for (unsigned int j=0; j<RB_size; j++)
    {
      // Sample the basis functions at the
      // new interpolation point
      get_rb_evaluation().get_basis_function(j).localize(*_ghosted_meshfunction_vector,
                                                         get_explicit_system().get_dof_map().get_send_list());

      eim_eval.interpolation_matrix(RB_size-1,j) =
        evaluate_mesh_function( eim_eval.interpolation_points_var[RB_size-1],
                                eim_eval.interpolation_points[RB_size-1] );
    }
}

} // namespace libMesh
