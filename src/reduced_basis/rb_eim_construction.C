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

  _rb_eim_assembly_objects.clear();

  _local_parametrized_functions_for_training.clear();
  _local_quad_point_locations.clear();
  _local_quad_point_JxW.clear();
  _local_quad_point_subdomain_ids.clear();

  _eim_projection_matrix.clear();
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
  initialize_parametrized_functions_in_training_set();
}

void RBEIMConstruction::train_eim_approximation()
{
  LOG_SCOPE("train_eim_approximation()", "RBConstruction");

  _eim_projection_matrix.resize(get_Nmax(),get_Nmax());

  RBEIMEvaluation & rbe = get_rb_eim_evaluation();
  rbe.initialize_parameters(*this);
  rbe.resize_data_structures(get_Nmax());    

  // If we are continuing from a previous training run,
  // we might already be at the max number of basis functions.
  // If so, we can just return.
  if (rbe.get_n_basis_functions() > 0)
    {
      libmesh_error_msg("Error: We currently only support EIM training starting from an empty basis");
    }

  libMesh::out << std::endl << "---- Performing Greedy EIM basis enrichment ----" << std::endl;
  Real initial_greedy_error = 0.;
  bool initial_greedy_error_initialized = false;
  std::vector<RBParameters> greedy_param_list;
  greedy_param_list.emplace_back();
  unsigned int current_training_index = 0;
  set_params_from_training_set(current_training_index);
  while (true)
    {
      libMesh::out << "Greedily selected parameter vector:" << std::endl;
      print_parameters();
      greedy_param_list.emplace_back(get_parameters());

      libMesh::out << "Enriching the EIM approximation" << std::endl;
      enrich_eim_approximation();
      update_eim_matrices();

      libMesh::out << std::endl << "---- Basis dimension: "
                   << rbe.get_n_basis_functions() << " ----" << std::endl;

      libMesh::out << "Computing EIM error on training set" << std::endl;
      std::pair<Real,unsigned int> max_error_pair = compute_max_eim_error();
      Real abs_greedy_error = max_error_pair.first;
      current_training_index = max_error_pair.second;
      set_params_from_training_set(current_training_index);

      libMesh::out << "Maximum EIM error is " << abs_greedy_error << std::endl << std::endl;

      // record the initial error
      if (!initial_greedy_error_initialized)
        {
          initial_greedy_error = abs_greedy_error;
          initial_greedy_error_initialized = true;
        }

      // Convergence and/or termination tests
      {
        if (rbe.get_n_basis_functions() >= this->get_Nmax())
          {
            libMesh::out << "Maximum number of basis functions reached: Nmax = "
                          << get_Nmax() << std::endl;
            break;
          }

        if (abs_greedy_error < this->abs_training_tolerance)
          {
            libMesh::out << "Absolute error tolerance reached." << std::endl;
            break;
          }

        Real rel_greedy_error = abs_greedy_error/initial_error;
        if (rel_greedy_error < this->rel_training_tolerance)
          {
            libMesh::out << "Relative error tolerance reached." << std::endl;
            break;
          }

        if (rbe.get_n_basis_functions() >= this->get_Nmax())
          {
            libMesh::out << "Maximum number of basis functions reached: Nmax = "
                         << get_Nmax() << std::endl;
            break;
          }

        for (auto & param : greedy_param_list)
          if (param == get_parameters())
            {
              libMesh::out << "Exiting greedy because the same parameters were selected twice"
                           << std::endl;
              break;
            }
      }
    }
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
        fe->get_xyz();

        auto side_fe = c.get_side_fe(var, dim);
        side_fe->get_JxW();
        side_fe->get_xyz();
      }
}

void RBEIMConstruction::get_parametrized_function_values_at_qps(
  const std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> & pf,
  dof_id_type elem_id,
  unsigned int comp,
  std::vector<Number> & values)
{
  LOG_SCOPE("get_parametrized_function_values_at_qps()", "RBEIMConstruction");

  values.clear();

  const auto it = pf.find(elem_id);
  if(it != pf.end())
  {
    const auto & comps_and_qps_on_elem = it->second;
    if(comp >= comps_and_qps_on_elem.size())
    {
      libmesh_error_msg("Invalid comp index: " + std::to_string(comp));
    }

    values = comps_and_qps_on_elem[comp];
  }
}

Number RBEIMConstruction::get_parametrized_function_value(
  const std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> & pf,
  dof_id_type elem_id,
  unsigned int comp,
  unsigned int qp)
{
  std::vector<Number> values;
  get_function_values_at_qps(pf, elem_id, comp);

  // In parallel, values should only be non-empty on one processor
  Number value = 0.;
  if(!values.empty())
  {
    if(qp >= values.size())
      libmesh_error_msg("Error: Invalid qp index");

    value = values[qp];
  }
  comm().sum(value);

  return value;
}

std::pair<Real, unsigned int> RBEIMConstruction::compute_max_eim_error()
{
  LOG_SCOPE("compute_max_eim_error()", "RBEIMConstruction");

  if (get_n_params() == 0)
    {
      // Just return 0 if we have no parameters.
      return std::make_pair<Real, unsigned int>(0.,0);
    }

  // keep track of the maximum error
  unsigned int max_err_index = 0;
  Real max_err = 0.;

  if (get_n_training_samples() != get_local_n_training_samples())
    libmesh_error_msg("Error: Training samples should be the same on all procs");

  for (unsigned int i=0; i<get_n_training_samples(); i++)
    {
      Real best_fit_error = compute_best_fit_error(i);

      if (best_fit_error > max_err)
        {
          max_err_index = i;
          max_err = best_fit_error;
        }
    }

  return std::make_pair<Real, unsigned int>(max_err,max_err_index);
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

  _local_parametrized_functions_for_training.resize( get_n_training_samples() );
  for (unsigned int i=0; i<get_n_training_samples(); i++)
    {
      libMesh::out << "Initializing parametrized function for training sample "
        << (i+1) << " of " << get_n_training_samples() << std::endl;

      set_params_from_training_set(i);
      eim_eval.get_parametrized_function().preevaluate_parametrized_function(get_parameters(),
                                                                             _local_quad_point_locations,
                                                                             _local_quad_point_subdomain_ids);

      unsigned int n_comps = eim_eval().get_n_parametrized_functions().get_n_components();

      for (const auto & [elem_id,xyz_vector] : _local_quad_point_locations)
      {
        std::vector<std::vector<Number>> comps_and_qps(n_comps);
        for (unsigned int comp : index_range(n_comps))
          {
            comps_and_qps[comp].resize(xyz_vector.size());
            for (unsigned int qp : index_range(xyz_vector.size()))
              {
                comps_and_qps[comp][qp] =
                  eim_eval.get_parametrized_function().evaluate(comp, elem_id, qp);
              }
          }

        _local_parametrized_functions_for_training[i][elem_id] = comps_and_qps;
      }
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

Number RBEIMConstruction::inner_product(
  const std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> & v,
  const std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> & w)
{
  LOG_SCOPE("inner_product()", "RBEIMConstruction");

  Number val = 0.;

  for (const auto & [elem_id, v_var_and_qp] : v)
    {
      auto w_var_and_qp_it = w.find(elem_id);
      if(w_var_and_qp_it == w.end())
        libmesh_error_msg("Error: elem_id not found");
      const auto & w_var_and_qp = w_var_and_qp_it->second;

      auto _local_quad_point_JxW_it = _local_quad_point_JxW.find(elem_id);
      if(w_var_and_qp_it == w.end())
        libmesh_error_msg("Error: elem_id not found");
      const auto & JxW = _local_quad_point_JxW_it->second;

      for (const auto & var : index_range(v_var_and_qp))
        {
          const std::vector<Number> & v_qp = v_var_and_qp[var];
          const std::vector<Number> & w_qp = w_var_and_qp[var];

          for (unsigned int qp : index_range(JxW))
            val += JxW[qp] * v_qp[qp] * w_qp[qp];
        }
    }

  comm().sum(val);
  return val;
}

Real RBEIMConstruction::get_max_abs_value(const std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> & v) const
{
  LOG_SCOPE("get_max_abs_value()", "RBEIMConstruction");

  Real max_value = 0.;

  for (const auto & [elem_id, v_var_and_qp] : v)
    {
      for (const auto & var : index_range(v_var_and_qp))
        {
          const std::vector<Number> & v_qp = v_var_and_qp[var];
          for (unsigned int qp : index_range(JxW))
            max_value = std::max(max_value, std::abs(v_qp[qp]);
        }
    }

  comm().max(max_value);
  return max_value;
}

void RBEIMConstruction::enrich_eim_approximation(unsigned int training_index)
{
  LOG_SCOPE("enrich_eim_approximation()", "RBEIMConstruction");

  RBEIMEvaluation & eim_eval = get_rb_eim_evaluation();

  set_params_from_training_set(training_index);

  // Make a copy of the parametrized function for training index, since we
  // will modify this below to give us a new basis function.
  auto local_pf = _local_parametrized_functions_for_training[training_index];

  // If we have at least one basis function, then we need to use
  // rb_eim_solve() to find the EIM interpolation error. Otherwise,
  // just use solution as is.
  if (get_rb_eim_evaluation().get_n_basis_functions() > 0)
    {
      // Get the right-hand side vector for the EIM approximation
      // by sampling the parametrized function (stored in solution)
      // at the interpolation points.
      unsigned int RB_size = get_rb_evaluation().get_n_basis_functions();
      DenseVector<Number> EIM_rhs(RB_size);
      for (unsigned int i=0; i<RB_size; i++)
        {
          EIM_rhs(i) =
            get_parametrized_function_value(local_pf,
                                            eim_eval.get_interpolation_points_elem_id(i),
                                            eim_eval.get_interpolation_points_comp(i),
                                            eim_eval.get_interpolation_points_qp(i));
        }

      eim_eval.set_parameters( get_parameters() );
      eim_eval.rb_eim_solve(EIM_rhs);

      // Load the "EIM residual" into solution by subtracting
      // the EIM approximation
      get_rb_eim_evaluation().decrement_vector(local_pf, eim_eval.get_rb_eim_solution());
    }

  // Find the quadrature point at which local_pf (which now stores
  // the "EIM residual") has maximum absolute value
  Number optimal_value = 0.;
  Point optimal_point;
  unsigned int optimal_comp = 0;
  dof_id_type optimal_elem_id = DofObject::invalid_id;
  subdomain_id optimal_subdomain_id = 0;
  unsigned int optimal_qp = 0;

  // Initialize largest_abs_value to be negative so that it definitely gets updated.
  Real largest_abs_value = -1.;

  for (const auto & [elem_id, comp_and_qp] : local_pf)
    {
      for (const auto & comp : index_range(comp_and_qp))
        {
          const std::vector<Number> & qp_values = comp_and_qp_values[comp];

          for (unsigned int qp : index_range(qp_values))
            {
              Number value = qp_values[qp];
              Real abs_value = std::abs(value);

              if (abs_value > largest_abs_value)
                {
                  largest_abs_value = abs_value;
                  optimal_value = value;
                  optimal_comp = comp;
                  optimal_elem_id = elem_id;
                  optimal_qp = qp;

                  auto point_it = _local_quad_point_locations.find(elem_id);
                  if(point_it == _local_quad_point_locations.end())
                    libmesh_error_msg("Error: Invalid element ID");
                  if(qp >= point_it->second.size())
                    libmesh_error_msg("Error: Invalid qp");
                  optimal_point = point_it->second[qp];

                  auto subdomain_it = _local_quad_point_subdomain_ids.find(elem_id);
                  if(subdomain_it == _local_quad_point_subdomain_ids.end())
                    libmesh_error_msg("Error: Invalid element ID");
                  if(qp >= subdomain_it->second.size())
                    libmesh_error_msg("Error: Invalid qp");
                  optimal_subdomain_id = subdomain_it->second[qp];
                }
            }
        }
    }

  // Find out which processor has the largest of the abs values
  // and broadcast from that processor.
  unsigned int proc_ID_index;
  this->comm().maxloc(largest_abs_value, proc_ID_index);

  this->comm().broadcast(optimal_value, proc_ID_index);
  this->comm().broadcast(optimal_point, proc_ID_index);
  this->comm().broadcast(optimal_comp, proc_ID_index);
  this->comm().broadcast(optimal_elem_id, proc_ID_index);
  this->comm().broadcast(optimal_subdomain_id, proc_ID_index);
  this->comm().broadcast(optimal_qp, proc_ID_index);

  if (optimal_elem_id == DofObject::invalid_id)
    libmesh_error_msg("Error: Invalid element ID");

  // Scale local_pf so that its largest value is 1.0
  scale_parametrized_function(local_pf, 1./optimal_value);

  // Add local_pf as the new basis function and store data
  // associated with the interpolation point.
  eim_eval.add_basis_function_and_interpolation_data(local_pf,
                                                     optimal_point,
                                                     optimal_comp,
                                                     optimal_elem_id,
                                                     optimal_subdomain_id,
                                                     optimal_qp);
}

void RBEIMConstruction::update_eim_matrices()
{
  LOG_SCOPE("update_eim_matrices()", "RBEIMConstruction");

  RBEIMEvaluation & eim_eval = get_rb_eim_evaluation();
  unsigned int RB_size = eim_eval.get_n_basis_functions();

  // update the matrix that is used to evaluate L2 projections
  // into the EIM approximation space
  for (unsigned int i=(RB_size-1); i<RB_size; i++)
    {
      for (unsigned int j=0; j<RB_size; j++)
        {
          _eim_projection_matrix(i,j) = inner_product(eim_eval.get_basis_function(i),
                                                      eim_eval.get_basis_function(j));
          if (i!=j)
            {
              // The inner product matrix is assumed to be hermitian
              _eim_projection_matrix(j,i) = libmesh_conj(value);
            }
        }
    }

  // update the EIM interpolation matrix
  for (unsigned int j=0; j<RB_size; j++)
    {
      // Evaluate the basis functions at the new interpolation point in order
      // to update the interpolation matrix
      eim_eval.interpolation_matrix(RB_size-1,j) =
        eim_eval.get_eim_basis_function_value(j,
                                              eim_eval.get_interpolation_points_elem_id(RB_size-1),
                                              eim_eval.get_interpolation_points_comp(RB_size-1),
                                              eim_eval.get_interpolation_points_qp(RB_size-1));
    }
}

Real RBEIMConstruction::compute_best_fit_error(unsigned int training_index)
{
  LOG_SCOPE("compute_best_fit_error()", "RBEIMConstruction");

  set_params_from_training_set(training_index);

  // Make a copy of the pre-computed solution for the specified training sample
  // since we will modify it below to compute the best fit error.
  std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> solution =
    _local_parametrized_functions_for_training[training_index];

  const unsigned int RB_size = get_rb_evaluation().get_n_basis_functions();
  DenseVector<Number> best_fit_coeffs;

  switch(best_fit_type_flag)
    {
    case(PROJECTION_BEST_FIT):
      {
        // Perform an L2 projection in order to find the best approximation to
        // the parametrized function from the current EIM space.
        DenseVector<Number> best_fit_rhs(RB_size);
        for (unsigned int i=0; i<RB_size; i++)
          {
            best_fit_rhs(i) = inner_product(solution, get_rb_eim_evaluation().get_basis_function(i));
          }

        // Now compute the best fit by an LU solve
        DenseMatrix<Number> RB_inner_product_matrix_N(RB_size);
        _eim_projection_matrix.get_principal_submatrix(RB_size, RB_inner_product_matrix_N);

        RB_inner_product_matrix_N.lu_solve(best_fit_rhs, best_fit_coeffs);
        break;
      }
    case(EIM_BEST_FIT):
      {
        // Perform EIM solve in order to find the approximation to solution
        // (rb_eim_solve provides the EIM basis function coefficients used below)

        // Turn off error estimation for this rb_eim_solve, we use the linfty norm instead
        get_rb_eim_evaluation().evaluate_RB_error_bound = false;
        get_rb_eim_evaluation().set_parameters( get_parameters() );
        get_rb_eim_evaluation().rb_eim_solve(RB_size);
        get_rb_eim_evaluation().evaluate_RB_error_bound = true;

        best_fit_coeffs = get_rb_eim_evaluation().get_rb_eim_solution();
        break;
      }
    default:
      libmesh_error_msg("Should not reach here");
    }

  get_rb_eim_evaluation().decrement_vector(solution, best_fit_coeffs);

  Real best_fit_error = get_max_abs_value(solution);
  return best_fit_error;
}

void RBEIMConstruction::scale_parametrized_function(
    std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> & local_pf,
    Number scaling_factor)
{
  LOG_SCOPE("scale_parametrized_function()", "RBEIMConstruction");

  for (auto & [elem_id, comp_and_qp] : local_pf)
    {
      for (auto & comp : index_range(comp_and_qp))
        {
          std::vector<Number> & qp_values = comp_and_qp_values[comp];

          for (unsigned int qp : index_range(qp_values))
            {
              qp_values[qp] *= scaling_factor;
            }
        }
    }
}

} // namespace libMesh
