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
#include <sstream>
#include <fstream>

// rbOOmit includes
#include "libmesh/rb_eim_evaluation.h"
#include "libmesh/rb_eim_theta.h"
#include "libmesh/rb_parametrized_function.h"

// libMesh includes
#include "libmesh/xdr_cxx.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/auto_ptr.h" // libmesh_make_unique

namespace libMesh
{

RBEIMEvaluation::RBEIMEvaluation()
:
evaluate_eim_error_bound(true)
{
}

RBEIMEvaluation::~RBEIMEvaluation()
{
}

void RBEIMEvaluation::clear()
{
  _interpolation_points.clear();
  _interpolation_points_var.clear();
  _interpolation_points_subdomain_id.clear();

  // Delete any RBTheta objects that were created
  _rb_eim_theta_objects.clear();
}

void RBEIMEvaluation::resize_data_structures(const unsigned int Nmax)
{
  // Resize the data structures relevant to the EIM system
  _interpolation_points.clear();
  _interpolation_points_var.clear();
  _interpolation_points_subdomain_id.clear();
  _interpolation_matrix.resize(Nmax,Nmax);
}

void RBEIMEvaluation::attach_parametrized_function(RBParametrizedFunction * pf)
{
  _parametrized_functions.push_back(pf);
}

unsigned int RBEIMEvaluation::get_n_parametrized_functions() const
{
  return cast_int<unsigned int>(_parametrized_functions.size());
}

Number RBEIMEvaluation::evaluate_parametrized_function(unsigned int var_index,
                                                       const Point & p,
                                                       subdomain_id_type subdomain_id)
{
  return _parametrized_functions[var_index]->evaluate(get_parameters(), p, subdomain_id);
}

Real RBEIMEvaluation::eim_solve(unsigned int N)
{
  LOG_SCOPE("rb_solve()", "RBEIMEvaluation");

  if (N > get_n_basis_functions())
    libmesh_error_msg("Error: N cannot be larger than the number of basis functions in rb_solve");

  if (N==0)
    libmesh_error_msg("Error: N must be greater than 0 in rb_solve");

  // Get the rhs by sampling parametrized_function
  // at the first N interpolation_points
  DenseVector<Number> EIM_rhs(N);
  for (unsigned int i=0; i<N; i++)
    {
      EIM_rhs(i) = evaluate_parametrized_function(interpolation_points_var[i],
                                                  interpolation_points[i],
                                                  interpolation_points_subdomain_id[i]);
    }

  DenseMatrix<Number> interpolation_matrix_N;
  _interpolation_matrix.get_principal_submatrix(N, interpolation_matrix_N);

  interpolation_matrix_N.lu_solve(EIM_rhs, _eim_solution);

  // Optionally evaluate an a posteriori error bound. The EIM error estimate
  // recommended in the literature is based on using "next" EIM point, so
  // we skip this if N == get_n_basis_functions()
  if (evaluate_eim_error_bound && (N != get_n_basis_functions()))
    {
      // Compute the a posteriori error bound
      // First, sample the parametrized function at x_{N+1}
      Number g_at_next_x = evaluate_parametrized_function(interpolation_points_var[N],
                                                          interpolation_points[N],
                                                          *interpolation_points_elem[N]);

      // Next, evaluate the EIM approximation at x_{N+1}
      Number EIM_approx_at_next_x = 0.;
      for (unsigned int j=0; j<N; j++)
        {
          EIM_approx_at_next_x += _eim_solution(j) * interpolation_matrix(N,j);
        }

      Real error_estimate = std::abs(g_at_next_x - EIM_approx_at_next_x);

      _previous_error_bound = error_estimate;
      return error_estimate;
    }
  else // Don't evaluate an error bound
    {
      _previous_error_bound = -1.;
      return -1.;
    }

}

void RBEIMEvaluation::rb_solve(DenseVector<Number> & EIM_rhs)
{
  LOG_SCOPE("rb_solve()", "RBEIMEvaluation");

  if (EIM_rhs.size() > get_n_basis_functions())
    libmesh_error_msg("Error: N cannot be larger than the number of basis functions in rb_solve");

  if (EIM_rhs.size()==0)
    libmesh_error_msg("Error: N must be greater than 0 in rb_solve");

  const unsigned int N = EIM_rhs.size();
  DenseMatrix<Number> interpolation_matrix_N;
  _interpolation_matrix.get_principal_submatrix(N, interpolation_matrix_N);

  interpolation_matrix_N.lu_solve(EIM_rhs, _eim_solution);
}

void RBEIMEvaluation::initialize_eim_theta_objects()
{
  // Initialize the rb_theta objects that access the solution from this rb_eim_evaluation
  _rb_eim_theta_objects.clear();
  for (unsigned int i=0; i<get_n_basis_functions(); i++)
    _rb_eim_theta_objects.emplace_back(build_eim_theta(i));
}

std::vector<std::unique_ptr<RBTheta>> & RBEIMEvaluation::get_eim_theta_objects()
{
  return _rb_eim_theta_objects;
}

std::unique_ptr<RBTheta> RBEIMEvaluation::build_eim_theta(unsigned int index)
{
  return libmesh_make_unique<RBEIMTheta>(*this, index);
}

void RBEIMEvaluation::get_eim_basis_function_value_at_qps(unsigned int basis_function_index,
                                                          dof_id_type elem_id,
                                                          unsigned int var,
                                                          std::vector<Number> & values) const
{
  LOG_SCOPE("get_eim_basis_function_value_at_qps", "RBEIMEvaluation");

  values.clear();

  if(basis_function_index >= _local_eim_basis_functions.size())
  {
    libmesh_error_msg("Invalid basis function index: " + std::to_string(basis_function_index));
  }

  const auto & eim_basis_function_map = _local_eim_basis_functions[basis_function_index];

  const auto it = eim_basis_function_map.find(elem_id);
  if(it != eim_basis_function_map.end())
  {
    const auto & vars_and_qps_on_elem = it->second;
    if(var >= vars_and_qps_on_elem.size())
    {
      libmesh_error_msg("Invalid var index: " + std::to_string(var));
    }

    values = vars_and_qps_on_elem[var];
  }
}

}
