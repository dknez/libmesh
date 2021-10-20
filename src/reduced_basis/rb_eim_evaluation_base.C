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

// rbOOmit includes
#include "libmesh/rb_eim_evaluation_base.h"
#include "libmesh/rb_eim_theta.h"
#include "libmesh/rb_parametrized_function.h"
#include "libmesh/rb_evaluation.h"
#include "libmesh/utility.h" // Utility::mkdir

// libMesh includes
#include "libmesh/xdr_cxx.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/system.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/quadrature.h"
#include "timpi/parallel_implementation.h"

// C++ includes
#include <sstream>
#include <fstream>
#include <numeric> // std::accumulate
#include <iterator> // std::advance

namespace libMesh
{

RBEIMEvaluationBase::RBEIMEvaluationBase(const Parallel::Communicator & comm)
:
ParallelObject(comm),
_rb_eim_solves_N(0),
_preserve_rb_eim_solutions(false)
{
}

RBEIMEvaluationBase::~RBEIMEvaluationBase() = default;

void RBEIMEvaluationBase::clear()
{
  _interpolation_points_xyz.clear();
  _interpolation_points_comp.clear();
  _interpolation_points_subdomain_id.clear();
  _interpolation_points_xyz_perturbations.clear();
  _interpolation_points_elem_id.clear();
  _interpolation_points_qp.clear();
  _interpolation_points_phi_i_qp.clear();

  _interpolation_matrix.resize(0,0);

  // Delete any RBTheta objects that were created
  _rb_eim_theta_objects.clear();
}

void RBEIMEvaluationBase::resize_data_structures(const unsigned int Nmax)
{
  // Resize the data structures relevant to the EIM system
  _interpolation_points_xyz.clear();
  _interpolation_points_comp.clear();
  _interpolation_points_subdomain_id.clear();
  _interpolation_points_xyz_perturbations.clear();
  _interpolation_points_elem_id.clear();
  _interpolation_points_qp.clear();
  _interpolation_points_phi_i_qp.clear();

  _interpolation_matrix.resize(Nmax,Nmax);
}

DenseVector<Number> RBEIMEvaluationBase::rb_eim_solve(DenseVector<Number> & EIM_rhs)
{
  LOG_SCOPE("rb_eim_solve()", "RBEIMEvaluationBase");

  libmesh_error_msg_if(EIM_rhs.size() > get_n_basis_functions(),
                       "Error: N cannot be larger than the number of basis functions in rb_solve");

  libmesh_error_msg_if(EIM_rhs.size()==0, "Error: N must be greater than 0 in rb_solve");

  const unsigned int N = EIM_rhs.size();
  DenseVector<Number> rb_eim_solution(N);
  DenseMatrix<Number> interpolation_matrix_N;
  _interpolation_matrix.get_principal_submatrix(N, interpolation_matrix_N);

  interpolation_matrix_N.lu_solve(EIM_rhs, rb_eim_solution);

  return rb_eim_solution;
}

void RBEIMEvaluationBase::rb_eim_solves(RBParametrizedFunction & parametrized_function,
                                        const std::vector<RBParameters> & mus,
                                        unsigned int N)
{
  if (_preserve_rb_eim_solutions)
    {
      // In this case we preserve _rb_eim_solutions and hence we
      // just return immediately so that we skip updating
      // _rb_eim_solutions below. This is relevant in cases where
      // we set up _rb_eim_solutions elsewhere and we don't want
      // to override it.
      return;
    }

  libmesh_error_msg_if(N > get_n_basis_functions(),
    "Error: N cannot be larger than the number of basis functions in rb_eim_solves");
  libmesh_error_msg_if(N==0, "Error: N must be greater than 0 in rb_eim_solves");

  // If mus and N are the same as before, then we return early
  if ((_rb_eim_solves_mus == mus) && (_rb_eim_solves_N == N))
    return;

  LOG_SCOPE("rb_eim_solves()", "RBEIMEvaluationBase");

  _rb_eim_solves_mus = mus;
  _rb_eim_solves_N = N;

  if (parametrized_function.is_lookup_table)
    {
      _rb_eim_solutions.resize(mus.size());
      for (auto mu_index : index_range(mus))
        {
          Real lookup_table_param =
            mus[mu_index].get_value(parametrized_function.lookup_table_param_name);

          // Cast lookup_table_param to an unsigned integer so that we can use
          // it as an index into the EIM rhs values obtained from the lookup table.
          unsigned int lookup_table_index =
            cast_int<unsigned int>(std::round(lookup_table_param));

          DenseVector<Number> values;
          _eim_solutions_for_training_set[lookup_table_index].get_principal_subvector(N, values);
          _rb_eim_solutions[mu_index] = values;
        }

      return;
    }

  // output all comps indexing is as follows:
  //   mu index --> interpolation point index --> component index --> value.
  std::vector<std::vector<std::vector<Number>>> output_all_comps;
  parametrized_function.vectorized_evaluate(mus,
                                            _interpolation_points_xyz,
                                            _interpolation_points_elem_id,
                                            _interpolation_points_qp,
                                            _interpolation_points_subdomain_id,
                                            _interpolation_points_xyz_perturbations,
                                            _interpolation_points_phi_i_qp,
                                            output_all_comps);

  std::vector<std::vector<Number>> evaluated_values_at_interp_points(output_all_comps.size());

  for(unsigned int mu_index : index_range(evaluated_values_at_interp_points))
    {
      evaluated_values_at_interp_points[mu_index].resize(N);
      for(unsigned int interp_pt_index=0; interp_pt_index<N; interp_pt_index++)
        {
          unsigned int comp = _interpolation_points_comp[interp_pt_index];

          evaluated_values_at_interp_points[mu_index][interp_pt_index] =
            output_all_comps[mu_index][interp_pt_index][comp];
        }
    }

  DenseMatrix<Number> interpolation_matrix_N;
  _interpolation_matrix.get_principal_submatrix(N, interpolation_matrix_N);

  _rb_eim_solutions.resize(mus.size());
  for(unsigned int mu_index : index_range(mus))
    {
      DenseVector<Number> EIM_rhs(N);
      for (unsigned int i=0; i<N; i++)
        {
          EIM_rhs(i) = evaluated_values_at_interp_points[mu_index][i];
        }

      interpolation_matrix_N.lu_solve(EIM_rhs, _rb_eim_solutions[mu_index]);
    }
}

void RBEIMEvaluationBase::initialize_eim_theta_objects()
{
  // Initialize the rb_theta objects that access the solution from this rb_eim_evaluation
  _rb_eim_theta_objects.clear();
  for (auto i : make_range(get_n_basis_functions()))
    _rb_eim_theta_objects.emplace_back(build_eim_theta(i));
}

std::vector<std::unique_ptr<RBTheta>> & RBEIMEvaluationBase::get_eim_theta_objects()
{
  return _rb_eim_theta_objects;
}

void RBEIMEvaluationBase::set_rb_eim_solutions(const std::vector<DenseVector<Number>> & rb_eim_solutions)
{
  _rb_eim_solutions = rb_eim_solutions;
}

const std::vector<DenseVector<Number>> & RBEIMEvaluationBase::get_rb_eim_solutions() const
{
  return _rb_eim_solutions;
}

std::vector<Number> RBEIMEvaluationBase::get_rb_eim_solutions_entries(unsigned int index) const
{
  LOG_SCOPE("get_rb_eim_solutions_entries()", "RBEIMEvaluationBase");

  std::vector<Number> rb_eim_solutions_entries(_rb_eim_solutions.size());
  for (unsigned int mu_index : index_range(_rb_eim_solutions))
    {
      libmesh_error_msg_if(index >= _rb_eim_solutions[mu_index].size(), "Error: Invalid index");
      rb_eim_solutions_entries[mu_index] = _rb_eim_solutions[mu_index](index);
    }

  return rb_eim_solutions_entries;
}

const std::vector<DenseVector<Number>> & RBEIMEvaluationBase::get_eim_solutions_for_training_set() const
{
  return _eim_solutions_for_training_set;
}

std::vector<DenseVector<Number>> & RBEIMEvaluationBase::get_eim_solutions_for_training_set()
{
  return _eim_solutions_for_training_set;
}

void RBEIMEvaluationBase::add_interpolation_points_xyz(Point p)
{
  _interpolation_points_xyz.emplace_back(p);
}

void RBEIMEvaluationBase::add_interpolation_points_comp(unsigned int comp)
{
  _interpolation_points_comp.emplace_back(comp);
}

void RBEIMEvaluationBase::add_interpolation_points_subdomain_id(subdomain_id_type sbd_id)
{
  _interpolation_points_subdomain_id.emplace_back(sbd_id);
}

void RBEIMEvaluationBase::add_interpolation_points_xyz_perturbations(const std::vector<Point> & perturbs)
{
  _interpolation_points_xyz_perturbations.emplace_back(perturbs);
}

void RBEIMEvaluationBase::add_interpolation_points_elem_id(dof_id_type elem_id)
{
  _interpolation_points_elem_id.emplace_back(elem_id);
}

void RBEIMEvaluationBase::add_interpolation_points_qp(unsigned int qp)
{
  _interpolation_points_qp.emplace_back(qp);
}

void RBEIMEvaluationBase::add_interpolation_points_phi_i_qp(const std::vector<Real> & phi_i_qp)
{
  _interpolation_points_phi_i_qp.emplace_back(phi_i_qp);
}

Point RBEIMEvaluationBase::get_interpolation_points_xyz(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_xyz.size(), "Error: Invalid index");

  return _interpolation_points_xyz[index];
}

unsigned int RBEIMEvaluationBase::get_interpolation_points_comp(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_comp.size(), "Error: Invalid index");

  return _interpolation_points_comp[index];
}

subdomain_id_type RBEIMEvaluationBase::get_interpolation_points_subdomain_id(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_subdomain_id.size(), "Error: Invalid index");

  return _interpolation_points_subdomain_id[index];
}

const std::vector<Point> & RBEIMEvaluationBase::get_interpolation_points_xyz_perturbations(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_xyz_perturbations.size(), "Error: Invalid index");

  return _interpolation_points_xyz_perturbations[index];
}

dof_id_type RBEIMEvaluationBase::get_interpolation_points_elem_id(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_elem_id.size(), "Error: Invalid index");

  return _interpolation_points_elem_id[index];
}

unsigned int RBEIMEvaluationBase::get_interpolation_points_qp(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_qp.size(), "Error: Invalid index");

  return _interpolation_points_qp[index];
}

const std::vector<Real> & RBEIMEvaluationBase::get_interpolation_points_phi_i_qp(unsigned int index) const
{
  libmesh_error_msg_if(index >= _interpolation_points_phi_i_qp.size(), "Error: Invalid index");

  return _interpolation_points_phi_i_qp[index];
}

void RBEIMEvaluationBase::set_interpolation_matrix_entry(unsigned int i, unsigned int j, Number value)
{
  libmesh_error_msg_if((i >= _interpolation_matrix.m()) || (j >= _interpolation_matrix.n()),
                       "Error: Invalid matrix indices");

  _interpolation_matrix(i,j) = value;
}

const DenseMatrix<Number> & RBEIMEvaluationBase::get_interpolation_matrix() const
{
  return _interpolation_matrix;
}

void RBEIMEvaluationBase::set_observation_points(const std::vector<Point> & observation_points_xyz)
{
  _observation_points_xyz = observation_points_xyz;
}

unsigned int RBEIMEvaluationBase::get_n_observation_points() const
{
  return _observation_points_xyz.size();
}

const std::vector<Point> & RBEIMEvaluationBase::get_observation_points() const
{
  return _observation_points_xyz;
}

const std::vector<Number> & RBEIMEvaluationBase::get_observation_values(unsigned int bf_index, unsigned int obs_pt_index) const
{
  libmesh_error_msg_if(bf_index >= _observation_points_values.size(), "Invalid basis function index: " << bf_index);
  libmesh_error_msg_if(obs_pt_index >= _observation_points_values[bf_index].size(), "Invalid observation point index: " << obs_pt_index);

  return _observation_points_values[bf_index][obs_pt_index];
}

const std::vector<std::vector<std::vector<Number>>> & RBEIMEvaluationBase::get_observation_values() const
{
  return _observation_points_values;
}

void RBEIMEvaluationBase::set_preserve_rb_eim_solutions(bool preserve_rb_eim_solutions)
{
  _preserve_rb_eim_solutions = preserve_rb_eim_solutions;
}

bool RBEIMEvaluationBase::get_preserve_rb_eim_solutions() const
{
  return _preserve_rb_eim_solutions;
}

void RBEIMEvaluationBase::add_observation_values_for_basis_function(const std::vector<std::vector<Number>> & values)
{
  _observation_points_values.emplace_back(values);
}

void RBEIMEvaluationBase::set_observation_values(const std::vector<std::vector<std::vector<Number>>> & values)
{
  _observation_points_values = values;
}

std::set<unsigned int> RBEIMEvaluationBase::get_eim_vars_to_project_and_write() const
{
  return std::set<unsigned int>();
}

bool RBEIMEvaluationBase::scale_components_in_enrichment() const
{
  // Return false by default, but we override this in subclasses
  // where the parametrized function components differ widely in
  // magnitude.
  return false;
}

} // namespace libMesh
