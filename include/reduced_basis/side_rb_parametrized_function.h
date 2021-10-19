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

#ifndef LIBMESH_SIDE_RB_PARAMETRIZED_FUNCTION_H
#define LIBMESH_SIDE_RB_PARAMETRIZED_FUNCTION_H

// libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/rb_parametrized_function_base.h"

// C++ includes
#include <unordered_map>
#include <vector>
#include <map>

namespace libMesh
{

class RBParameters;
class Point;
class System;

/**
 * A simple functor class that provides a RBParameter-dependent function.
 * This function is defined on element sides.
 */
class SideRBParametrizedFunction : public RBParametrizedFunctionBase
{
public:

  /**
   * Constructor.
   */
  SideRBParametrizedFunction() = default;

  /**
   * Special functions.
   * - This class can be default copy/move assigned/constructed.
   * - The destructor is defaulted out-of-line.
   */
  SideRBParametrizedFunction (SideRBParametrizedFunction &&) = default;
  SideRBParametrizedFunction (const SideRBParametrizedFunction &) = default;
  SideRBParametrizedFunction & operator= (const SideRBParametrizedFunction &) = default;
  SideRBParametrizedFunction & operator= (SideRBParametrizedFunction &&) = default;
  virtual ~SideRBParametrizedFunction() = default;

  /**
   * Evaluate the parametrized function at the specified point for
   * parameter \p mu.  If requires_xyz_perturbations==false, then
   * xyz_perturb will not be used.
   *
   * In this case we return the value for component \p comp only, but
   * the base class implementation simply calls the vector-returning
   * evaluate() function below and returns the comp'th component, so
   * derived classes should provide a more efficient routine or just call
   * the vector-returning function instead.
   */
  virtual Number evaluate_comp(const RBParameters & mu,
                               unsigned int comp,
                               const Point & xyz,
                               dof_id_type elem_id,
                               unsigned int side_index,
                               unsigned int qp,
                               boundary_id_type boundary_id,
                               const std::vector<Point> & xyz_perturb,
                               const std::vector<Real> & phi_i_qp);

  /**
   * Evaluate the parametrized function at the specified point for
   * parameter \p mu.  If requires_xyz_perturbations==false, then
   * xyz_perturb will not be used.
   *
   * In this case we evaluate for all components.
   */
  virtual std::vector<Number> evaluate(const RBParameters & mu,
                                       const Point & xyz,
                                       dof_id_type elem_id,
                                       unsigned int side_index,
                                       unsigned int qp,
                                       boundary_id_type boundary_id,
                                       const std::vector<Point> & xyz_perturb,
                                       const std::vector<Real> & phi_i_qp) = 0;

  /**
   * Vectorized version of evaluate. If requires_xyz_perturbations==false, then all_xyz_perturb will not be used.
   */
  virtual void vectorized_evaluate(const std::vector<RBParameters> & mus,
                                   const std::vector<Point> & all_xyz,
                                   const std::vector<dof_id_type> & elem_ids,
                                   const std::vector<unsigned int> & side_indices,
                                   const std::vector<unsigned int> & qps,
                                   const std::vector<boundary_id_type> & boundary_ids,
                                   const std::vector<std::vector<Point>> & all_xyz_perturb,
                                   const std::vector<std::vector<Real>> & phi_i_qp,
                                   std::vector<std::vector<std::vector<Number>>> & output);

  /**
   * Store the result of vectorized_evaluate. This is helpful during EIM training,
   * since we can pre-evaluate and store the parameterized function for each training
   * sample. If requires_xyz_perturbations==false, then all_xyz_perturb will not be used.
   */
  virtual void preevaluate_parametrized_function_on_mesh(const RBParameters & mu,
                                                         const std::unordered_map<std::pair<dof_id_type,unsigned int>, std::vector<Point>> & all_xyz,
                                                         const std::unordered_map<std::pair<dof_id_type,unsigned int>, boundary_id_type> & boundary_ids,
                                                         const std::unordered_map<std::pair<dof_id_type,unsigned int>, std::vector<std::vector<Point>> > & all_xyz_perturb,
                                                         const System & sys);

  /**
   * Look up the preevaluate values of the parametrized function for
   * component \p comp, element \p elem_id, and quadrature point \p qp.
   */
  virtual Number lookup_preevaluated_value_on_mesh(unsigned int comp,
                                                   dof_id_type elem_id,
                                                   unsigned int side_index,
                                                   unsigned int qp) const;

  /**
   * Storage for pre-evaluated values. The indexing is given by:
   *   parameter index --> point index --> component index --> value.
   */
  std::vector<std::vector<std::vector<Number>>> preevaluated_values;

  /**
   * Indexing into preevaluated_values for the case where the preevaluated values
   * were obtained from evaluations at elements/quadrature points on a mesh.
   * The indexing here is:
   *   (elem_id,side index) --> qp --> point_index
   * Then preevaluated_values[0][point_index] provides the vector of component values at
   * that point.
   */
  std::unordered_map<std::pair<dof_id_type,unsigned int>, std::vector<unsigned int>> mesh_to_preevaluated_values_map;

};

}

#endif // LIBMESH_RB_PARAMETRIZED_FUNCTION_H
