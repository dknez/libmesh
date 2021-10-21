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

#ifndef LIBMESH_SIDE_RB_EIM_CONSTRUCTION_H
#define LIBMESH_SIDE_RB_EIM_CONSTRUCTION_H

// rbOOmit includes
#include "libmesh/rb_eim_construction_base.h"
#include "libmesh/rb_assembly_expansion.h"
#include "libmesh/rb_eim_assembly.h"
#include "libmesh/rb_eim_evaluation.h"

// libMesh includes
#include "libmesh/mesh_function.h"
#include "libmesh/coupling_matrix.h"

// C++ includes
#include <unordered_map>

namespace libMesh
{

/**
 * This class is part of the rbOOmit framework.
 *
 * SideRBEIMConstruction implements the Construction stage of the
 * Empirical Interpolation Method (EIM). This can be used to
 * create an approximation to parametrized functions. In the context
 * of the reduced basis (RB) method, the EIM approximation is typically
 * used to create an affine approximation to non-affine operators,
 * so that the standard RB method can be applied in that case.
 */
class SideRBEIMConstruction : public RBEIMConstructionBase
{
public:

  /**
   * Constructor.  Optionally initializes required
   * data structures.
   */
  SideRBEIMConstruction (EquationSystems & es,
                         const std::string & name,
                         const unsigned int number);

  /**
   * Special functions.
   * - This class has the same restrictions/defaults as its base class.
   * - Destructor is defaulted out-of-line
   */
  SideRBEIMConstruction (SideRBEIMConstruction &&) = default;
  SideRBEIMConstruction (const SideRBEIMConstruction &) = delete;
  SideRBEIMConstruction & operator= (const SideRBEIMConstruction &) = delete;
  SideRBEIMConstruction & operator= (SideRBEIMConstruction &&) = delete;
  virtual ~SideRBEIMConstruction ();

  /**
   * Type of the data structure used to map from (elem id,side_index) -> [n_vars][n_qp] data.
   */
  typedef SideRBEIMEvaluation::SideQpDataMap SideQpDataMap;

  /**
   * Clear this object.
   */
  virtual void clear() override;

  /**
   * Print out info that describes the current setup of this RBConstruction.
   */
  virtual void print_info();

  /**
   * Set the RBEIMEvaluation object.
   */
  void set_rb_eim_evaluation(SideRBEIMEvaluation & rb_eim_eval_in);

  /**
   * Get a reference to the RBEvaluation object.
   */
  SideRBEIMEvaluation & get_rb_eim_evaluation();

  /**
   * Get a const reference to the RBEvaluation object.
   */
  const SideRBEIMEvaluation & get_rb_eim_evaluation() const;

  /**
   * Get the EIM solution vector at all parametrized functions in the training
   * set. In some cases we want to store this data for future use. For example
   * this is useful in the case that the parametrized function is defined
   * based on a look-up table rather than an analytical function, since
   * if we store the EIM solution data, we can do Online solves without
   * initializing the look-up table data.
   */
  virtual void store_eim_solutions_for_training_set() override;

  /**
   * Get a const reference to the specified parametrized function from
   * the training set.
   */
  const SideQpDataMap & get_parametrized_function_from_training_set(unsigned int training_index) const;

protected:

  /**
   * Add a new basis function to the EIM approximation.
   */
  virtual void enrich_eim_approximation(unsigned int training_index) override;

  /**
   * Update the matrices used in training the EIM approximation.
   */
  virtual void update_eim_matrices() override;

  /**
   * Compute and store the parametrized function for each
   * parameter in the training set at all the stored qp locations.
   */
  virtual void initialize_parametrized_functions_in_training_set() override;

  /**
   * Find the training sample that has the largest EIM approximation error
   * based on the current EIM approximation. Return the maximum error, and
   * the training sample index at which it occured.
   */
  virtual std::pair<Real, unsigned int> compute_max_eim_error() override;

private:

  /**
   * Initialize the data associated with each quad point (location, JxW, etc.)
   * so that we can use this in evaluation of the parametrized functions.
   */
  void initialize_qp_data();

  /**
   * Evaluate the inner product of vec1 and vec2 which specify values at
   * quadrature points. The inner product includes the JxW contributions
   * stored in _local_quad_point_JxW, so that this is equivalent to
   * computing w^t M v, where M is the mass matrix.
   */
  Number inner_product(const SideQpDataMap & v, const SideQpDataMap & w);

  /**
   * Get the maximum absolute value from a vector stored in the format that we use
   * for basis functions.
   */
  Real get_max_abs_value(const SideQpDataMap & v) const;

  /**
   * Scale all values in \p pf by \p scaling_factor
   */
  static void scale_parametrized_function(
    SideQpDataMap & local_pf,
    Number scaling_factor);

  /**
   * The SideRBEIMEvaluation object that we use to perform the EIM training.
   */
  SideRBEIMEvaluation * _rb_eim_eval;

  /**
   * The parametrized functions that are used for training. We pre-compute and
   * store all of these functions, rather than recompute them at each iteration
   * of the training.
   *
   * We store values at quadrature points on elements that are local to this processor.
   * The indexing is as follows:
   *   basis function index --> (element ID,side index) --> variable --> quadrature point --> value
   * We use a map to index the (element ID,side_index), since the IDs on this processor in
   * generally will not start at zero.
   */
  std::vector<SideQpDataMap> _local_parametrized_functions_for_training;

  /**
   * The quadrature point locations, quadrature point weights (JxW), and subdomain IDs
   * on every element local to this processor.
   *
   * The indexing is as follows:
   *   (element ID,side_index) --> quadrature point --> xyz
   *   (element ID,side_index) --> quadrature point --> JxW
   *   (element ID,side_index) --> subdomain_id
   * We use a map to index the (element ID,side_index), since the IDs on this processor in
   * generally will not start at zero.
   */
  std::map<std::pair<dof_id_type,unsigned int>, std::vector<Point> > _local_quad_point_locations;
  std::map<std::pair<dof_id_type,unsigned int>, std::vector<Real> > _local_quad_point_JxW;
  std::map<std::pair<dof_id_type,unsigned int>, subdomain_id_type > _local_quad_point_boundary_ids;

  /**
   * EIM approximations often arise when applying a geometric mapping to a Reduced Basis
   * formulation. In this context, we often need to approximate derivates of the mapping
   * function via EIM. In order to enable this, we also optionally store perturbations
   * about each point in _local_quad_point_locations to enable finite difference approximation
   * to the mapping function derivatives.
   */
  std::unordered_map<std::pair<dof_id_type,unsigned int>, std::vector<std::vector<Point>> > _local_quad_point_locations_perturbations;

};

} // namespace libMesh

#endif // LIBMESH_SIDE_RB_EIM_CONSTRUCTION_H
