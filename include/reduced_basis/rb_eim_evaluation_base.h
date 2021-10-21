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

#ifndef LIBMESH_RB_EIM_EVALUATION_BASE_H
#define LIBMESH_RB_EIM_EVALUATION_BASE_H

// libMesh includes
#include "libmesh/point.h"
#include "libmesh/rb_theta_expansion.h"
#include "libmesh/rb_parametrized.h"
#include "libmesh/parallel_object.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"

// C++ includes
#include <memory>
#include <map>
#include <vector>
#include <string>

namespace libMesh
{

class RBParameters;
class RBParametrizedFunction;
class RBTheta;
class System;
class Elem;

/**
 * This class enables evaluation of an Empirical Interpolation Method (EIM)
 * approximation. RBEvaluation plays an analogous role in the context of
 * the regular reduced basis method.
 *
 * This is the base class. Extend in subclasses to handle EIM on element
 * interiors or sides.
 */
class RBEIMEvaluationBase : public RBParametrized,
                            public ParallelObject
{
public:

  /**
   * Constructor.
   */
  RBEIMEvaluationBase(const Parallel::Communicator & comm);

  /**
   * Special functions.
   * - This class contains unique_ptrs, so it can't be default copy
       constructed/assigned.
   * - The destructor is defaulted out of line.
   */
  RBEIMEvaluationBase (RBEIMEvaluationBase &&) = default;
  RBEIMEvaluationBase (const RBEIMEvaluationBase &) = delete;
  RBEIMEvaluationBase & operator= (const RBEIMEvaluationBase &) = delete;
  RBEIMEvaluationBase & operator= (RBEIMEvaluationBase &&) = default;
  virtual ~RBEIMEvaluationBase ();

  /**
   * Clear this object.
   */
  virtual void clear() override;

  /**
   * Resize the data structures for storing data associated
   * with this object.
   */
  void resize_data_structures(const unsigned int Nmax);

  /**
   * Calculate the EIM approximation for the given
   * right-hand side vector \p EIM_rhs. Store the
   * solution coefficients in the member _eim_solution.
   */
  DenseVector<Number> rb_eim_solve(DenseVector<Number> & EIM_rhs);

  /**
   * Perform rb_eim_solves at each mu in \p mus and store the results
   * in _rb_eim_solutions.
   */
  void rb_eim_solves(RBParametrizedFunction & parametrized_function,
                     const std::vector<RBParameters> & mus,
                     unsigned int N);

  /**
   * Return the current number of EIM basis functions.
   * Override in sub-classes based on the type of EIM that
   * is being performed.
   */
  virtual unsigned int get_n_basis_functions() const = 0;

  /**
   * Set the number of basis functions. Useful when reading in
   * stored data.
   * Override in sub-classes based on the type of EIM that
   * is being performed.
   */
  virtual void set_n_basis_functions(unsigned int n_bfs) = 0;

  /**
   * Build a vector of RBTheta objects that accesses the components
   * of the RB_solution member variable of this RBEvaluation.
   * Store these objects in the member vector rb_theta_objects.
   */
  void initialize_eim_theta_objects();

  /**
   * \returns The vector of theta objects that point to this RBEIMEvaluation.
   */
  std::vector<std::unique_ptr<RBTheta>> & get_eim_theta_objects();

  /**
   * Build a theta object corresponding to EIM index \p index.
   * The default implementation builds an RBEIMTheta object, possibly
   * override in subclasses if we need more specialized behavior.
   */
  virtual std::unique_ptr<RBTheta> build_eim_theta(unsigned int index) = 0;

  /**
   * Set _rb_eim_solutions. Normally we update _rb_eim_solutions by performing
   * and EIM solve, but in some cases we want to set the EIM solution coefficients
   * elsewhere, so this setter enables us to do that.
   */
  void set_rb_eim_solutions(const std::vector<DenseVector<Number>> & rb_eim_solutions);

  /**
   * Return the EIM solution coefficients from the most recent call to rb_eim_solves().
   */
  const std::vector<DenseVector<Number>> & get_rb_eim_solutions() const;

  /**
   * Return entry \p index for each solution in _rb_eim_solutions.
   */
  std::vector<Number> get_rb_eim_solutions_entries(unsigned int index) const;

  /**
   * Return a const reference to the EIM solutions for the parameters in the training set.
   */
  const std::vector<DenseVector<Number>> & get_eim_solutions_for_training_set() const;

  /**
   * Return a writeable reference to the EIM solutions for the parameters in the training set.
   */
  std::vector<DenseVector<Number>> & get_eim_solutions_for_training_set();

  /**
   * Set the data associated with EIM interpolation points.
   */
  void add_interpolation_points_xyz(Point p);
  void add_interpolation_points_comp(unsigned int comp);
  void add_interpolation_points_subdomain_id(subdomain_id_type sbd_id);
  void add_interpolation_points_xyz_perturbations(const std::vector<Point> & perturbs);
  void add_interpolation_points_elem_id(dof_id_type elem_id);
  void add_interpolation_points_qp(unsigned int qp);
  void add_interpolation_points_phi_i_qp(const std::vector<Real> & phi_i_qp);

  /**
   * Get the data associated with EIM interpolation points.
   */
  Point get_interpolation_points_xyz(unsigned int index) const;
  unsigned int get_interpolation_points_comp(unsigned int index) const;
  subdomain_id_type get_interpolation_points_subdomain_id(unsigned int index) const;
  const std::vector<Point> & get_interpolation_points_xyz_perturbations(unsigned int index) const;
  dof_id_type get_interpolation_points_elem_id(unsigned int index) const;
  unsigned int get_interpolation_points_qp(unsigned int index) const;
  const std::vector<Real> & get_interpolation_points_phi_i_qp(unsigned int index) const;

  /**
   * Set entry of the EIM interpolation matrix.
   */
  void set_interpolation_matrix_entry(unsigned int i, unsigned int j, Number value);

  /**
   * Get the EIM interpolation matrix.
   */
  const DenseMatrix<Number> & get_interpolation_matrix() const;

  /**
   * Set the observation points and components.
   */
  void set_observation_points(const std::vector<Point> & observation_points_xyz);

  /**
   * Get the number of observation points.
   */
  unsigned int get_n_observation_points() const;

  /**
   * Get the observation points.
   */
  const std::vector<Point> & get_observation_points() const;

  /**
   * Get the observation value for the specified basis function and observation point.
   */
  const std::vector<Number> & get_observation_values(unsigned int bf_index, unsigned int obs_pt_index) const;

  /**
   * Get a const reference to all the observation values, indexed as follows:
   *  basis_function index --> observation point index --> value.
   */
  const std::vector<std::vector<std::vector<Number>>> & get_observation_values() const;

  /**
   * Add values at the observation points for a new basis function.
   */
  void add_observation_values_for_basis_function(const std::vector<std::vector<Number>> & values);

  /**
   * Set all observation values.
   */
  void set_observation_values(const std::vector<std::vector<std::vector<Number>>> & values);

  /**
   * Set _preserve_rb_eim_solutions.
   */
  void set_preserve_rb_eim_solutions(bool preserve_rb_eim_solutions);

  /**
   * Get _preserve_rb_eim_solutions.
   */
  bool get_preserve_rb_eim_solutions() const;

  /**
   * Return a set that specifies which EIM variables will be projected
   * and written out in write_out_projected_basis_functions().
   * By default this returns an empty vector, but can be overridden in
   * subclasses to specify the EIM variables that are relevant for visualization.
   */
  virtual std::set<unsigned int> get_eim_vars_to_project_and_write() const;

  /**
   * Indicate whether we should apply scaling to the components of the parametrized
   * function during basis function enrichment in order give an approximately uniform
   * magnitude for all components. This is helpful in cases where the components vary
   * widely in magnitude.
   */
  virtual bool scale_components_in_enrichment() const;

protected:

  /**
   * The EIM solution coefficients from the most recent call to rb_eim_solves().
   */
  std::vector<DenseVector<Number>> _rb_eim_solutions;

  /**
   * Storage for EIM solutions from the training set. This is typically used in
   * the case that we have is_lookup_table==true in our RBParametrizedFunction,
   * since in that case we need to store all the EIM solutions on the training
   * set so that we do not always need to refer to the lookup table itself
   * (since in some cases, like in the Online stage, the lookup table is not
   * available).
   */
  std::vector<DenseVector<Number>> _eim_solutions_for_training_set;

  /**
   * The parameters and the number of basis functions that were used in the
   * most recent call to rb_eim_solves(). We store this so that we can
   * check if we can skip calling rb_eim_solves() again if the inputs
   * haven't changed.
   */
  std::vector<RBParameters> _rb_eim_solves_mus;
  unsigned int _rb_eim_solves_N;

  /**
   * Dense matrix that stores the lower triangular
   * interpolation matrix that can be used
   */
  DenseMatrix<Number> _interpolation_matrix;

  /**
   * We need to store interpolation point data in order to
   * evaluate parametrized functions at the interpolation points.
   * This requires the xyz locations, the components to evaluate,
   * and the subdomain IDs.
   */
  std::vector<Point> _interpolation_points_xyz;
  std::vector<unsigned int> _interpolation_points_comp;
  std::vector<subdomain_id_type> _interpolation_points_subdomain_id;

  /**
   * We also store perturbations of the xyz locations that may be
   * needed to evaluate finite difference approximations to derivatives.
   */
  std::vector<std::vector<Point>> _interpolation_points_xyz_perturbations;

  /**
   * We also store the element ID and qp index of each interpolation
   * point so that we can evaluate our basis functions at these
   * points by simply looking up the appropriate stored values.
   * This data is only needed during the EIM training.
   */
  std::vector<dof_id_type> _interpolation_points_elem_id;
  std::vector<unsigned int> _interpolation_points_qp;

  /**
   * If the EIM approximation applies to element sides, then we need to
   * store the side index and boundary ID for each quadrature point.
   */
  std::vector<unsigned int> _interpolation_points_side_index;
  std::vector<boundary_id_type> _interpolation_points_boundary_id;

  /**
   * We store the shape function values at the qp as well. These values
   * allows us to evaluate parametrized functions that depend on nodal
   * data.
   */
  std::vector<std::vector<Real>> _interpolation_points_phi_i_qp;

  /**
   * The vector of RBTheta objects that are created to point to
   * this RBEIMEvaluationBase.
   */
  std::vector<std::unique_ptr<RBTheta>> _rb_eim_theta_objects;

  /**
   * Let {p_1,...,p_n} be a set of n "observation points", where we can
   * observe the values of our EIM basis functions. Also, let
   * {comp_k} be the components of the EIM basis function that
   * we will observe. Then the corresponding observation values, v_ijk,
   * are given by:
   *  v_ijk = eim_basis_function[i][p_j][comp_k].
   *
   * These observation values can be used to observe the EIM approximation
   * at specific points of interest, where the points of interest are defined
   * by the observation points.
   *
   * _observation_points_value is indexed as follows:
   *  basis_function index --> observation point index --> comp index --> value
   */
  std::vector<Point> _observation_points_xyz;
  std::vector<std::vector<std::vector<Number>>> _observation_points_values;

  /**
   * Boolean to indicate if we skip updating _rb_eim_solutions in rb_eim_solves().
   * This is relevant for cases when we set up _rb_eim_solutions elsewhere and we
   * want to avoid changing it.
   */
  bool _preserve_rb_eim_solutions;

};

}

#endif // LIBMESH_RB_EIM_EVALUATION_H
