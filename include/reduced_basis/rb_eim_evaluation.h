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

#ifndef LIBMESH_RB_EIM_EVALUATION_H
#define LIBMESH_RB_EIM_EVALUATION_H

// libMesh includes
#include "libmesh/point.h"
#include "libmesh/rb_evaluation.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/rb_theta_expansion.h"

// C++ includes
#include <memory>

namespace libMesh
{

class RBParameters;
class RBParametrizedFunction;
class Elem;
class RBTheta;

/**
 * This class enables evaluation of an Empirical Interpolation Method (EIM)
 * approximation. RBEvaluation plays an analogous role in the context of
 * the regular reduced basis method.
 */
class RBEIMEvaluation
{
public:

  /**
   * Constructor.
   */
  RBEIMEvaluation();

  /**
   * Destructor.
   */
  virtual ~RBEIMEvaluation ();

  /**
   * Clear this object.
   */
  virtual void clear() override;

  /**
   * Resize the data structures for storing data associated
   * with this object.
   */
  virtual void resize_data_structures(const unsigned int Nmax,
                                      bool resize_error_bound_data=true) override;

  /**
   * Attach the parametrized function that we will approximate
   * using the Empirical Interpolation Method.
   */
  void attach_parametrized_function(RBParametrizedFunction * pf);


  /**
   * Get the number of parametrized functions that have
   * been attached to this system. Each function will
   * be approximated by a separate variable in our EIM
   * approximation.
   */
  unsigned int get_n_parametrized_functions() const;

  /**
   * Evaluate the parametrized function for each entry in \p var_indices, \p qps,
   * and \p subdomani_ids, and set the corresponding value of the values vector
   * for each evaluation.
   */
  void evaluate_parametrized_function(unsigned int var,
                                      const std::vector<Point> & qps,
                                      const std::vector<subdomain_id_type> & subdomain_ids,
                                      std::vector<Number> & values);

  /**
   * Calculate the EIM approximation to parametrized_function
   * using the first \p N EIM basis functions. Store the
   * solution coefficients in the member _eim_solution.
   * \returns The EIM a posteriori error bound.
   */
  virtual Real eim_solve(unsigned int N);

  /**
   * Calculate the EIM approximation for the given
   * right-hand side vector \p EIM_rhs. Store the
   * solution coefficients in the member _eim_solution.
   */
  void eim_solve(DenseVector<Number> & EIM_rhs);

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
  virtual std::unique_ptr<RBTheta> build_eim_theta(unsigned int index);

  /**
   * Fill up \p values with the basis function values for basis function
   * \p basis_function_index and variable \p var, at all quadrature points
   * on element \p elem_id. Each processor stores data for only the
   * elements local to that processor, so if elem_id is not on this processor
   * then \p values will be empty.
   */
  void get_eim_basis_function_values_at_qps(unsigned int basis_function_index,
                                            dof_id_type elem_id,
                                            unsigned int var,
                                            std::vector<Number> & values) const;

  /**
   * Return a const reference to the EIM solution coefficients from the most
   * recent solve.
   */
  const DenseVector<Number> & get_eim_solution() const;

  /**
   * Boolean to indicate whether we evaluate a posteriori error bounds
   * when eim_solve is called.
   */
  bool evaluate_eim_error_bound;

private:

  /**
   * The EIM solution coefficients from the most recent eim_solve().
   */
  DenseVector<Number> _eim_solution;

  /**
   * Dense matrix that stores the lower triangular
   * interpolation matrix that can be used
   */
  DenseMatrix<Number> _interpolation_matrix;

  /**
   * The list of interpolation points, i.e. locations at
   * which the basis functions are maximized.
   */
  std::vector<Point> _interpolation_points;

  /**
   * The corresponding list of variables indices at which
   * the interpolation points were identified.
   */
  std::vector<unsigned int> _interpolation_points_var;

  /**
   * The corresponding list of subdomain IDs at which
   * the interpolation points were identified.
   */
  std::vector<subdomain_id> _interpolation_points_subdomain_id;

  /**
   * This vector stores the parametrized functions
   * that will be approximated in this EIM system.
   */
  std::vector<RBParametrizedFunction *> _parametrized_functions;

  /**
   * The vector of RBTheta objects that are created to point to
   * this RBEIMEvaluation.
   */
  std::vector<std::unique_ptr<RBTheta>> _rb_eim_theta_objects;

  /**
   * The EIM basis functions. We store values at quadrature points
   * on elements that are local to this processor. The indexing
   * is as follows:
   *   basis function index --> element ID --> variable --> quadrature point --> value
   * We use a map to index the element ID, since the IDs on this processor in
   * generally will not start at zero.
   */
  std::vector<std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> > _local_eim_basis_functions;

};

}

#endif // LIBMESH_RB_EIM_EVALUATION_H
