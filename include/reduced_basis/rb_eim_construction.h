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

#ifndef LIBMESH_RB_EIM_CONSTRUCTION_H
#define LIBMESH_RB_EIM_CONSTRUCTION_H

// rbOOmit includes
#include "libmesh/rb_construction.h"
#include "libmesh/rb_assembly_expansion.h"
#include "libmesh/rb_eim_assembly.h"

// libMesh includes
#include "libmesh/mesh_function.h"
#include "libmesh/coupling_matrix.h"

// C++ includes

namespace libMesh
{

/**
 * This class is part of the rbOOmit framework.
 *
 * RBEIMConstruction implements the Construction stage of the
 * Empirical Interpolation Method (EIM). This can be used to
 * create an approximation to parametrized functions. In the context
 * of the reduced basis (RB) method, the EIM approximation is typically
 * used to create an affine approximation to non-affine operators,
 * so that the standard RB method can be applied in that case.
 */
class RBEIMConstruction : public RBConstructionBase
{
public:

  enum BEST_FIT_TYPE { PROJECTION_BEST_FIT, EIM_BEST_FIT };

  /**
   * Constructor.  Optionally initializes required
   * data structures.
   */
  RBEIMConstruction (EquationSystems & es,
                     const std::string & name,
                     const unsigned int number);

  /**
   * Destructor.
   */
  virtual ~RBEIMConstruction ();

  /**
   * Clear this object.
   */
  virtual void clear() override;

  /**
   * Perform initialization of this object to prepare for running
   * train_eim_approximation().
   */
  void initalize_eim_construction();

  /**
   * Read parameters in from file and set up this system
   * accordingly.
   */
  virtual void process_parameters_file (const std::string & parameters_filename) override;

  /**
   * Specify which type of "best fit" we use to guide the EIM
   * greedy algorithm.
   */
  void set_best_fit_type_flag (const std::string & best_fit_type_string);

  /**
   * Print out info that describes the current setup of this RBConstruction.
   */
  virtual void print_info() override;

  /**
   * Generate the EIM approximation for the specified parametrized function.
   */
  Real train_eim_approximation();

  /**
   * We compute the best fit of parametrized_function
   * into the EIM space and then evaluate the error
   * in the norm defined by inner_product_matrix.
   *
   * \returns The error in the best fit
   */
  Real compute_best_fit_error();

  /**
   * Build a vector of ElemAssembly objects that accesses the basis
   * functions stored in this RBEIMConstruction object. This is useful
   * for performing the Offline stage of the Reduced Basis method where
   * we want to use assembly functions based on this EIM approximation.
   */
  virtual void initialize_eim_assembly_objects();

  /**
   * \returns The vector of assembly objects that point to this RBEIMConstruction.
   */
  std::vector<std::unique_ptr<ElemAssembly>> & get_eim_assembly_objects();

  /**
   * Build an element assembly object that will access basis function
   * \p bf_index.
   * This is pure virtual, override in subclasses to specify the appropriate
   * ElemAssembly object.
   */
  virtual std::unique_ptr<ElemAssembly> build_eim_assembly(unsigned int bf_index) = 0;

  /**
   * Pre-request FE data needed for calculations.
   */
  virtual void init_context(FEMContext &) override;

  /**
   * Enum that indicates which type of "best fit" algorithm
   * we should use.
   * a) projection: Find the best fit in the inner product
   * b) eim: Use empirical interpolation to find a "best fit"
   */
  BEST_FIT_TYPE best_fit_type_flag;

private:

  /**
   * Compute and store the parametrized function for each
   * parameter in the training set. Then in 
   */
  void initialize_parametrized_functions_in_training_set();

  /**
   * Evaluate the parametrized function at the current parameter values
   * at the quadrature points and store the result in
   * _local_parametrized_functions_for_training.
   * If \p init_quad_point_data, we also store the quadrature point data
   * for the entire mesh in 
   */
  void evaluate_parametrized_function_at_all_qps(unsigned int training_index,
                                                 bool init_quad_point_data);

  /**
   * Add a new basis function to the EIM approximation.
   */
  virtual void enrich_eim_approximation();

  /**
   * Update the matrices used in training the EIM approximation.
   */
  virtual void update_eim_matrices();

  /**
   * The vector of assembly objects that are created to point to
   * this RBEIMConstruction.
   */
  std::vector<std::unique_ptr<ElemAssembly>> _rb_eim_assembly_objects;

  /**
   * The parametrized functions that are used for training. We pre-compute and
   * store all of these functions, rather than recompute them at each iteration
   * of the training.
   *
   * We store values at quadrature points on elements that are local to this processor.
   * The indexing is as follows:
   *   basis function index --> element ID --> quadrature point --> variable --> value
   * We use a map to index the element ID, since the IDs on this processor in
   * generally will not start at zero.
   */
  std::vector<std::unordered_map<dof_id_type, std::vector<std::vector<Number>>> >
    _local_parametrized_functions_for_training;

  /**
   * The quadrature point locations, and quadrature point weights on every element
   * local to this processor.
   *
   * The indexing is as follows:
   *   element ID --> quadrature point --> xyz
   *   element ID --> quadrature point --> weight
   * We use a map to index the element ID, since the IDs on this processor in
   * generally will not start at zero.
   */
  std::unordered_map<dof_id_type, std::vector<Point>> > _local_quad_point_locations;
  std::unordered_map<dof_id_type, std::vector<Point>> > _local_quad_point_weights;

};

} // namespace libMesh

#endif // LIBMESH_RB_EIM_CONSTRUCTION_H
