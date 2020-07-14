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
 * generate an affine approximation to non-affine
 * operators.
 *
 * \author David J. Knezevic
 * \date 2010
 */
class RBEIMConstruction : public System
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
  virtual Real train_eim_approximation();

  /**
   * Load the truth representation of the parametrized function
   * at the current parameters into the solution vector.
   * The truth representation is the projection of
   * parametrized_function into the finite element space.
   * If \p plot_solution > 0 the solution will be plotted
   * to an output file.
   */
  virtual Real truth_solve(int plot_solution) override;

  /**
   * We compute the best fit of parametrized_function
   * into the EIM space and then evaluate the error
   * in the norm defined by inner_product_matrix.
   *
   * \returns The error in the best fit
   */
  virtual Real compute_best_fit_error();

  /**
   * Initialize \p c based on \p sys.
   */
  virtual void init_context_with_sys(FEMContext & c, System & sys);

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
   * @return true if the most recent truth solve gave a zero solution.
   */
  bool check_if_zero_truth_solve() override;

  //----------- PUBLIC DATA MEMBERS -----------//

  /**
   * Enum that indicates which type of "best fit" algorithm
   * we should use.
   * a) projection: Find the best fit in the inner product
   * b) eim: Use empirical interpolation to find a "best fit"
   *
   * \returns The error associated with the "best fit" in the
   * norm induced by inner_product_matrix.
   */
  BEST_FIT_TYPE best_fit_type_flag;

protected:

  /**
   * Add a new basis function to the EIM approximation.
   */
  virtual void enrich_eim_approximation();

  /**
   * Update the matrices used in training the EIM approximation.
   */
  virtual void update_eim_matrices();

  /**
   * Override to return the best fit error. This function is used in
   * the Greedy algorithm to select the next parameter.
   */
  virtual Real get_eim_error_bound();

  /**
   * Pre-request FE data needed for calculations.
   */
  virtual void init_context(FEMContext &) override;

  /**
   * Loop over the training set and compute the parametrized function for each
   * training index.
   */
  void initialize_parametrized_functions_in_training_set();

  /**
   * Boolean flag to indicate whether or not we have called
   * compute_parametrized_functions_in_training_set() yet.
   */
  bool _parametrized_functions_in_training_set_initialized;

  /**
   * The libMesh vectors storing the finite element coefficients
   * of the RB basis functions.
   */
  std::vector<std::unique_ptr<NumericVector<Number>>> _parametrized_functions_in_training_set;

private:

  /**
   * The vector of assembly objects that are created to point to
   * this RBEIMConstruction.
   */
  std::vector<std::unique_ptr<ElemAssembly>> _rb_eim_assembly_objects;

  /**
   * The point locator tolerance.
   */
  Real _point_locator_tol;
};

} // namespace libMesh

#endif // LIBMESH_RB_EIM_CONSTRUCTION_H
