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

#ifndef LIBMESH_RB_EIM_CONSTRUCTION_BASE_H
#define LIBMESH_RB_EIM_CONSTRUCTION_BASE_H

// rbOOmit includes
#include "libmesh/rb_construction.h"
#include "libmesh/rb_assembly_expansion.h"
#include "libmesh/rb_eim_assembly.h"
#include "libmesh/rb_eim_evaluation.h"

// libMesh includes
#include "libmesh/mesh_function.h"
#include "libmesh/coupling_matrix.h"

// C++ includes
#include <unordered_map>
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace libMesh
{

/**
 * RBEIMConstructionBase implements the Construction stage of the
 * Empirical Interpolation Method (EIM). This can be used to
 * create an approximation to parametrized functions. In the context
 * of the reduced basis (RB) method, the EIM approximation is typically
 * used to create an affine approximation to non-affine operators,
 * so that the standard RB method can be applied in that case.
 *
 * This provides a base class which can be extended to handle various
 * cases such as "EIM on element interiors" or "EIM on element sides".
 */
class RBEIMConstructionBase : public RBConstructionBase<System>
{
public:

  enum BEST_FIT_TYPE { PROJECTION_BEST_FIT, EIM_BEST_FIT };

  /**
   * Constructor.  Optionally initializes required
   * data structures.
   */
  RBEIMConstructionBase (EquationSystems & es,
                         const std::string & name,
                         const unsigned int number);

  /**
   * Special functions.
   * - This class has the same restrictions/defaults as its base class.
   * - Destructor is defaulted out-of-line
   */
  RBEIMConstructionBase (RBEIMConstructionBase &&) = default;
  RBEIMConstructionBase (const RBEIMConstructionBase &) = delete;
  RBEIMConstructionBase & operator= (const RBEIMConstructionBase &) = delete;
  RBEIMConstructionBase & operator= (RBEIMConstructionBase &&) = delete;
  virtual ~RBEIMConstructionBase ();

  /**
   * Clear this object.
   */
  virtual void clear() override;

  /**
   * Set the RBEIMEvaluation object.
   */
  void set_rb_eim_evaluation(RBEIMEvaluationBase & rb_eim_eval_in);

  /**
   * Get a reference to the RBEvaluation object.
   */
  RBEIMEvaluationBase & get_rb_eim_evaluation();

  /**
   * Get a const reference to the RBEvaluation object.
   */
  const RBEIMEvaluationBase & get_rb_eim_evaluation() const;

  /**
   * Perform initialization of this object to prepare for running
   * train_eim_approximation().
   */
  void initialize_eim_construction();

  /**
   * Read parameters in from file and set up this system
   * accordingly.
   */
  virtual void process_parameters_file (const std::string & parameters_filename);

  /**
   * Set the state of this RBConstruction object based on the arguments
   * to this function.
   */
  void set_rb_construction_parameters(unsigned int n_training_samples_in,
                                      bool deterministic_training_in,
                                      unsigned int training_parameters_random_seed_in,
                                      bool quiet_mode_in,
                                      unsigned int Nmax_in,
                                      Real rel_training_tolerance_in,
                                      Real abs_training_tolerance_in,
                                      RBParameters mu_min_in,
                                      RBParameters mu_max_in,
                                      std::map<std::string, std::vector<Real>> discrete_parameter_values_in,
                                      std::map<std::string,bool> log_scaling,
                                      std::map<std::string, std::vector<Number>> * training_sample_list=nullptr);

  /**
   * Specify which type of "best fit" we use to guide the EIM
   * greedy algorithm.
   */
  void set_best_fit_type_flag (const std::string & best_fit_type_string);

  /**
   * Print out info that describes the current setup of this RBConstruction.
   */
  virtual void print_info();

  /**
   * Generate the EIM approximation for the specified parametrized function.
   * Return the final tolerance from the training algorithm.
   */
  Real train_eim_approximation();

  /**
   * Build a vector of ElemAssembly objects that accesses the basis
   * functions stored in this RBEIMConstructionBase object. This is useful
   * for performing the Offline stage of the Reduced Basis method where
   * we want to use assembly functions based on this EIM approximation.
   */
  virtual void initialize_eim_assembly_objects();

  /**
   * \returns The vector of assembly objects that point to this RBEIMConstructionBase.
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
  virtual void init_context(FEMContext &);

  /**
   * Get/set the relative tolerance for the basis training.
   */
  void set_rel_training_tolerance(Real new_training_tolerance);
  Real get_rel_training_tolerance();

  /**
   * Get/set the absolute tolerance for the basis training.
   */
  void set_abs_training_tolerance(Real new_training_tolerance);
  Real get_abs_training_tolerance();

  /**
   * Get/set Nmax, the maximum number of RB
   * functions we are willing to compute.
   */
  unsigned int get_Nmax() const;
  virtual void set_Nmax(unsigned int Nmax);

  /**
   * Get the EIM solution vector at all parametrized functions in the training
   * set. In some cases we want to store this data for future use. For example
   * this is useful in the case that the parametrized function is defined
   * based on a look-up table rather than an analytical function, since
   * if we store the EIM solution data, we can do Online solves without
   * initializing the look-up table data.
   */
  void store_eim_solutions_for_training_set();

  /**
   * Enum that indicates which type of "best fit" algorithm
   * we should use.
   * a) projection: Find the best fit in the inner product
   * b) eim: Use empirical interpolation to find a "best fit"
   */
  BEST_FIT_TYPE best_fit_type_flag;

protected:

  /**
   * The RBEIMEvaluation object that we use to perform the EIM training.
   */
  RBEIMEvaluationBase * _rb_eim_eval;

  /**
   * Maximum number of EIM basis functions we are willing to use.
   */
  unsigned int _Nmax;

  /**
   * Relative and absolute tolerances for training the EIM approximation.
   */
  Real _rel_training_tolerance;
  Real _abs_training_tolerance;

  /**
   * The matrix we use in order to perform L2 projections of
   * parametrized functions as part of EIM training.
   */
  DenseMatrix<Number> _eim_projection_matrix;

  /**
   * The vector of assembly objects that are created to point to
   * this RBEIMConstructionBase.
   */
  std::vector<std::unique_ptr<ElemAssembly>> _rb_eim_assembly_objects;

  /**
   * Keep track of a scaling factor for each component of the parametrized functions in
   * the training set which "scales up" each component to have a similar magnitude as
   * the largest component encountered in the training set. This can give more uniform
   * scaling across all components and is helpful in cases where components have widely
   * varying magnitudes.
   */
  std::vector<Real> _component_scaling_in_training_set;

};

} // namespace libMesh

#endif // LIBMESH_RB_EIM_CONSTRUCTION_BASE_H
