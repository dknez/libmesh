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

#ifndef LIBMESH_SIDE_RB_EIM_EVALUATION_H
#define LIBMESH_SIDE_RB_EIM_EVALUATION_H

// libMesh includes
#include "libmesh/rb_eim_evaluation_base.h"

namespace libMesh
{

class SideRBParametrizedFunction;

/**
 * Evaluation class for EIM on element sides.
 */
class SideRBEIMEvaluation : public RBEIMEvaluationBase
{
public:

  /**
   * Constructor.
   */
  SideRBEIMEvaluation(const Parallel::Communicator & comm);

  /**
   * Special functions.
   * - This class contains unique_ptrs, so it can't be default copy
       constructed/assigned.
   * - The destructor is defaulted out of line.
   */
  SideRBEIMEvaluation (SideRBEIMEvaluation &&) = default;
  SideRBEIMEvaluation (const SideRBEIMEvaluation &) = delete;
  SideRBEIMEvaluation & operator= (const SideRBEIMEvaluation &) = delete;
  SideRBEIMEvaluation & operator= (SideRBEIMEvaluation &&) = default;
  virtual ~SideRBEIMEvaluation ();

  /**
   * Type of the data structure used to map from (elem id, side index) -> [n_vars][n_qp] data.
   */
  typedef std::map<std::pair<dof_id_type,unsigned int>, std::vector<std::vector<Number>>> SideQpDataMap;

  /**
   * Set the parametrized function that we will approximate
   * using the Empirical Interpolation Method. This object
   * will take ownership of the unique pointer.
   */
  void set_parametrized_function(std::unique_ptr<SideRBParametrizedFunction> pf);

  /**
   * Get a const reference to the parametrized function.
   */
  SideRBParametrizedFunction & get_parametrized_function();

  /**
   * Return the current number of EIM basis functions.
   */
  virtual unsigned int get_n_basis_functions() const override;

  /**
   * Set the number of basis functions. Useful when reading in
   * stored data.
   */
  virtual void set_n_basis_functions(unsigned int n_bfs) override;

  /**
   * Subtract coeffs[i]*basis_function[i] from \p v.
   */
  void decrement_vector(SideQpDataMap & v,
                        const DenseVector<Number> & coeffs);

  /**
   * Fill up values by evaluating the parametrized function \p pf for all quadrature
   * points on element \p elem_id and component \p comp.
   */
  static void get_parametrized_function_values_at_qps(
    const SideQpDataMap & pf,
    dof_id_type elem_id,
    unsigned int side_index,
    unsigned int comp,
    std::vector<Number> & values);

  /**
   * Same as above, except that we just return the value at the qp^th
   * quadrature point.
   */
  static Number get_parametrized_function_value(
    const Parallel::Communicator & comm,
    const SideQpDataMap & pf,
    dof_id_type elem_id,
    unsigned int side_index,
    unsigned int comp,
    unsigned int qp);

  /**
   * Fill up \p values with the basis function values for basis function
   * \p basis_function_index and variable \p var, at all quadrature points
   * on the specified element/side. Each processor stores data for only the
   * elements local to that processor, so if elem_id is not on this processor
   * then \p values will be empty.
   */
  void get_eim_basis_function_values_at_qps(unsigned int basis_function_index,
                                            dof_id_type elem_id,
                                            unsigned int side_index,
                                            unsigned int var,
                                            std::vector<Number> & values) const;

  /**
   * Same as above, except that we just return the value at the qp^th
   * quadrature point.
   */
  Number get_eim_basis_function_value(unsigned int basis_function_index,
                                      dof_id_type elem_id,
                                      unsigned int side_index,
                                      unsigned int comp,
                                      unsigned int qp) const;

  /**
   * Get a reference to the i^th basis function.
   */
  const SideQpDataMap & get_basis_function(unsigned int i) const;

  /**
   * Add \p bf to our EIM basis.
   */
  void add_basis_function_and_interpolation_data(
    const SideQpDataMap & bf,
    Point p,
    unsigned int comp,
    dof_id_type elem_id,
    unsigned int side_index,
    boundary_id_type boundary_id,
    unsigned int qp,
    const std::vector<Point> & perturbs,
    const std::vector<Real> & phi_i_qp);

  /**
   * Write out all the basis functions to file.
   * \p sys is used for file IO
   * \p directory_name specifies which directory to write files to
   * \p read_binary_basis_functions indicates whether to write
   * binary or ASCII data
   *
   * Note: this is not currently a virtual function and is not related
   * to the RBEvaluation function of the same name.
   */
  void write_out_basis_functions(const std::string & directory_name = "offline_data",
                                 bool write_binary_basis_functions = true);

  /**
   * Read in all the basis functions from file.
   *
   * \param sys The Mesh in this System determines the parallel distribution of the basis functions.
   * \param directory_name Specifies which directory to write files to.
   * \param read_binary_basis_functions Indicates whether to expect binary or ASCII data.
   *
   * Note: this is not a virtual function and is not related to the
   * RBEvaluation function of the same name.
   */
  void read_in_basis_functions(const System & sys,
                               const std::string & directory_name = "offline_data",
                               bool read_binary_basis_functions = true);

protected:

  /**
   * Evaluate the parametrized function at \p mus, and store the results
   * in \p output_all_comps.
   */
  virtual void parametrized_function_vectorized_evaluate(const std::vector<RBParameters> & mus,
                                                         std::vector<std::vector<std::vector<Number>>> & output_all_comps) override;

  /**
   * Store the parametrized function that will be approximated
   * by this EIM system. Note that the parametrized function
   * may have more than one component, and each component is
   * approximated by a separate variable in the EIM system.
   */
  std::unique_ptr<SideRBParametrizedFunction> _parametrized_function;

  /**
   * The EIM basis functions. We store values at quadrature points
   * on elements that are local to this processor. The indexing
   * is as follows:
   *   basis function index --> element ID --> variable --> quadrature point --> value
   * We use a map to index the element ID, since the IDs on this processor in
   * general will not start at zero.
   */
  std::vector<SideQpDataMap> _local_eim_basis_functions;

  /**
   * Print the contents of _local_eim_basis_functions to libMesh::out.
   * Helper function mainly useful for debugging.
   */
  void print_local_eim_basis_functions() const;

  /**
   * Helper function that gathers the contents of
   * _local_eim_basis_functions to processor 0 in preparation for
   * printing to file.
   */
  void gather_bfs();

  /**
   * Helper function that distributes the entries of
   * _local_eim_basis_functions to their respective processors after
   * they are read in on processor 0.
   */
  void distribute_bfs(const System & sys);

};

}

#endif // LIBMESH_SIDE_RB_EIM_EVALUATION_H
