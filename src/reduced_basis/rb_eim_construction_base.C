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
#include <fstream>
#include <sstream>

// LibMesh includes
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/equation_systems.h"
#include "libmesh/parallel.h"
#include "libmesh/parallel_algebra.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature.h"
#include "libmesh/utility.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_compute_data.h"
#include "libmesh/getpot.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/fem_context.h"
#include "libmesh/elem.h"
#include "libmesh/int_range.h"
#include "libmesh/auto_ptr.h"

// rbOOmit includes
#include "libmesh/rb_eim_construction_base.h"
#include "libmesh/rb_eim_evaluation.h"
#include "libmesh/rb_parametrized_function_base.h"

// C++ include
#include <limits>

namespace libMesh
{

RBEIMConstructionBase::RBEIMConstructionBase (EquationSystems & es,
                                              const std::string & name_in,
                                              const unsigned int number_in)
  : RBConstructionBase(es, name_in, number_in),
    best_fit_type_flag(PROJECTION_BEST_FIT),
    _Nmax(0),
    _rel_training_tolerance(1.e-4),
    _abs_training_tolerance(1.e-12),
    _max_abs_value_in_training_set(0.),
    _max_abs_value_in_training_set_index(0)
{
  // The training set should be the same on all processors in the
  // case of EIM training.
  serial_training_set = true;
}

RBEIMConstructionBase::~RBEIMConstructionBase () = default;

void RBEIMConstructionBase::clear()
{
  RBConstructionBase::clear();

  _rb_eim_assembly_objects.clear();
  _eim_projection_matrix.resize(0,0);
}

void RBEIMConstructionBase::initialize_eim_assembly_objects(const RBEIMEvaluationBase & rb_eim_evaluation)
{
  _rb_eim_assembly_objects.clear();
  for (auto i : make_range(rb_eim_evaluation.get_n_basis_functions()))
    _rb_eim_assembly_objects.push_back(build_eim_assembly(i));
}

std::vector<std::unique_ptr<ElemAssembly>> & RBEIMConstructionBase::get_eim_assembly_objects()
{
  return _rb_eim_assembly_objects;
}

void RBEIMConstructionBase::init_context(FEMContext & c)
{
  // Pre-request FE data for all element dimensions present in the
  // mesh.  Note: we currently pre-request FE data for all variables
  // in the current system but in some cases that may be overkill, for
  // example if only variable 0 is used.
  const System & sys = c.get_system();
  const MeshBase & mesh = sys.get_mesh();

  for (unsigned int dim=1; dim<=3; ++dim)
    if (mesh.elem_dimensions().count(dim))
      for (auto var : make_range(sys.n_vars()))
      {
        auto fe = c.get_element_fe(var, dim);
        fe->get_JxW();
        fe->get_xyz();

        auto side_fe = c.get_side_fe(var, dim);
        side_fe->get_JxW();
        side_fe->get_xyz();
      }
}

void RBEIMConstructionBase::set_best_fit_type_flag (const std::string & best_fit_type_string)
{
  if (best_fit_type_string == "projection")
    {
      best_fit_type_flag = PROJECTION_BEST_FIT;
    }
  else
    if (best_fit_type_string == "eim")
      {
        best_fit_type_flag = EIM_BEST_FIT;
      }
    else
      libmesh_error_msg("Error: invalid best_fit_type in input file");
}

void RBEIMConstructionBase::print_info()
{
  // Print out info that describes the current setup
  libMesh::out << "system name: " << this->name() << std::endl;
  libMesh::out << "Nmax: " << get_Nmax() << std::endl;
  libMesh::out << "Greedy relative error tolerance: " << get_rel_training_tolerance() << std::endl;
  libMesh::out << "Greedy absolute error tolerance: " << get_abs_training_tolerance() << std::endl;
  libMesh::out << "Number of parameters: " << get_n_params() << std::endl;
  for (const auto & pr : get_parameters())
    if (!is_discrete_parameter(pr.first))
      {
        libMesh::out <<   "Parameter " << pr.first
                     << ": Min = " << get_parameter_min(pr.first)
                     << ", Max = " << get_parameter_max(pr.first) << std::endl;
      }

  print_discrete_parameter_values();
  libMesh::out << "n_training_samples: " << get_n_training_samples() << std::endl;
  libMesh::out << "quiet mode? " << is_quiet() << std::endl;

  if (best_fit_type_flag == PROJECTION_BEST_FIT)
    {
      libMesh::out << "EIM best fit type: projection" << std::endl;
    }
  else
    if (best_fit_type_flag == EIM_BEST_FIT)
      {
        libMesh::out << "EIM best fit type: eim" << std::endl;
      }
  libMesh::out << std::endl;
}


Real RBEIMConstructionBase::get_max_abs_value_in_training_set() const
{
  return _max_abs_value_in_training_set;
}

void RBEIMConstructionBase::initialize_eim_construction()
{
  initialize_parametrized_functions_in_training_set();
}

void RBEIMConstructionBase::process_parameters_file (const RBParametrizedFunctionBase & parametrized_function,
                                                     const std::string & parameters_filename)
{
  // First read in data from input_filename
  GetPot infile(parameters_filename);

  std::string best_fit_type_string = infile("best_fit_type","projection");
  set_best_fit_type_flag(best_fit_type_string);

  const unsigned int n_training_samples = infile("n_training_samples",0);
  const bool deterministic_training = infile("deterministic_training",false);
  unsigned int training_parameters_random_seed_in =
    static_cast<unsigned int>(-1);
  training_parameters_random_seed_in = infile("training_parameters_random_seed",
                                              training_parameters_random_seed_in);
  const bool quiet_mode_in = infile("quiet_mode", quiet_mode);
  const unsigned int Nmax_in = infile("Nmax", _Nmax);
  const Real rel_training_tolerance_in = infile("rel_training_tolerance",
                                                _rel_training_tolerance);
  const Real abs_training_tolerance_in = infile("abs_training_tolerance",
                                                _abs_training_tolerance);

  // Read in the parameters from the input file too
  unsigned int n_continuous_parameters = infile.vector_variable_size("parameter_names");
  RBParameters mu_min_in;
  RBParameters mu_max_in;
  for (unsigned int i=0; i<n_continuous_parameters; i++)
    {
      // Read in the parameter names
      std::string param_name = infile("parameter_names", "NONE", i);

      {
        Real min_val = infile(param_name, 0., 0);
        mu_min_in.set_value(param_name, min_val);
      }

      {
        Real max_val = infile(param_name, 0., 1);
        mu_max_in.set_value(param_name, max_val);
      }
    }

  std::map<std::string, std::vector<Real>> discrete_parameter_values_in;

  unsigned int n_discrete_parameters = infile.vector_variable_size("discrete_parameter_names");
  for (unsigned int i=0; i<n_discrete_parameters; i++)
    {
      std::string param_name = infile("discrete_parameter_names", "NONE", i);

      unsigned int n_vals_for_param = infile.vector_variable_size(param_name);
      std::vector<Real> vals_for_param(n_vals_for_param);
      for (auto j : make_range(vals_for_param.size()))
        vals_for_param[j] = infile(param_name, 0., j);

      discrete_parameter_values_in[param_name] = vals_for_param;
    }

  std::map<std::string,bool> log_scaling_in;
  // For now, just set all entries to false.
  // TODO: Implement a decent way to specify log-scaling true/false
  // in the input text file
  for (const auto & pr : mu_min_in)
    log_scaling_in[pr.first] = false;

  // Set the parameters that have been read in
  set_rb_construction_parameters(parametrized_function,
                                 n_training_samples,
                                 deterministic_training,
                                 training_parameters_random_seed_in,
                                 quiet_mode_in,
                                 Nmax_in,
                                 rel_training_tolerance_in,
                                 abs_training_tolerance_in,
                                 mu_min_in,
                                 mu_max_in,
                                 discrete_parameter_values_in,
                                 log_scaling_in);
}

void RBEIMConstructionBase::set_rb_construction_parameters(const RBParametrizedFunctionBase & parametrized_function,
                                                           unsigned int n_training_samples_in,
                                                           bool deterministic_training_in,
                                                           unsigned int training_parameters_random_seed_in,
                                                           bool quiet_mode_in,
                                                           unsigned int Nmax_in,
                                                           Real rel_training_tolerance_in,
                                                           Real abs_training_tolerance_in,
                                                           RBParameters mu_min_in,
                                                           RBParameters mu_max_in,
                                                           std::map<std::string, std::vector<Real>> discrete_parameter_values_in,
                                                           std::map<std::string,bool> log_scaling_in,
                                                           std::map<std::string, std::vector<Number>> * training_sample_list)
{
  // Read in training_parameters_random_seed value.  This is used to
  // seed the RNG when picking the training parameters.  By default the
  // value is -1, which means use std::time to seed the RNG.
  set_training_random_seed(training_parameters_random_seed_in);

  // Set quiet mode
  set_quiet_mode(quiet_mode_in);

  // Initialize RB parameters
  set_Nmax(Nmax_in);

  set_rel_training_tolerance(rel_training_tolerance_in);
  set_abs_training_tolerance(abs_training_tolerance_in);

  if (parametrized_function.is_lookup_table)
    {
      const std::string & lookup_table_param_name =
        parametrized_function.lookup_table_param_name;

      libmesh_error_msg_if(!discrete_parameter_values_in.count(lookup_table_param_name),
        "Lookup table parameter should be discrete");

      std::vector<Real> & lookup_table_param_values =
        libmesh_map_find(discrete_parameter_values_in, lookup_table_param_name);

      // Overwrite the discrete values for lookup_table_param to make sure that
      // it is: 0, 1, 2, ..., size-1.
      std::iota(lookup_table_param_values.begin(), lookup_table_param_values.end(), 0);

      // Also, overwrite n_training_samples_in to make sure it matches
      // lookup_table_size so that we will get full coverage of the
      // lookup table in our training set.
      n_training_samples_in = lookup_table_param_values.size();
    }

  // Initialize the parameter ranges and the parameters themselves
  initialize_parameters(mu_min_in, mu_max_in, discrete_parameter_values_in);

  initialize_training_parameters(this->get_parameters_min(),
                                 this->get_parameters_max(),
                                 n_training_samples_in,
                                 log_scaling_in,
                                 deterministic_training_in);

  if (training_sample_list)
    {
      // Note that we must call initialize_training_parameters() before
      // load_training_set() in order to initialize the parameter vectors.
      load_training_set(*training_sample_list);
    }


  if (parametrized_function.is_lookup_table)
    {
      // Also, now that we've initialized the training set, overwrite the training
      // samples to ensure that we have full coverage of the lookup tbale.
      const std::string & lookup_table_param_name =
        parametrized_function.lookup_table_param_name;

      std::vector<Number> lookup_table_training_samples(n_training_samples_in);
      std::iota(lookup_table_training_samples.begin(), lookup_table_training_samples.end(), 0);

      set_training_parameter_values(lookup_table_param_name, lookup_table_training_samples);
    }
}

Real RBEIMConstructionBase::train_eim_approximation(RBEIMEvaluationBase & rbe,
                                                    const RBParametrizedFunctionBase & parametrized_function)
{
  LOG_SCOPE("train_eim_approximation()", "RBConstruction");

  _eim_projection_matrix.resize(get_Nmax(),get_Nmax());

  rbe.initialize_parameters(*this);
  rbe.resize_data_structures(get_Nmax());

  // If we are continuing from a previous training run,
  // we might already be at the max number of basis functions.
  // If so, we can just return.
  libmesh_error_msg_if(rbe.get_n_basis_functions() > 0,
                       "Error: We currently only support EIM training starting from an empty basis");

  libMesh::out << std::endl << "---- Performing Greedy EIM basis enrichment ----" << std::endl;
  Real greedy_error = 0.;
  std::vector<RBParameters> greedy_param_list;

  // Initialize the current training index to the index that corresponds
  // to the largest (in terms of infinity norm) function in the training set.
  // We do this to ensure that the first EIM basis function is not zero.
  unsigned int current_training_index = _max_abs_value_in_training_set_index;
  set_params_from_training_set(current_training_index);
  while (true)
    {
      libMesh::out << "Greedily selected parameter vector:" << std::endl;
      print_parameters();
      greedy_param_list.emplace_back(get_parameters());

      libMesh::out << "Enriching the EIM approximation" << std::endl;
      enrich_eim_approximation(current_training_index);
      update_eim_matrices();

      libMesh::out << std::endl << "---- Basis dimension: "
                   << rbe.get_n_basis_functions() << " ----" << std::endl;

      if (parametrized_function.is_lookup_table &&
          best_fit_type_flag == EIM_BEST_FIT)
        {
          // If this is a lookup table and we're using "EIM best fit" then we
          // need to update the eim_solutions after each EIM enrichment so that
          // we can call rb_eim_eval.rb_eim_solve() from within compute_max_eim_error().
          store_eim_solutions_for_training_set();
        }

      libMesh::out << "Computing EIM error on training set" << std::endl;
      std::pair<Real,unsigned int> max_error_pair = compute_max_eim_error();
      greedy_error = max_error_pair.first;
      current_training_index = max_error_pair.second;
      set_params_from_training_set(current_training_index);

      libMesh::out << "Maximum EIM error is " << greedy_error << std::endl << std::endl;

      // Convergence and/or termination tests
      {
        if (rbe.get_n_basis_functions() >= this->get_Nmax())
          {
            libMesh::out << "Maximum number of basis functions reached: Nmax = "
                          << get_Nmax() << std::endl;
            break;
          }

        // We consider the relative tolerance as relative to the maximum value in the training
        // set, since we assume that this maximum value provides a relevant scaling.
        if (greedy_error < (get_rel_training_tolerance() * get_max_abs_value_in_training_set()))
          {
            libMesh::out << "Relative error tolerance reached." << std::endl;
            break;
          }

        if (greedy_error < get_abs_training_tolerance())
          {
            libMesh::out << "Absolute error tolerance reached." << std::endl;
            break;
          }

        if (rbe.get_n_basis_functions() >= this->get_Nmax())
          {
            libMesh::out << "Maximum number of basis functions reached: Nmax = "
                         << get_Nmax() << std::endl;
            break;
          }

        {
          bool do_exit = false;
          for (auto & param : greedy_param_list)
            if (param == get_parameters())
              {
                libMesh::out << "Exiting greedy because the same parameters were selected twice"
                             << std::endl;
                do_exit = true;
                break;
              }

          if (do_exit)
            break; // out of while
        }
      }
    } // end while(true)

  if (parametrized_function.is_lookup_table &&
      best_fit_type_flag != EIM_BEST_FIT)
    {
      // We only enter here if best_fit_type_flag != EIM_BEST_FIT because we
      // already called this above in the EIM_BEST_FIT case.
      store_eim_solutions_for_training_set();
    }

  return greedy_error;
}

void RBEIMConstructionBase::set_rel_training_tolerance(Real new_training_tolerance)
{
  _rel_training_tolerance = new_training_tolerance;
}

Real RBEIMConstructionBase::get_rel_training_tolerance()
{
  return _rel_training_tolerance;
}

void RBEIMConstructionBase::set_abs_training_tolerance(Real new_training_tolerance)
{
  _abs_training_tolerance = new_training_tolerance;
}

Real RBEIMConstructionBase::get_abs_training_tolerance()
{
  return _abs_training_tolerance;
}

unsigned int RBEIMConstructionBase::get_Nmax() const
{
  return _Nmax;
}

void RBEIMConstructionBase::set_Nmax(unsigned int Nmax)
{
  _Nmax = Nmax;
}

} // namespace libMesh
