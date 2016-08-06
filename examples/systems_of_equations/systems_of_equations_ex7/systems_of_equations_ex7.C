// The libMesh Finite Element Library.
// Copyright (C) 2002-2016 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


// <h1> Systems Example 7 - Large deformation elasticity using Hencky Model </h1>
// \author Lorenzo Zanon
// \author David Knezevic
//
// In this example, we consider an elastic cantilever beam using the Hencky model,
// which is appropriate for the large strain case. We follow the formulation from
// Computational Methods for Plasticity, by Neta, Peric, Owen. (We refer to this book
// as NPO below.) The implementation here uses NonlinearImplicitSystem to assmble
// and solve the nonlinear system.
//
// We use an Updated Lagrangian approach, in which all data refers to
// the current configuration and we move the mesh at every iteration. (The alternative, not
// considered here, is the Total Lagrangian formulation in which all computations are
// performed on the reference geometry. These two approaches are equivalent, but have various
// pros and cons in terms of implementation.)
//
// With the Updated Lagrangian approach, the nonlinear residual has the form:
//  G(u,v) = \int_\phi(\Omega) f_i v_i dx + \int_\phi(\Gamma) g_i v_i ds
//           - \int_\phi(\Omega) \sigma_ij v_i,j dx,
// where:
//  * \phi is the mapping from reference to current geometry
//  * \sigma is the Cauchy stress tensor
//  * f is a body load.
//  * g is a surface traction on the surface \Gamma.
// In this example we only consider a body load (e.g. gravity), hence we set g = 0.
//
// We solve the PDE using Newton's method, hence we must linearize the formulation. The directional
// derivate of G at u (which yields the Jacobian matrix) is given by:
//  DG(u,v)[\deltau] = -\int_\phi(\Omega) a_ijkl v_i,j \deltau_k,l dx
// where:
//  a_ijkl = (1/J) \partial \tau_ij / \partial B_mn BB_mnkl - \sigma_il \delta_jk
// and:
//  * \tau is the Kirchoff stress
//  * B is the left Cauchy-Green tensor, B = F F^T
//  * BB_mnkl = delta_mk B_nl + delta_nk B_ml
//  * F is the deformation gradient
//  * J = det(F)
//  * \delta is the Kronecker delta
// Note that \phi, \tau, B, and F all depend on the current displacement, u, and hence this
// problem is nonlinear.
//
// In the case of the Hencky model, we define the strain as follows:
//  hencky_strain_ij = 0.5 ln(B_ij),
// and the Hencky strain energy is given by:
//  \psi(hencky_strain) = 0.5 hencky_strain_ij D_ijkl hencky_strain_kl,
// where D_ijkl is the fourth order linear elasticity tensor (e.g. see systems_of_equations_ex6).
//
// Note that the Hencky strain function, 0.5 ln(B_ij), is an isotropic tensor-valued
// function, defined by:
//  ln(B) = \sum_i=1^p ln(\lambda_i) E_i
// where p is the number of distinct eigenvalues, \lambda_i is the i^th eigenvalue of B_ij,
// and E_i is the i^th "eigenprojection" operator (i.e. the projection onto the eigenspace).
// We follow the approach in Appendix A.5 of NPO in order to evaluate ln(B_ij) and its derivative
// \partial(ln(B_ij)) / \partial B_kl (we require the derivative for a_ijkl, as discussed below).
//
// Also, for the Hencky model, we have:
//  * \tau_ij = \partial \psi / \partial hencky_strain_ij = D_ijkl hencky_strain_kl.
//  * \sigma_ij = J tau_ij = J D_ijkl hencky_strain_kl.
//
// Putting the above pieces together, we can now obtain the complete expression for a_ijkl.
// First we differentiate \tau wrt B_ij, which gives:
//  \tau_ij / \partial B_kl = D_ijmn L_mnkl
// where:
//  L_ijkl = \partial(hencky_strain_ij) / \partial B_kl.
// Plugging these values into the formula above for a_ijkl gives:
//  a_ijkl = (1/J) D_ijrs L_rsmn BB_mnkl - \sigma_il \delta_jk.
//
// Based on the information above, we can assemble the residual and Jacobian for this
// model and hence solve the nonlinear system using the Newton's method solver provided
// by NonlinearImplicitSystem.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <cmath>

// Various include files needed for the mesh & solver functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/getpot.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/zero_function.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_compute_data.h"

// The nonlinear solver and system we will be using
#include "libmesh/nonlinear_solver.h"
#include "libmesh/nonlinear_implicit_system.h"

#define BOUNDARY_ID_MIN_Z 0
#define BOUNDARY_ID_MIN_Y 1
#define BOUNDARY_ID_MAX_X 2
#define BOUNDARY_ID_MAX_Y 3
#define BOUNDARY_ID_MIN_X 4
#define BOUNDARY_ID_MAX_Z 5

using namespace libMesh;

/**
  * Kronecker delta function.
  */
Real kronecker_delta(unsigned int i,
                     unsigned int j)
{
  return i == j ? 1. : 0.;
}

class Dijkl
{
public:

  /**
   * Constructor.
   */
  Dijkl(Real young_modulus, Real poisson_ratio)
  :
  _young_modulus(young_modulus),
  _poisson_ratio(poisson_ratio)
  {
  }

  /**
   * Evaluate the tensor for the specified indices.
   */
  Real evaluate(
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int l) const
  {
    const Real lambda_1 = (_young_modulus*_poisson_ratio)/((1.+_poisson_ratio)*(1.-2.*_poisson_ratio));
    const Real lambda_2 = _young_modulus/(2.*(1.+_poisson_ratio));

    Real value =
      lambda_1 * kronecker_delta(i,j) * kronecker_delta(k,l) +
      lambda_2 * (kronecker_delta(i,k) * kronecker_delta(j,l) + kronecker_delta(i,l) * kronecker_delta(j,k));

    return value;
  }

private:

  Real _young_modulus;
  Real _poisson_ratio;

};

/**
  * This class computes the Hencky strain, and the derivative of the Hencky strain,
  * based on the tensor B. Here we follow the implementation described in Box A.5 of NPO.
  */
class HenckyTensors
{
public:

  /**
   * Constructor.
   */
  HenckyTensors(const DenseMatrix<Number>& B)
  :
  _B(B)
  {
    init_hencky_strain_data();
  }

  /**
   * Evaluate the second-order Hencky strain tensor for the specified indices.
   */
  Real evaluate_hencky_strain(
    unsigned int i,
    unsigned int j) const
  {
    libmesh_assert_less(i,3);
    libmesh_assert_less(j,3);

    return _hencky_strain(i,j);
  }

  /**
   * Evaluate the fourth-order Hencky strain derivative tensor for the specified indices.
   */
  Real evaluate_hencky_strain_deriv(
    unsigned int i,
    unsigned int j,
    unsigned int k,
    unsigned int l) const
  {
    libmesh_assert_equal_to(_distinct_B_eigenvalues.size(), _distinct_B_eigenprojections.size());

    Number dBsquared_dB_ijkl = _B(i,k) * kronecker_delta(k,l) +
                               _B(l,j) * kronecker_delta(i,k);
    Number I_S_ijkl = 0.5 * (kronecker_delta(i,k)*kronecker_delta(j,l) +
                             kronecker_delta(i,l)*kronecker_delta(j,k));

    // We have a separate case depending on the number of distinct eigenvalues (1, 2, or 3)
    if(_distinct_B_eigenvalues.size() == 3)
    {
      for(unsigned int a=0; a<3; a++)
      {
        unsigned int b = (a + 1) % 3;
        unsigned int c = (a + 2) % 3;

        Number x_a = _distinct_B_eigenvalues[a];
        Number x_b = _distinct_B_eigenvalues[b];
        Number x_c = _distinct_B_eigenvalues[c];

        const DenseMatrix<Number>& E_a = _distinct_B_eigenprojections[a];
        const DenseMatrix<Number>& E_b = _distinct_B_eigenprojections[b];
        const DenseMatrix<Number>& E_c = _distinct_B_eigenprojections[c];

        Number y_x_a = 0.5 * log(x_a);
        Number deriv_y_x_a = 0.5 / x_a;

        Number value =
          y_x_a / ((x_a - x_b) * (x_a - x_c)) *
            ( dBsquared_dB_ijkl
            - (x_b + x_c) * I_S_ijkl
            - ((x_a - x_b) + (x_a - x_c)) * E_a(i,j) * E_a(k,l)
            - (x_b - x_c) * E_b(i,j) * E_b(k,l)
            - E_c(i,j) * E_c(k,l) )
          + deriv_y_x_a * E_a(i,j) * E_a(k,l);
        return value;
      }
    }
    else if(_distinct_B_eigenvalues.size() == 2)
    {
      Number x_a = _distinct_B_eigenvalues[0]; // This is the distinct eigenvalue
      Number x_c = _distinct_B_eigenvalues[1]; // This is the repeated eigenvalue

      Number y_x_a = 0.5 * log(x_a);
      Number y_x_c = 0.5 * log(x_c);
      Number deriv_y_x_a = 0.5 / x_a;
      Number deriv_y_x_c = 0.5 / x_c;

      Number s_1 =
        (y_x_a - y_x_c)/std::pow(x_a - x_c, 2.) - deriv_y_x_c/(x_a - x_c);
      Number s_2 =
          2.*x_c*(y_x_a - y_x_c)/std::pow(x_a - x_c, 2.)
        - deriv_y_x_c*(x_a + x_c)/(x_a - x_c);
      Number s_3 =
          2.*(y_x_a - y_x_c)/std::pow(x_a - x_c, 3.)
        - (deriv_y_x_a + deriv_y_x_c)/std::pow(x_a - x_c, 2.);
      Number s_4 = x_c * s_3;
      Number s_5 = s_4;
      Number s_6 = x_c*x_c*s_3; 

      Number value =
          s_1 * dBsquared_dB_ijkl
        - s_2 * I_S_ijkl
        - s_3 * _B(i,j) * _B(k,l)
        + s_4 * _B(i,j) * kronecker_delta(k,l)
        + s_5 * kronecker_delta(i,j) * _B(k,l)
        - s_6 * kronecker_delta(i,j) * kronecker_delta(k,l);
      return value;
    }
    else if(_distinct_B_eigenvalues.size() == 1)
    {
      Number x_a = _distinct_B_eigenvalues[0];
      Number deriv_y_x_a = 0.5 / x_a;

      return (deriv_y_x_a * I_S_ijkl);
    }

    libmesh_error_msg("Should not reach here!");
    return 0.;
  }

private:

  /**
   * Helper to initialize the data required for evaluation of the Hencky
   * strain and its derivative.
   */
  void init_hencky_strain_data()
  {
    DenseMatrix<Number> B_squared = _B;
    B_squared.right_multiply(_B);

    Number I1 = _B(0,0) + _B(1,1) + _B(2,2);
    Number I2 = 0.5 * ( I1*I1 - B_squared(0,0) - B_squared(1,1) - B_squared(2,2) );

    // det is non-const, so make a copy of B
    DenseMatrix<Number> B_copy = _B;
    Number I3 = B_copy.det();

    Number R = (-2.*I1*I1*I1 + 9.*I1*I2 - 27.*I3) / 54.;
    Number Q = (I1*I1 - 3.*I2) / 9.;

    std::vector<Number> B_eigenvalues(3);
    Real TOL = 1.e-10;
    if( std::abs(Q) > TOL )
    {
      Number theta_argument = R / std::sqrt(Q*Q*Q);
      libmesh_assert_greater_equal( theta_argument, -1.);
      libmesh_assert_less_equal( theta_argument, 1. );

      Number theta = acos( theta_argument );

      B_eigenvalues[0] = -2. * std::sqrt(Q) * cos(theta/3.) + I1 / 3.;
      B_eigenvalues[1] = -2. * std::sqrt(Q) * cos( (theta+2.*pi)/3.) + I1 / 3.;
      B_eigenvalues[2] = -2. * std::sqrt(Q) * cos( (theta-2.*pi)/3.) + I1 / 3.;
    }
    else
    {
      B_eigenvalues[0] = I1 / 3.;
      B_eigenvalues[1] = I1 / 3.;
      B_eigenvalues[2] = I1 / 3.;
    }

    bool equal_01 =
      ( std::abs(B_eigenvalues[0] - B_eigenvalues[1])/std::abs(B_eigenvalues[0]) < TOL );
    bool equal_02 =
      ( std::abs(B_eigenvalues[0] - B_eigenvalues[2])/std::abs(B_eigenvalues[0]) < TOL );
    bool equal_12 =
      ( std::abs(B_eigenvalues[1] - B_eigenvalues[2])/std::abs(B_eigenvalues[0]) < TOL );

    DenseMatrix<Number> identity(3,3);
    identity(0,0) = 1.;
    identity(1,1) = 1.;
    identity(2,2) = 1.;

    _distinct_B_eigenvalues.clear();
    _distinct_B_eigenprojections.clear();
    if( equal_01 && equal_02 && equal_12 )
    {
      _distinct_B_eigenvalues.push_back( B_eigenvalues[0] );
      _distinct_B_eigenprojections.push_back( identity );
    }
    else if( ( equal_01 && !equal_12) ||
             (!equal_01 &&  equal_12) ||
             ( equal_02 && !equal_12) )
    {
      unsigned int not_equal_index =0;
      if(equal_12)
      {
        not_equal_index = 0;
      }
      else if(equal_02)
      {
        not_equal_index = 1;
      }
      else if(equal_01)
      {
        not_equal_index = 2;
      }
      else
      {
        libmesh_error_msg("Should not reach here!");
      }
      unsigned int equal_index = (not_equal_index + 1) % 3;

      DenseMatrix<Number> E_not_equal_index =
        get_distinct_eigenprojection(B_eigenvalues[not_equal_index], I1, I3);
      _distinct_B_eigenvalues.push_back( B_eigenvalues[not_equal_index] );
      _distinct_B_eigenprojections.push_back( E_not_equal_index );

      DenseMatrix<Number> E_equal_index = identity;
      E_equal_index.add(-1., E_not_equal_index);
      _distinct_B_eigenvalues.push_back( B_eigenvalues[equal_index] );
      _distinct_B_eigenprojections.push_back( E_equal_index );
    }
    else if(!equal_01 && !equal_02 && !equal_12)
    {
      for(unsigned int eval_index=0; eval_index<3; eval_index++)
      {
        DenseMatrix<Number> E_i = get_distinct_eigenprojection( B_eigenvalues[eval_index], I1, I3);
        _distinct_B_eigenvalues.push_back( B_eigenvalues[eval_index] );
        _distinct_B_eigenprojections.push_back( E_i );
      }
    }
    else
    {
      libmesh_error_msg("Should not reach here!");
    }

    _hencky_strain.resize(3,3);
    for(unsigned int i=0; i<_distinct_B_eigenvalues.size(); i++)
    {
      libmesh_assert_greater( _distinct_B_eigenvalues[i], 0.);

      Number log_lambda_i = 0.5 * log( _distinct_B_eigenvalues[i] );
      _hencky_strain.add(log_lambda_i, _distinct_B_eigenprojections[i]);
    }

    libmesh_assert_equal_to(_distinct_B_eigenvalues.size(), _distinct_B_eigenprojections.size());
  }

  /**
   * Helper to get the eigenprojection of a 3x3 matrix \p B corresponding
   * to the eigenvalue \p eval_i. This method assumes that eval_i is a distinct
   * eigenvalue. \p I1 and \p I3 are the first and third invariants of B.
   */
  DenseMatrix<Number> get_distinct_eigenprojection(Number eval_i,
                                                   Number I1,
                                                   Number I3)
  {
    DenseMatrix<Number> B_squared = _B;
    B_squared.right_multiply(_B);

    DenseMatrix<Number> identity(3,3);
    identity(0,0) = 1.;
    identity(1,1) = 1.;
    identity(2,2) = 1.;

    DenseMatrix<Number> E_i = B_squared;
    E_i.add( -(I1 - eval_i), _B);
    E_i.add( (I3/eval_i), identity);
    E_i.scale( eval_i / (2.*eval_i*eval_i*eval_i - I1*eval_i*eval_i + I3) );

    return E_i;
  }

  // The second-order tensor that we compute the strain for, via 0.5 * ln(B)
  const DenseMatrix<Number> _B;

  // The second-order Hencky strain tensor
  DenseMatrix<Number> _hencky_strain;

  // The distinct eigenvalues and the projection operators
  // onto the corresponding eigenspaces.
  std::vector<Number> _distinct_B_eigenvalues;
  std::vector< DenseMatrix<Number> > _distinct_B_eigenprojections;

};

class FiniteStrainElasticity : public NonlinearImplicitSystem::ComputeResidual,
                               public NonlinearImplicitSystem::ComputeJacobian
{
private:
  EquationSystems & es;

public:

  FiniteStrainElasticity (EquationSystems & es_in) :
    es(es_in)
  {}


  /**
   * Evaluate the Jacobian of the nonlinear system.
   */
  virtual void jacobian (const NumericVector<Number> & soln,
                         SparseMatrix<Number> & jacobian,
                         NonlinearImplicitSystem & /*sys*/)
  {
    // First we move the mesh, since we use the Updated Lagrangian approach.
    move_mesh(soln, /*scaling_factor*/ 1.);

    const Real young_modulus = es.parameters.get<Real>("young_modulus");
    const Real poisson_ratio = es.parameters.get<Real>("poisson_ratio");

    const MeshBase & mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("FiniteStrainElasticity");

    const unsigned int u_var = system.variable_number ("u");

    const DofMap & dof_map = system.get_dof_map();

    FEType fe_type = dof_map.variable_type(u_var);
    UniquePtr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, fe_type.default_quadrature_order());
    fe->attach_quadrature_rule (&qrule);

    UniquePtr<FEBase> fe_face (FEBase::build(dim, fe_type));
    QGauss qface (dim-1, fe_type.default_quadrature_order());
    fe_face->attach_quadrature_rule (&qface);

    const std::vector<Real> & JxW = fe->get_JxW();
    const std::vector<std::vector<Real> > & phi = fe->get_phi();
    const std::vector<std::vector<RealGradient> > & dphi = fe->get_dphi();

    DenseMatrix<Number> Ke;
    DenseSubMatrix<Number> Ke_var[3][3] =
      {
        {DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke)},
        {DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke)},
        {DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke)}
      };

    std::vector<dof_id_type> dof_indices;
    std::vector<std::vector<dof_id_type> > dof_indices_var(3);

    jacobian.zero();

    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    for ( ; el != end_el; ++el)
      {
        const Elem * elem = *el;
        dof_map.dof_indices (elem, dof_indices);
        for (unsigned int var=0; var<3; var++)
          dof_map.dof_indices (elem, dof_indices_var[var], var);

        const unsigned int n_dofs = dof_indices.size();
        const unsigned int n_var_dofs = dof_indices_var[0].size();

        fe->reinit (elem);

        Ke.resize (n_dofs, n_dofs);
        for (unsigned int var_i=0; var_i<3; var_i++)
          for (unsigned int var_j=0; var_j<3; var_j++)
            Ke_var[var_i][var_j].reposition (var_i*n_var_dofs, var_j*n_var_dofs, n_var_dofs, n_var_dofs);

        for (unsigned int qp=0; qp<qrule.n_points(); qp++)
          {
            // The quantities required below are:
            //  * The deformation gradient F
            //  * J = det(F)
            //  * B = F F^T
            //  * hencky_strain_ij = 0.5 ln(B_ij)
            //  * \sigma_ij = J D_ijkl hencky_strain_kl

            DenseVector<Number> u_vec(3);
            DenseMatrix<Number> grad_u(3, 3);
            for (unsigned int var_i=0; var_i<3; var_i++)
              {
                for (unsigned int j=0; j<n_var_dofs; j++)
                  u_vec(var_i) += phi[j][qp]*soln(dof_indices_var[var_i][j]);

                // Row is variable u, v, or w column is x, y, or z
                for (unsigned int var_j=0; var_j<3; var_j++)
                  for (unsigned int j=0; j<n_var_dofs; j++)
                    grad_u(var_i,var_j) += dphi[j][qp](var_j)*soln(dof_indices_var[var_i][j]);
              }

            DenseMatrix<Number> F(3, 3);
            F = grad_u;
            for (unsigned int var=0; var<3; var++)
              F(var, var) += 1.;

            Real J = F.det();

            DenseMatrix<Number> B = F;
            B.right_multiply_transpose(F);

            HenckyTensors hencky_strain_data(B);

            Dijkl D_ijkl_tensor(young_modulus, poisson_ratio);

            DenseMatrix<Number> sigma(3, 3);
            for (unsigned int i=0; i<3; i++)
              for (unsigned int j=0; j<3; j++)
                for (unsigned int k=0; k<3; k++)
                  for (unsigned int l=0; l<3; l++)
                    sigma(i,j) +=
                      J * D_ijkl_tensor.evaluate(i, j, k, l) *
                      hencky_strain_data.evaluate_hencky_strain(k,l);

            // Now we assemble the Jacobian, which is given by:
            // DG(u,v)[\deltau] = -\int_\phi(\Omega) a_ijkl v_i,j \deltau_k,l dx
            // where:
            //  * a_ijkl = (1/J) D_ijrs L_rsmn BB_mnkl - \sigma_il \delta_jk
            //  * L_ijkl is the Hencky strain derivative
            //  * BB_mnkl = delta_mk B_nl + delta_nk B_ml

            for (unsigned int i=0; i<3; i++)
              for (unsigned int j=0; j<3; j++)
                for (unsigned int k=0; k<3; k++)
                  for (unsigned int l=0; l<3; l++)
                  {
                    Number a_ijkl = 0.;
                    for (unsigned int m=0; m<3; m++)
                      for (unsigned int n=0; n<3; n++)
                      {
                        Number term_ijmn = 0;
                        for (unsigned int r=0; r<3; r++)
                          for (unsigned int s=0; s<3; s++)
                          {
                            term_ijmn +=
                              D_ijkl_tensor.evaluate(i,j,r,s) *
                              hencky_strain_data.evaluate_hencky_strain_deriv(r,s,m,n);
                          }

                        Number BB_mnkl = kronecker_delta(m,k)*B(n,l) + kronecker_delta(n,k)*B(m,l);
                        a_ijkl += (1./J) * term_ijmn * BB_mnkl;
                      }
                    a_ijkl -= sigma(i,l) * kronecker_delta(j,k);

                    for (unsigned int dof_i=0; dof_i<n_var_dofs; dof_i++)
                      for (unsigned int dof_k=0; dof_k<n_var_dofs; dof_k++)
                        {
                          Ke_var[i][k](dof_i,dof_k) -=
                            JxW[qp] * a_ijkl * dphi[dof_k][qp](l) * dphi[dof_i][qp](j);
                        }
                  }
          }

        dof_map.constrain_element_matrix (Ke, dof_indices);
        jacobian.add_matrix (Ke, dof_indices);
      }

    // Finally undo the move.
    move_mesh(soln, /*scaling_factor*/ -1.);
  }

  /**
   * Evaluate the residual of the nonlinear system.
   */
  virtual void residual (const NumericVector<Number> & soln,
                         NumericVector<Number> & residual,
                         NonlinearImplicitSystem & /*sys*/)
  {
    // First we move the mesh, since we use the Updated Lagrangian approach.
    move_mesh(soln, /*scaling_factor*/ 1.);

    const Real young_modulus = es.parameters.get<Real>("young_modulus");
    const Real poisson_ratio = es.parameters.get<Real>("poisson_ratio");
    const Real forcing_magnitude = es.parameters.get<Real>("forcing_magnitude");

    const MeshBase & mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("FiniteStrainElasticity");

    const unsigned int u_var = system.variable_number ("u");

    const DofMap & dof_map = system.get_dof_map();

    FEType fe_type = dof_map.variable_type(u_var);
    UniquePtr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, fe_type.default_quadrature_order());
    fe->attach_quadrature_rule (&qrule);

    UniquePtr<FEBase> fe_face (FEBase::build(dim, fe_type));
    QGauss qface (dim-1, fe_type.default_quadrature_order());
    fe_face->attach_quadrature_rule (&qface);

    const std::vector<Real> & JxW = fe->get_JxW();
    const std::vector<std::vector<Real> > & phi = fe->get_phi();
    const std::vector<std::vector<RealGradient> > & dphi = fe->get_dphi();

    DenseVector<Number> Re;

    DenseSubVector<Number> Re_var[3] =
      {DenseSubVector<Number>(Re),
       DenseSubVector<Number>(Re),
       DenseSubVector<Number>(Re)};

    std::vector<dof_id_type> dof_indices;
    std::vector< std::vector<dof_id_type> > dof_indices_var(3);

    residual.zero();

    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    for ( ; el != end_el; ++el)
      {
        const Elem * elem = *el;
        dof_map.dof_indices (elem, dof_indices);
        for (unsigned int var=0; var<3; var++)
          dof_map.dof_indices (elem, dof_indices_var[var], var);

        const unsigned int n_dofs = dof_indices.size();
        const unsigned int n_var_dofs = dof_indices_var[0].size();

        fe->reinit (elem);

        Re.resize (n_dofs);
        for (unsigned int var=0; var<3; var++)
          Re_var[var].reposition (var*n_var_dofs, n_var_dofs);

        for (unsigned int qp=0; qp<qrule.n_points(); qp++)
          {
            // The quantities required below are:
            //  * The deformation gradient F
            //  * J = det(F)
            //  * B = F F^T
            //  * hencky_strain_ij = 0.5 ln(B_ij)
            //  * \sigma_ij = J D_ijkl hencky_strain_kl

            DenseVector<Number> u_vec(3);
            DenseMatrix<Number> grad_u(3, 3);
            for (unsigned int var_i=0; var_i<3; var_i++)
              {
                for (unsigned int j=0; j<n_var_dofs; j++)
                  u_vec(var_i) += phi[j][qp]*soln(dof_indices_var[var_i][j]);

                // Row is variable u, v, or w column is x, y, or z
                for (unsigned int var_j=0; var_j<3; var_j++)
                  for (unsigned int j=0; j<n_var_dofs; j++)
                    grad_u(var_i,var_j) += dphi[j][qp](var_j)*soln(dof_indices_var[var_i][j]);
              }

            DenseMatrix<Number> F(3, 3);
            F = grad_u;
            for (unsigned int var=0; var<3; var++)
              F(var, var) += 1.;

            Real J = F.det();

            DenseMatrix<Number> B = F;
            B.right_multiply_transpose(F);

            HenckyTensors hencky_strain_data(B);

            Dijkl D_ijkl_tensor(young_modulus, poisson_ratio);

            DenseMatrix<Number> sigma(3, 3);
            for (unsigned int i=0; i<3; i++)
              for (unsigned int j=0; j<3; j++)
                for (unsigned int k=0; k<3; k++)
                  for (unsigned int l=0; l<3; l++)
                    sigma(i,j) +=
                      J * D_ijkl_tensor.evaluate(i, j, k, l) *
                      hencky_strain_data.evaluate_hencky_strain(k,l);

            // Now we assemble the residual, which is given by:
            //  G(u,v) = \int_\phi(\Omega) f_i v_i dx - \int_\phi(\Omega) \sigma_ij v_i,j dx,

            DenseVector<Number> f_vec(3);
            f_vec(0) = 0.;
            f_vec(1) = 0.;
            f_vec(2) = -forcing_magnitude;

            for (unsigned int dof_i=0; dof_i<n_var_dofs; dof_i++)
              for (unsigned int i=0; i<3; i++)
                {
                  for (unsigned int j=0; j<3; j++)
                    {
                      Re_var[i](dof_i) += JxW[qp] * (-sigma(i,j) * dphi[dof_i][qp](j));
                    }

                  Re_var[i](dof_i) += JxW[qp] * (f_vec(i) * phi[dof_i][qp]);
                }
          }

        dof_map.constrain_element_vector (Re, dof_indices);
        residual.add_vector (Re, dof_indices);
      }

    // Finally undo the move.
    move_mesh(soln, /*scaling_factor*/ -1.);
  }

  /**
   * Move the mesh nodes based on the current solution.
   * We scale \p soln by \p scaling_factor.
   */
  void move_mesh (const NumericVector<Number> & soln,
                  Real scaling_factor)
  {
    MeshBase & mesh = es.get_mesh();

    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("FiniteStrainElasticity");

    // Maintain a set of node ids that we've encountered.
    std::set<dof_id_type> encountered_node_ids;

    // Localize soln so that we have the data to move all
    // elements (not just elements local to this processor).
    UniquePtr< NumericVector<Number> > localized_solution =
      NumericVector<Number>::build(es.comm());

    localized_solution->init (system.solution->size(), false, SERIAL);
    soln.localize(*localized_solution);
    localized_solution->scale(scaling_factor);

    MeshBase::const_element_iterator       el     = mesh.active_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_elements_end();

    for ( ; el != end_el; ++el)
      {
        Elem * elem = *el;
        Elem * orig_elem = mesh.elem_ptr(elem->id());

        for (unsigned int node_id=0; node_id<elem->n_nodes(); node_id++)
          {
            Node & node = elem->node_ref(node_id);

            if (encountered_node_ids.find(node.id()) != encountered_node_ids.end())
              continue;

            encountered_node_ids.insert(node.id());

            std::vector<std::string> uvw_names(3);
            uvw_names[0] = "u";
            uvw_names[1] = "v";
            uvw_names[2] = "w";

            {
              const Point master_point = elem->master_point(node_id);

              Point uvw;
              for (unsigned int index=0; index<uvw_names.size(); index++)
                {
                  const unsigned int var = system.variable_number(uvw_names[index]);
                  const FEType & fe_type = system.get_dof_map().variable_type(var);

                  FEComputeData data (es, master_point);

                  FEInterface::compute_data(elem->dim(),
                                            fe_type,
                                            elem,
                                            data);

                  std::vector<dof_id_type> dof_indices_var;
                  system.get_dof_map().dof_indices (orig_elem, dof_indices_var, var);

                  for (unsigned int i=0; i<dof_indices_var.size(); i++)
                    {
                      Number value = (*localized_solution)(dof_indices_var[i]) * data.shape[i];
                      uvw(index) += libmesh_real(value);
                    }
                }

              // Update the node's location
              node += uvw;
            }
          }
      }
  }

  /**
   * Compute the Cauchy stress for the current solution.
   */
  void compute_stresses()
  {
    const Real young_modulus = es.parameters.get<Real>("young_modulus");
    const Real poisson_ratio = es.parameters.get<Real>("poisson_ratio");

    const MeshBase & mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("FiniteStrainElasticity");

    unsigned int displacement_vars[3];
    displacement_vars[0] = system.variable_number ("u");
    displacement_vars[1] = system.variable_number ("v");
    displacement_vars[2] = system.variable_number ("w");
    const unsigned int u_var = system.variable_number ("u");

    const DofMap & dof_map = system.get_dof_map();
    FEType fe_type = dof_map.variable_type(u_var);
    UniquePtr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, fe_type.default_quadrature_order());
    fe->attach_quadrature_rule (&qrule);

    const std::vector<Real> & JxW = fe->get_JxW();
    const std::vector<std::vector<RealGradient> > & dphi = fe->get_dphi();

    // Also, get a reference to the ExplicitSystem
    ExplicitSystem & stress_system = es.get_system<ExplicitSystem>("StressSystem");
    const DofMap & stress_dof_map = stress_system.get_dof_map();
    unsigned int sigma_vars[6];
    sigma_vars[0] = stress_system.variable_number ("sigma_00");
    sigma_vars[1] = stress_system.variable_number ("sigma_01");
    sigma_vars[2] = stress_system.variable_number ("sigma_02");
    sigma_vars[3] = stress_system.variable_number ("sigma_11");
    sigma_vars[4] = stress_system.variable_number ("sigma_12");
    sigma_vars[5] = stress_system.variable_number ("sigma_22");

    // Storage for the stress dof indices on each element
    std::vector< std::vector<dof_id_type> > dof_indices_var(system.n_vars());
    std::vector<dof_id_type> stress_dof_indices_var;

    // To store the stress tensor on each element
    DenseMatrix<Number> elem_avg_stress_tensor(3, 3);

    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    for ( ; el != end_el; ++el)
      {
        const Elem * elem = *el;

        for (unsigned int var=0; var<3; var++)
          dof_map.dof_indices (elem, dof_indices_var[var], displacement_vars[var]);

        const unsigned int n_var_dofs = dof_indices_var[0].size();

        fe->reinit (elem);

        // clear the stress tensor
        elem_avg_stress_tensor.resize(3, 3);

        for (unsigned int qp=0; qp<qrule.n_points(); qp++)
          {
            DenseMatrix<Number> grad_u(3, 3);
            // Row is variable u1, u2, or u3, column is x, y, or z
            for (unsigned int var_i=0; var_i<3; var_i++)
              for (unsigned int var_j=0; var_j<3; var_j++)
                for (unsigned int j=0; j<n_var_dofs; j++)
                  grad_u(var_i,var_j) += dphi[j][qp](var_j) * system.current_solution(dof_indices_var[var_i][j]);

            DenseMatrix<Number> strain_tensor(3, 3);
            for (unsigned int i=0; i<3; i++)
              for (unsigned int j=0; j<3; j++)
                {
                  strain_tensor(i,j) += 0.5 * (grad_u(i,j) + grad_u(j,i));

                  for (unsigned int k=0; k<3; k++)
                    strain_tensor(i,j) += 0.5 * grad_u(k,i)*grad_u(k,j);
                }

            // Define the deformation gradient
            DenseMatrix<Number> F(3, 3);
            F = grad_u;
            for (unsigned int var=0; var<3; var++)
              F(var, var) += 1.;

            Dijkl D_ijkl_tensor(young_modulus, poisson_ratio);

            DenseMatrix<Number> stress_tensor(3, 3);
            for (unsigned int i=0; i<3; i++)
              for (unsigned int j=0; j<3; j++)
                for (unsigned int k=0; k<3; k++)
                  for (unsigned int l=0; l<3; l++)
                    stress_tensor(i,j) +=
                      D_ijkl_tensor.evaluate(i, j, k, l) * strain_tensor(k, l);

            // stress_tensor now holds the second Piola-Kirchoff stress (PK2) at point qp.
            // However, in this example we want to compute the Cauchy stress which is given by
            // 1/det(F) * F * PK2 * F^T, hence we now apply this transformation.
            stress_tensor.scale(1./F.det());
            stress_tensor.left_multiply(F);
            stress_tensor.right_multiply_transpose(F);

            // We want to plot the average Cauchy stress on each element, hence
            // we integrate stress_tensor
            elem_avg_stress_tensor.add(JxW[qp], stress_tensor);
          }

        // Get the average stress per element by dividing by volume
        elem_avg_stress_tensor.scale(1./elem->volume());

        // load elem_sigma data into stress_system
        unsigned int stress_var_index = 0;
        for (unsigned int i=0; i<3; i++)
          for (unsigned int j=i; j<3; j++)
            {
              stress_dof_map.dof_indices (elem, stress_dof_indices_var, sigma_vars[stress_var_index]);

              // We are using CONSTANT MONOMIAL basis functions, hence we only need to get
              // one dof index per variable
              dof_id_type dof_index = stress_dof_indices_var[0];

              if ((stress_system.solution->first_local_index() <= dof_index) &&
                  (dof_index < stress_system.solution->last_local_index()))
                stress_system.solution->set(dof_index, elem_avg_stress_tensor(i,j));

              stress_var_index++;
            }
      }

    // Should call close and update when we set vector entries directly
    stress_system.solution->close();
    stress_system.update();
  }

};


int main (int argc, char ** argv)
{
  LibMeshInit init (argc, argv);

  // This example requires the PETSc nonlinear solvers
  libmesh_example_requires(libMesh::default_solver_package() == PETSC_SOLVERS, "--enable-petsc");

  // We use a 3D domain.
  libmesh_example_requires(LIBMESH_DIM > 2, "--disable-1D-only --disable-2D-only");

  GetPot infile("systems_of_equations_ex7.in");
  const Real x_length = infile("x_length", 0.);
  const Real y_length = infile("y_length", 0.);
  const Real z_length = infile("z_length", 0.);
  const Real n_elem_x = infile("n_elem_x", 0);
  const Real n_elem_y = infile("n_elem_y", 0);
  const Real n_elem_z = infile("n_elem_z", 0);
  const std::string approx_order = infile("approx_order", "FIRST");
  const std::string fe_family = infile("fe_family", "LAGRANGE");

  const Real young_modulus = infile("Young_modulus", 1.0);
  const Real poisson_ratio = infile("poisson_ratio", 0.3);
  const Real forcing_magnitude = infile("forcing_magnitude", 0.001);

  const Real nonlinear_abs_tol = infile("nonlinear_abs_tol", 1.e-8);
  const Real nonlinear_rel_tol = infile("nonlinear_rel_tol", 1.e-8);
  const unsigned int nonlinear_max_its = infile("nonlinear_max_its", 50);

  const unsigned int n_solves = infile("n_solves", 10);
  const Real force_scaling = infile("force_scaling", 5.0);

  Mesh mesh(init.comm());

  MeshTools::Generation::build_cube(mesh,
                                    n_elem_x,
                                    n_elem_y,
                                    n_elem_z,
                                    0., x_length,
                                    0., y_length,
                                    0., z_length,
                                    HEX27);

  mesh.print_info();

  EquationSystems equation_systems (mesh);
  FiniteStrainElasticity fse(equation_systems);

  NonlinearImplicitSystem & system =
    equation_systems.add_system<NonlinearImplicitSystem> ("FiniteStrainElasticity");

  unsigned int u_var =
    system.add_variable("u",
                        Utility::string_to_enum<Order>   (approx_order),
                        Utility::string_to_enum<FEFamily>(fe_family));

  unsigned int v_var =
    system.add_variable("v",
                        Utility::string_to_enum<Order>   (approx_order),
                        Utility::string_to_enum<FEFamily>(fe_family));

  unsigned int w_var =
    system.add_variable("w",
                        Utility::string_to_enum<Order>   (approx_order),
                        Utility::string_to_enum<FEFamily>(fe_family));

  // Also, initialize an ExplicitSystem to store stresses
  ExplicitSystem & stress_system =
    equation_systems.add_system<ExplicitSystem> ("StressSystem");
  stress_system.add_variable("sigma_00", CONSTANT, MONOMIAL);
  stress_system.add_variable("sigma_01", CONSTANT, MONOMIAL);
  stress_system.add_variable("sigma_02", CONSTANT, MONOMIAL);
  stress_system.add_variable("sigma_11", CONSTANT, MONOMIAL);
  stress_system.add_variable("sigma_12", CONSTANT, MONOMIAL);
  stress_system.add_variable("sigma_22", CONSTANT, MONOMIAL);

  equation_systems.parameters.set<Real>         ("nonlinear solver absolute residual tolerance") = nonlinear_abs_tol;
  equation_systems.parameters.set<Real>         ("nonlinear solver relative residual tolerance") = nonlinear_rel_tol;
  equation_systems.parameters.set<unsigned int> ("nonlinear solver maximum iterations")          = nonlinear_max_its;

  system.nonlinear_solver->residual_object = &fse;
  system.nonlinear_solver->jacobian_object = &fse;

  equation_systems.parameters.set<Real>("young_modulus") = young_modulus;
  equation_systems.parameters.set<Real>("poisson_ratio") = poisson_ratio;
  equation_systems.parameters.set<Real>("forcing_magnitude") = forcing_magnitude;

  // Attach Dirichlet boundary conditions
  std::set<boundary_id_type> clamped_boundaries;
  clamped_boundaries.insert(BOUNDARY_ID_MIN_X);

  std::vector<unsigned int> uvw;
  uvw.push_back(u_var);
  uvw.push_back(v_var);
  uvw.push_back(w_var);

  ZeroFunction<Number> zero;

  system.get_dof_map().add_dirichlet_boundary(DirichletBoundary (clamped_boundaries, uvw, &zero));

  equation_systems.init();
  equation_systems.print_info();

  // Provide a loop here so that we can do a sequence of solves
  // where solve n gives a good starting guess for solve n+1.
  // This "continuation" approach is helpful for solving for
  // large values of "forcing_magnitude".
  // Set n_solves and force_scaling in nonlinear_elasticity.in.
  for (unsigned int count=0; count<n_solves; count++)
    {
      Real previous_forcing_magnitude = equation_systems.parameters.get<Real>("forcing_magnitude");
      equation_systems.parameters.set<Real>("forcing_magnitude") = previous_forcing_magnitude*force_scaling;

      libMesh::out << "Performing solve "
                   << count
                   << ", forcing_magnitude: "
                   << equation_systems.parameters.get<Real>("forcing_magnitude")
                   << std::endl;

      system.solve();

      libMesh::out << "System solved at nonlinear iteration "
                   << system.n_nonlinear_iterations()
                   << " , final nonlinear residual norm: "
                   << system.final_nonlinear_residual()
                   << std::endl
                   << std::endl;

      libMesh::out << "Computing stresses..." << std::endl;

      fse.compute_stresses();

#ifdef LIBMESH_HAVE_EXODUS_API
      std::stringstream filename;
      filename << "solution_" << count << ".exo";
      ExodusII_IO (mesh).write_equation_systems(filename.str(), equation_systems);
#endif
    }

  return 0;
}
