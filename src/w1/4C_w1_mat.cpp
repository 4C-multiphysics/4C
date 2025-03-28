// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_w1.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Constitutive matrix C and stresses (private)                mgit 05/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_call_matgeononl(
    const Core::LinAlg::SerialDenseVector& strain,        ///< Green-Lagrange strain vector
    Core::LinAlg::SerialDenseMatrix& stress,              ///< stress vector
    Core::LinAlg::SerialDenseMatrix& C,                   ///< elasticity matrix
    const int numeps,                                     ///< number of strains
    std::shared_ptr<const Core::Mat::Material> material,  ///< the material data
    Teuchos::ParameterList& params,                       ///< element parameter list
    const int gp                                          ///< Gauss point
)
{
  if (material->material_type() == Core::Materials::m_structporo or
      material->material_type() == Core::Materials::m_structpororeaction or
      material->material_type() == Core::Materials::m_structpororeactionECM)
  {
    std::shared_ptr<const Mat::StructPoro> actmat =
        std::static_pointer_cast<const Mat::StructPoro>(material);
    // setup is done in so3_poro
    // actmat->setup(NUMGPT_SOH8);
    material = actmat->get_material();
  }

  /*--------------------------- call material law -> get tangent modulus--*/
  switch (material->material_type())
  {
    case Core::Materials::m_stvenant: /*----------------------- linear elastic ---*/
    {
      const Mat::StVenantKirchhoff* actmat =
          static_cast<const Mat::StVenantKirchhoff*>(material.get());
      double ym = actmat->youngs();
      double pv = actmat->poisson_ratio();


      /*-------------- some comments, so that even fluid people are able to
         understand this quickly :-)
         the "strain" vector looks like:

             | EPS_xx |
             | EPS_yy |
             | EPS_xy |
             | EPS_yx | */

      switch (wtype_)
      {
          /*---------------------------------material-tangente-- plane stress ---*/
        case plane_stress:
        {
          const double e1 = ym / (1. - pv * pv);
          const double e2 = pv * e1;
          const double e3 = e1 * (1. - pv) / 2.;

          C(0, 0) = e1;
          C(0, 1) = e2;
          C(0, 2) = 0.;
          C(0, 3) = 0.;

          C(1, 0) = e2;
          C(1, 1) = e1;
          C(1, 2) = 0.;
          C(1, 3) = 0.;

          C(2, 0) = 0.;
          C(2, 1) = 0.;
          C(2, 2) = e3;
          C(2, 3) = e3;

          C(3, 0) = 0.;
          C(3, 1) = 0.;
          C(3, 2) = e3;
          C(3, 3) = e3;

          break;
        }

          /*----------- material-tangente - plane strain, rotational symmetry ---*/
        case plane_strain:
        {
          const double c1 = ym / (1.0 + pv);
          const double b1 = c1 * pv / (1.0 - 2.0 * pv);
          const double a1 = b1 + c1;

          C(0, 0) = a1;
          C(0, 1) = b1;
          C(0, 2) = 0.;
          C(0, 3) = 0.;

          C(1, 0) = b1;
          C(1, 1) = a1;
          C(1, 2) = 0.;
          C(1, 3) = 0.;

          C(2, 0) = 0.;
          C(2, 1) = 0.;
          C(2, 2) = c1 / 2.;
          C(2, 3) = c1 / 2.;

          C(3, 0) = 0.;
          C(3, 1) = 0.;
          C(3, 2) = c1 / 2;
          C(3, 3) = c1 / 2;

          break;
        }

        default:
        {
          FOUR_C_THROW("Only plane strain and plane stress options exist for wtype_");
          break;
        }
      }  // switch(wtype_)

      /*-------------------------- evaluate 2.PK-stresses -------------------*/
      /*------------------ Summenschleife -> += (2.PK stored as vecor) ------*/

      Core::LinAlg::SerialDenseVector svector;
      svector.size(3);

      for (int k = 0; k < 3; k++)
      {
        for (int i = 0; i < numeps; i++)
        {
          svector(k) += C(k, i) * strain(i);
        }
      }
      /*------------------ 2.PK stored as matrix -----------------------------*/
      stress(0, 0) = svector(0);
      stress(0, 2) = svector(2);
      stress(1, 1) = svector(1);
      stress(1, 3) = svector(2);
      stress(2, 0) = svector(2);
      stress(2, 2) = svector(1);
      stress(3, 1) = svector(2);
      stress(3, 3) = svector(0);

      break;
    }

    case Core::Materials::m_elasthyper:  // general hyperelastic material (bborn, 06/09)
                                         // case Core::Materials::m_stvenant:  //
                                         // st.venant-kirchhoff-material
    {
      material_response3d_plane(stress, C, strain, params, gp);
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid type of material law for wall element");
      break;
    }
  }  // switch(material->material_type())

  return;
}  // Discret::Elements::Wall1::w1_call_matgeononl

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Discret::Elements::Wall1::material_response3d_plane(Core::LinAlg::SerialDenseMatrix& stress,
    Core::LinAlg::SerialDenseMatrix& C, const Core::LinAlg::SerialDenseVector& strain,
    Teuchos::ParameterList& params, const int gp)
{
  // make 3d equivalent of Green-Lagrange strain
  Core::LinAlg::Matrix<6, 1> gl(Core::LinAlg::Initialization::uninitialized);
  green_lagrange_plane3d(strain, gl);

  // call 3d stress response
  Core::LinAlg::Matrix<6, 1> pk2(Core::LinAlg::Initialization::zero);   // must be zerofied!!!
  Core::LinAlg::Matrix<6, 6> cmat(Core::LinAlg::Initialization::zero);  // must be zerofied!!!
  material_response3d(&pk2, &cmat, &gl, params, gp);

  // dimension reduction type
  if (wtype_ == plane_strain)
  {
    ;  // go on
  }
  else if (wtype_ == plane_stress)
  {
    // strain vector above bears final values on: E_{11},E_{22},E_{12}
    // strain vector above bears initial guesses on: E_{33},E_{23},E_{31}

    // initial plane stress error
    double pserr = std::sqrt(pk2(2) * pk2(2) + pk2(4) * pk2(4) + pk2(5) * pk2(5));

    // make Newton-Raphson iteration to identify
    // E_{33},E_{23},E_{31} which satisfy S_{33}=S_{23}=S_{31}=0
    int i = 0;
    const double tol = 1e-6;
    const int n = 10;
    // working arrays
    Core::LinAlg::Matrix<3, 3> crr(
        Core::LinAlg::Initialization::zero);  // LHS // constitutive matrix of restraint compo
                                              // this matrix needs to be zeroed out for further
                                              // usage in case the following while loop is
                                              // entirely skipped during runtime
    Core::LinAlg::Matrix<3, 1> rr(
        Core::LinAlg::Initialization::uninitialized);  // RHS // stress residual of restraint
                                                       // compo
    Core::LinAlg::Matrix<3, 1> ir(
        Core::LinAlg::Initialization::uninitialized);  // SOL  // restraint strain components
    // the Newton-Raphson loop
    while ((pserr > tol) and (i < n))
    {
      // build sub-system a.b=c to solve
      crr(0, 0) = cmat(2, 2);
      crr(0, 1) = cmat(2, 4);
      crr(0, 2) = cmat(2, 5);
      crr(1, 0) = cmat(4, 2);
      crr(1, 1) = cmat(4, 4);
      crr(1, 2) = cmat(4, 5);
      crr(2, 0) = cmat(5, 2);
      crr(2, 1) = cmat(5, 4);
      crr(2, 2) = cmat(5, 5);
      rr(0) = -pk2(2);
      rr(1) = -pk2(4);
      rr(2) = -pk2(5);
      // solution
      // an in-place inversion is used, 'coz the inverse is needed below
      crr.invert();
      ir.multiply(crr, rr);
      // update
      gl(2) += ir(0);
      gl(4) += 2.0 * ir(1);  // NOT SURE ABOUT 2.0, LACKED TESTING MATERIAL
      gl(5) += 2.0 * ir(2);  // NOT SURE ABOUT 2.0, LACKED TESTING MATERIAL

      // call for new 3d stress response
      pk2.clear();   // must be blanked!!
      cmat.clear();  // must be blanked!!
      material_response3d(&pk2, &cmat, &gl, params, gp);

      // current plane stress error
      pserr = std::sqrt(pk2(2) * pk2(2) + pk2(4) * pk2(4) + pk2(5) * pk2(5));

      // increment loop index
      i += 1;
    }

    // check if convergence was reached
    if ((i >= n) and (pserr > tol))
    {
      FOUR_C_THROW("Failed to identify plane stress solution");
    }
    else
    {
      // static condensation
      // The restraint strains E_{33},E_{23},E_{31} have been made
      // dependent on free strains E_{11},E_{22},E_{12}
      // --- with an implicit function.
      // Thus the effect of the linearisation with respect to the
      // dependent strains must be added onto the free strains.
      Core::LinAlg::Matrix<3, 3> cfr(Core::LinAlg::Initialization::uninitialized);
      cfr(0, 0) = cmat(0, 2);
      cfr(0, 1) = cmat(0, 4);
      cfr(0, 2) = cmat(0, 5);
      cfr(1, 0) = cmat(1, 2);
      cfr(1, 1) = cmat(1, 4);
      cfr(1, 2) = cmat(1, 5);
      cfr(2, 0) = cmat(3, 2);
      cfr(2, 1) = cmat(3, 4);
      cfr(2, 2) = cmat(3, 5);
      Core::LinAlg::Matrix<3, 3> crrrf(Core::LinAlg::Initialization::uninitialized);
      crrrf.multiply_nt(crr, cfr);
      Core::LinAlg::Matrix<3, 3> cfrrrrf(Core::LinAlg::Initialization::uninitialized);
      cfrrrrf.multiply_nn(cfr, crrrf);
      // update constitutive matrix of free components
      cmat(0, 0) -= cfrrrrf(0, 0);
      cmat(0, 1) -= cfrrrrf(0, 1);
      cmat(0, 3) -= cfrrrrf(0, 2);
      cmat(1, 0) -= cfrrrrf(1, 0);
      cmat(1, 1) -= cfrrrrf(1, 1);
      cmat(1, 3) -= cfrrrrf(1, 2);
      cmat(3, 0) -= cfrrrrf(2, 0);
      cmat(3, 1) -= cfrrrrf(2, 1);
      cmat(3, 3) -= cfrrrrf(2, 2);
    }
  }
  else
  {
    FOUR_C_THROW("Dimension reduction type wtype_={} is not available.", wtype_);
  }

  // transform 2nd Piola--Kirchhoff stress back to 2d stress matrix
  stress.putScalar(0.0);                                               // zerofy
  stress(0, 0) = stress(3, 3) = pk2(0);                                // S_{11}
  stress(1, 1) = stress(2, 2) = pk2(1);                                // S_{22}
  stress(0, 2) = stress(1, 3) = stress(3, 1) = stress(2, 0) = pk2(3);  // S_{12}

  // transform elasticity matrix  back to 2d matrix
  C(0, 0) = cmat(0, 0);  // C_{1111}
  C(0, 1) = cmat(0, 1);  // C_{1122}
  C(0, 2) = cmat(0, 3);  // C_{1112}
  C(0, 3) = cmat(0, 3);  // C_{1112} = C_{1121}

  C(1, 0) = cmat(1, 0);  // C_{2211}
  C(1, 1) = cmat(1, 1);  // C_{2222}
  C(1, 2) = cmat(1, 3);  // C_{2212}
  C(1, 3) = cmat(1, 3);  // C_{2221} = C_{2212}

  C(2, 0) = cmat(3, 0);  // C_{1211}
  C(2, 1) = cmat(3, 1);  // C_{1222}
  C(2, 2) = cmat(3, 3);  // C_{1212}
  C(2, 3) = cmat(3, 3);  // C_{1221} = C_{1212}

  C(3, 0) = cmat(3, 0);  // C_{2111} = C_{1211}
  C(3, 1) = cmat(3, 1);  // C_{2122} = C_{1222}
  C(3, 2) = cmat(3, 3);  // C_{2112} = C_{1212}
  C(3, 3) = cmat(3, 3);  // C_{2121} = C_{1212}

  // leave this dump
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Discret::Elements::Wall1::material_response3d(Core::LinAlg::Matrix<6, 1>* stress,
    Core::LinAlg::Matrix<6, 6>* cmat, const Core::LinAlg::Matrix<6, 1>* glstrain,
    Teuchos::ParameterList& params, const int gp)
{
  solid_material()->evaluate(nullptr, glstrain, params, stress, cmat, gp, id());

  return;
}

/*-----------------------------------------------------------------------------*
| deliver internal/strain energy                                    bborn 08/08|
*-----------------------------------------------------------------------------*/
double Discret::Elements::Wall1::energy_internal(
    std::shared_ptr<const Core::Mat::Material> material, Teuchos::ParameterList& params,
    const Core::LinAlg::SerialDenseVector& Ev, const int gp)
{
  // switch material type
  switch (material->material_type())
  {
    case Core::Materials::m_stvenant:  // linear elastic
    {
      Core::LinAlg::SerialDenseMatrix Cm(Wall1::numnstr_, Wall1::numnstr_);  // elasticity matrix
      Core::LinAlg::SerialDenseMatrix Sm(Wall1::numnstr_, Wall1::numnstr_);  // 2nd PK stress matrix
      w1_call_matgeononl(Ev, Sm, Cm, Wall1::numnstr_, material, params, gp);
      Core::LinAlg::SerialDenseVector Sv(Wall1::numnstr_);  // 2nd PK stress vector
      Sv(0) = Sm(0, 0);
      Sv(1) = Sm(1, 1);
      Sv(2) = Sv(3) = Sm(0, 2);
      return 0.5 * Sv.dot(Ev);
    }
    break;
    case Core::Materials::m_elasthyper:
    {
      // transform the 2d Green-Lagrange strains into 3d notation
      Core::LinAlg::Matrix<6, 1> glstrain(Core::LinAlg::Initialization::zero);
      green_lagrange_plane3d(Ev, glstrain);

      // strain energy
      double psi = 0.0;

      // call material for evaluation of strain energy function
      solid_material()->strain_energy(glstrain, psi, gp, id());

      return psi;
    }
    break;
    case Core::Materials::m_structporo:
    case Core::Materials::m_structpororeaction:
    case Core::Materials::m_structpororeactionECM:
    {
      // transform the 2d Green-Lagrange strains into 3d notation
      Core::LinAlg::Matrix<6, 1> glstrain(Core::LinAlg::Initialization::zero);
      green_lagrange_plane3d(Ev, glstrain);

      // strain energy
      double psi = 0.0;

      // call material for evaluation of strain energy function
      solid_material()->strain_energy(glstrain, psi, gp, id());

      return psi;
    }
    break;
    default:
    {
      FOUR_C_THROW("Illegal type of material for this element");
      return 0.0;
    }
    break;
  }  // end of switch (material->material_type())

  FOUR_C_THROW(
      "You should never end up here, since all possible cases should be "
      "covered by the material selection.");

  return 0.0;
}

/*-----------------------------------------------------------------------------*
| deliver kinetic energy                                            bborn 08/08|
*-----------------------------------------------------------------------------*/
double Discret::Elements::Wall1::energy_kinetic(
    const Core::LinAlg::SerialDenseMatrix& mass, const std::vector<double>& vel)
{
  double kin = 0.0;
  for (int i = 0; i < 2 * num_node(); ++i)
    for (int j = 0; j < 2 * num_node(); ++j) kin += 0.5 * vel[i] * mass(i, j) * vel[j];
  return kin;
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
