// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elements_jacobian.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_w1.hpp"

#include <Teuchos_BLAS.hpp>
#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                            mwgee 12/06|
 *----------------------------------------------------------------------*/
int Discret::Elements::Wall1::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // Check whether the solid material post_setup() routine has already been called and call it if
  // not
  ensure_material_post_setup(params);

  set_params_interface_ptr(params);
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())
  {
    act = params_interface().get_action_type();
  }
  else
  {
    // get the action required
    std::string action = params.get<std::string>("action", "calc_none");
    if (action == "calc_none")
      FOUR_C_THROW("No action supplied");
    else if (action == "calc_struct_linstiff")
      act = Core::Elements::struct_calc_linstiff;
    else if (action == "calc_struct_nlnstiff")
      act = Core::Elements::struct_calc_nlnstiff;
    else if (action == "calc_struct_internalforce")
      act = Core::Elements::struct_calc_internalforce;
    else if (action == "calc_struct_linstiffmass")
      act = Core::Elements::struct_calc_linstiffmass;
    else if (action == "calc_struct_nlnstiffmass")
      act = Core::Elements::struct_calc_nlnstiffmass;
    else if (action == "calc_struct_nlnstifflmass")
      act = Core::Elements::struct_calc_nlnstifflmass;
    else if (action == "calc_struct_stress")
      act = Core::Elements::struct_calc_stress;
    else if (action == "calc_struct_eleload")
      act = Core::Elements::struct_calc_eleload;
    else if (action == "calc_struct_fsiload")
      act = Core::Elements::struct_calc_fsiload;
    else if (action == "calc_struct_update_istep")
      act = Core::Elements::struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      act = Core::Elements::struct_calc_reset_istep;
    else if (action == "calc_struct_energy")
      act = Core::Elements::struct_calc_energy;
    else if (action == "calc_struct_mass_volume")
      act = Core::Elements::struct_calc_mass_volume;
    else
      FOUR_C_THROW("Unknown type of action %s for Wall1", action.c_str());
  }
  // get the material law
  std::shared_ptr<const Core::Mat::Material> actmat = material();

  // --------------------------------------------------
  // Now do the nurbs specific stuff
  std::vector<Core::LinAlg::SerialDenseVector> myknots(2);

  if (shape() == Core::FE::CellType::nurbs4 or shape() == Core::FE::CellType::nurbs9)
  {
    switch (act)
    {
      case Core::Elements::struct_calc_linstiff:
      case Core::Elements::struct_calc_nlnstiffmass:
      case Core::Elements::struct_calc_nlnstifflmass:
      case Core::Elements::struct_calc_nlnstiff:
      case Core::Elements::struct_calc_internalforce:
      case Core::Elements::struct_calc_stress:
      case Core::Elements::struct_calc_mass_volume:
      {
        auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

        bool zero_sized = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, id());

        // skip zero sized elements in knot span --- they correspond to interpolated nodes
        if (zero_sized) return (0);

        break;
      }
      default:
        myknots.clear();
        break;
    }
  }

  switch (act)
  {
    //==================================================================================
    case Core::Elements::struct_calc_linstiff:
    {
      // need current displacement and residual forces
      std::vector<double> mydisp(lm.size());
      for (int i = 0; i < (int)mydisp.size(); ++i) mydisp[i] = 0.0;
      std::vector<double> myres(lm.size());
      for (int i = 0; i < (int)myres.size(); ++i) myres[i] = 0.0;

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        w1_linstiffmass(lm, mydisp, myres, myknots, &elemat1, &elemat2, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else
      {
        w1_nlnstiffmass(lm, mydisp, myres, myknots, &elemat1, &elemat2, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        w1_linstiffmass(lm, mydisp, myres, myknots, &elemat1, &elemat2, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else
      {
        w1_nlnstiffmass(lm, mydisp, myres, myknots, &elemat1, &elemat2, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }

      if (act == Core::Elements::struct_calc_nlnstifflmass) w1_lumpmass(&elemat2);
      break;
    }
    //==================================================================================
    // nullptr-pointer for mass matrix in case of calculating only stiff matrix
    case Core::Elements::struct_calc_nlnstiff:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        w1_linstiffmass(lm, mydisp, myres, myknots, &elemat1, nullptr, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else
      {
        w1_nlnstiffmass(lm, mydisp, myres, myknots, &elemat1, nullptr, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_internalforce:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      // create a dummy element matrix (initialised to zero)
      // This matrix is not utterly useless. It is used to apply EAS-stuff in a linearised manner
      // onto the internal force vector.
      Core::LinAlg::SerialDenseMatrix myemat(lm.size(), lm.size());

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        w1_linstiffmass(lm, mydisp, myres, myknots, &myemat, nullptr, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else
      {
        w1_nlnstiffmass(lm, mydisp, myres, myknots, &myemat, nullptr, &elevec1, nullptr, nullptr,
            actmat, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none);
      }
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_recover:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (disp == nullptr || res == nullptr)
      {
        FOUR_C_THROW(
            "Cannot get state vectors \"displacement\" "
            "and/or \"residual displacement\"");
      }
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      w1_recover(lm, mydisp, myres);
      /* ToDo Probably we have to recover the history information of some special
       * materials as well.                                 hiermeier 04/2016  */
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_update_istep:
    {
      // do something with internal EAS, etc parameters
      if (iseas_)
      {
        Core::LinAlg::SerialDenseMatrix* alpha = &easdata_.alpha;    // Alpha_{n+1}
        Core::LinAlg::SerialDenseMatrix* alphao = &easdata_.alphao;  // Alpha_n
        Teuchos::BLAS<unsigned int, double> blas;
        blas.COPY((*alphao).numRows() * (*alphao).numCols(), (*alpha).values(), 1,
            (*alphao).values(),
            1);  // alphao := alpha
      }
      solid_material()->update();
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_reset_istep:
    {
      // do something with internal EAS, etc parameters
      if (iseas_)
      {
        Core::LinAlg::SerialDenseMatrix* alpha = &easdata_.alpha;    // Alpha_{n+1}
        Core::LinAlg::SerialDenseMatrix* alphao = &easdata_.alphao;  // Alpha_n
        Teuchos::BLAS<unsigned int, double> blas;
        blas.COPY((*alphao).numRows() * (*alphao).numCols(), (*alphao).values(), 1,
            (*alpha).values(),
            1);  // alpha := alphao
      }
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_stress:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      std::shared_ptr<std::vector<char>> stressdata = nullptr;
      std::shared_ptr<std::vector<char>> straindata = nullptr;
      Inpar::Solid::StressType iostress = Inpar::Solid::stress_none;
      Inpar::Solid::StrainType iostrain = Inpar::Solid::strain_none;
      if (is_params_interface())
      {
        stressdata = str_params_interface().stress_data_ptr();
        straindata = str_params_interface().strain_data_ptr();

        iostress = str_params_interface().get_stress_output_type();
        iostrain = str_params_interface().get_strain_output_type();
      }
      else
      {
        stressdata = params.get<std::shared_ptr<std::vector<char>>>("stress", nullptr);
        straindata = params.get<std::shared_ptr<std::vector<char>>>("strain", nullptr);
        iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
        iostrain = params.get<Inpar::Solid::StrainType>("iostrain", Inpar::Solid::strain_none);
      }
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      if (stressdata == nullptr) FOUR_C_THROW("Cannot get stress 'data'");
      if (straindata == nullptr) FOUR_C_THROW("Cannot get strain 'data'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
      Core::LinAlg::SerialDenseMatrix stress(intpoints.nquad, Wall1::numstr_);
      Core::LinAlg::SerialDenseMatrix strain(intpoints.nquad, Wall1::numstr_);

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        w1_linstiffmass(lm, mydisp, myres, myknots, nullptr, nullptr, nullptr, &stress, &strain,
            actmat, params, iostress, iostrain);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else
      {
        w1_nlnstiffmass(lm, mydisp, myres, myknots, nullptr, nullptr, nullptr, &stress, &strain,
            actmat, params, iostress, iostrain);
      }

      {
        Core::Communication::PackBuffer data;
        add_to_pack(data, stress);
        std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
      }

      {
        Core::Communication::PackBuffer data;
        add_to_pack(data, strain);
        std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
      }
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_energy:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // determine energies
      energy(params, lm, mydisp, &elevec1, actmat);
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_mass_volume:
    {
      // check length of elevec1
      if (elevec1.length() < 6) FOUR_C_THROW("The given result vector is too short.");

      // declaration of variables
      double volume_ref = 0.0;
      double volume_mat = 0.0;
      double volume_cur = 0.0;
      double mass_ref = 0.0;
      double mass_mat = 0.0;
      double mass_cur = 0.0;
      double density = actmat->density();

      // some definitions
      const int numnode = num_node();
      const int numdf = 2;

      Core::LinAlg::SerialDenseMatrix xjm;
      xjm.shape(2, 2);
      double det = 0.0;
      double detcur = 0.0;
      double detFmat = 0.0;  // F[0]*F[1]-F[2]*F[3];

      // shape functions, derivatives and integration rule
      Core::LinAlg::SerialDenseVector funct(numnode);
      Core::LinAlg::SerialDenseMatrix deriv;
      deriv.shape(2, numnode);
      const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

      // get displacements and extract values of this element
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // reference and current geometry (nodal positions)
      Core::LinAlg::SerialDenseMatrix xrefe(2, numnode);
      Core::LinAlg::SerialDenseMatrix xcure(2, numnode);
      Core::LinAlg::SerialDenseMatrix xmat(2, numnode);
      Core::LinAlg::SerialDenseVector strain;
      strain.size(4);
      Core::LinAlg::SerialDenseMatrix boplin;
      boplin.shape(4, 2 * numnode);
      Core::LinAlg::SerialDenseVector F;
      F.size(4);

      for (int k = 0; k < numnode; ++k)
      {
        xrefe(0, k) = nodes()[k]->x()[0];
        xrefe(1, k) = nodes()[k]->x()[1];
        xcure(0, k) = xrefe(0, k) + mydisp[k * numdf + 0];
        xcure(1, k) = xrefe(1, k) + mydisp[k * numdf + 1];
      }

      /*------------------------- get node weights for nurbs elements */
      const Core::FE::CellType distype = shape();
      Core::LinAlg::SerialDenseVector weights(numnode);
      if (distype == Core::FE::CellType::nurbs4 || distype == Core::FE::CellType::nurbs9)
      {
        for (int inode = 0; inode < numnode; ++inode)
        {
          auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);
          weights(inode) = cp->w();
        }
      }

      //----------------------------------------------------------------
      // loop over all Gauss points
      //----------------------------------------------------------------
      for (int ip = 0; ip < intpoints.nquad; ++ip)
      {
        const double e1 = intpoints.qxg[ip][0];
        const double e2 = intpoints.qxg[ip][1];
        const double wgt = intpoints.qwgt[ip];

        // get values of shape functions and derivatives in the gausspoint
        if (distype != Core::FE::CellType::nurbs4 && distype != Core::FE::CellType::nurbs9)
        {
          // shape functions and their derivatives for polynomials
          Core::FE::shape_function_2d(funct, e1, e2, distype);
          Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);
        }
        else
        {
          // nurbs version
          Core::LinAlg::SerialDenseVector gp(2);
          gp(0) = e1;
          gp(1) = e2;

          Core::FE::Nurbs::nurbs_get_2d_funct_deriv(funct, deriv, gp, myknots, weights, distype);
        }

        // REF ------------------------
        /*--------------------------------------- compute jacobian Matrix */
        w1_jacobianmatrix(xrefe, deriv, xjm, &det, numnode);

        /*------------------------------------ integration factor  -------*/
        double fac = wgt * det * thickness_;
        volume_ref += fac;
        fac = wgt * det * thickness_ * density;
        mass_ref += fac;

        // MAT ------------------------
        {
          w1_boplin(boplin, deriv, xjm, det, numnode);
          w1_defgrad(F, strain, xrefe, xcure, boplin, numnode);
          detFmat = F[0] * F[1] - F[2] * F[3];

          // CUR ------------------------
          /*--------------------------------------- compute jacobian Matrix */
          w1_jacobianmatrix(xcure, deriv, xjm, &detcur, numnode);

          /*------------------------------------ integration factor  -------*/
          fac = wgt * detcur * thickness_;
          volume_cur += fac;
          fac = wgt * detcur * thickness_ * density * 1 / detFmat;
          mass_cur += fac;
        }
      }
      //----------------------------------------------------------------

      // return results
      {
        volume_mat = volume_ref;
        mass_mat = mass_ref;
      }

      elevec1[0] = volume_ref;
      elevec1[1] = volume_mat;
      elevec1[2] = volume_cur;
      elevec1[3] = mass_ref;
      elevec1[4] = mass_mat;
      elevec1[5] = mass_cur;
      break;
    }
    //==================================================================================
    case Core::Elements::analyse_jacobian_determinant:
    {
      // get displacements and extract values of this element
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      std::vector<double> mydispmat(lm.size(), 0.0);
      // reference and current geometry (nodal positions)
      Core::LinAlg::Matrix<2, 4> xcurr;  // current  coord. of element

      for (int k = 0; k < 4; ++k)
      {
        xcurr(0, k) = nodes()[k]->x()[0] + mydisp[k * 2 + 0];
        xcurr(1, k) = nodes()[k]->x()[1] + mydisp[k * 2 + 1];
      }

      const double min_detj =
          Core::Elements::get_minimal_jac_determinant_at_nodes<Core::FE::CellType::quad4>(xcurr);

      if (min_detj < 0.0)
        error_handling(min_detj, params, __LINE__, Solid::Elements::ele_error_determinant_analysis);

      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_eleload:
    {
      FOUR_C_THROW("this method is not supposed to evaluate a load, use evaluate_neumann(...)");
      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_predict:
      break;
    //==================================================================================
    case Core::Elements::struct_create_backup:
    {
      if (iseas_) FOUR_C_THROW("EAS for the wall element is not yet considered!");

      break;
    }
    //==================================================================================
    case Core::Elements::struct_recover_from_backup:
    {
      if (iseas_) FOUR_C_THROW("EAS for the wall element is not yet considered!");

      break;
    }
    //==================================================================================
    default:
    {
      FOUR_C_THROW(
          "Unknown type of action for Wall1 element: %s", action_type_to_string(act).c_str());
      break;
    }
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public)  mgit 05/07|
 *----------------------------------------------------------------------*/

int Discret::Elements::Wall1::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);
  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto funct = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  // check total time
  double time = -1.0;
  if (is_params_interface())
    time = params_interface().get_total_time();
  else
    time = params.get("total time", -1.0);

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < noddof_)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  // no. of nodes on this surface
  const int iel = num_node();

  // do the isogeometric extras --- get knots and weights
  std::vector<Core::LinAlg::SerialDenseVector> myknots(numdim_);
  Core::LinAlg::SerialDenseVector weights(iel);

  if (shape() == Core::FE::CellType::nurbs4 || shape() == Core::FE::CellType::nurbs9)
  {
    auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

    bool zero_sized = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots, id());

    // skip zero sized elements in knot span --- they correspond to interpolated nodes
    if (zero_sized)
    {
      return (0);
    }

    for (int inode = 0; inode < iel; ++inode)
    {
      auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);

      weights(inode) = cp->w();
    }
  }

  // general arrays
  Core::LinAlg::SerialDenseMatrix xjm(numdim_, numdim_);  // iso-parametric Jacobian
  double det = 0.0;                                       // determinant of iso-parametric Jacobian

  // quad, tri, etc
  const Core::FE::CellType distype = shape();

  // gaussian points
  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
  // shape functions
  Core::LinAlg::SerialDenseVector shapefcts(iel);
  // natural derivatives of shape functions
  Core::LinAlg::SerialDenseMatrix deriv(numdim_, iel);

  // reference co-ordinates of element nodes
  Core::LinAlg::SerialDenseMatrix xrefe(numdim_, iel);


  /*----------------------------------------------------- geometry update */
  for (int k = 0; k < iel; ++k)
  {
    xrefe(0, k) = nodes()[k]->x()[0];
    xrefe(1, k) = nodes()[k]->x()[1];
  }

  /*=================================================== integration loops */
  for (int ip = 0; ip < intpoints.nquad; ++ip)
  {
    /*================================== gaussian point and weight at it */
    const double e1 = intpoints.qxg[ip][0];
    const double e2 = intpoints.qxg[ip][1];
    const double wgt = intpoints.qwgt[ip];

    /*-------------------- shape functions at gp e1,e2 on mid surface */
    if (distype != Core::FE::CellType::nurbs4 && distype != Core::FE::CellType::nurbs9)
    {
      // shape functions and their derivatives for polynomials
      Core::FE::shape_function_2d(shapefcts, e1, e2, distype);
      Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);
    }
    else
    {
      // nurbs version
      Core::LinAlg::SerialDenseVector gp(2);
      gp(0) = e1;
      gp(1) = e2;

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(shapefcts, deriv, gp, myknots, weights, distype);
    }

    /*--------------------------------------- compute jacobian Matrix */
    w1_jacobianmatrix(xrefe, deriv, xjm, &det, iel);

    /*------------------------------------ integration factor  -------*/
    double fac = wgt * det;

    // load vector ar
    double ar[noddof_];
    // loop the dofs of a node
    // ar[i] = ar[i] * facr * ds * onoff[i] * val[i]
    for (int i = 0; i < noddof_; ++i)
    {
      // factor given by spatial function
      double functfac = 1.0;
      if (funct[i].has_value() && funct[i].value() > 0)
      {
        // calculate reference position of GP
        Core::LinAlg::SerialDenseMatrix gp_coord(1, numdim_);
        gp_coord.multiply(Teuchos::TRANS, Teuchos::TRANS, 1.0, shapefcts, xrefe, 0.0);

        // write coordinates in another datatype
        double gp_coord2[3];  // the position vector has to be given in 3D!!!
        for (int k = 0; k < numdim_; k++) gp_coord2[k] = gp_coord(0, k);
        for (int k = numdim_; k < 3; k++)  // set a zero value for the remaining spatial directions
          gp_coord2[k] = 0.0;
        const double* coordgpref = gp_coord2;  // needed for function evaluation

        // evaluate function at current gauss point
        functfac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[i].value() - 1)
                       .evaluate(coordgpref, time, i);
      }

      ar[i] = fac * onoff[i] * val[i] * functfac;
    }

    // add load components
    for (int node = 0; node < iel; ++node)
      for (int dof = 0; dof < noddof_; ++dof)
        elevec1[node * noddof_ + dof] += shapefcts[node] * ar[dof];

  }  // for (int ip=0; ip<totngp; ++ip)

  // finished
  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_recover(const std::vector<int>& lm,
    const std::vector<double>& disp, const std::vector<double>& residual)
{
  // for eas
  Core::LinAlg::SerialDenseMatrix* alpha = nullptr;
  Core::LinAlg::SerialDenseMatrix* eas_inc = nullptr;
  // get access to the interface parameters
  const double step_length = str_params_interface().get_step_length();

  // have eas?
  if (iseas_)
  {
    // access general eas history stuff stored in element
    // get alpha of previous iteration
    alpha = &easdata_.alpha;
    // get the old eas increment
    eas_inc = &easdata_.eas_inc;
    if (!alpha || !eas_inc) FOUR_C_THROW("Missing EAS history data (eas_inc and/or alpha)");
  }

  /* if it is a default step, we have to recover the condensed
   * solution vectors */
  if (str_params_interface().is_default_step())
  {
    /* recovery of the enhanced assumed strain increment and
     * update of the eas dofs. */
    if (iseas_)
    {
      // first, store the eas state of the previous accepted Newton step
      str_params_interface().sum_into_my_previous_sol_norm(
          NOX::Nln::StatusTest::quantity_eas, w1_neas(), (*alpha)[0], owner());

      // get stored EAS history
      Core::LinAlg::SerialDenseMatrix* oldfeas = &easdata_.feas;
      Core::LinAlg::SerialDenseMatrix* oldKaainv = &easdata_.invKaa;
      Core::LinAlg::SerialDenseMatrix* oldKda = &easdata_.Kda;
      if (!oldKaainv or !oldKda or !oldfeas) FOUR_C_THROW("Missing EAS history-data");

      // we need the (residual) displacement at the previous step
      const int numnode = num_node();
      Core::LinAlg::SerialDenseVector res_d(2 * numnode);
      for (int i = 0; i < (2 * numnode); ++i)
      {
        res_d(i) = residual[i];
      }

      // add Kda . res_d to feas
      Core::LinAlg::multiply_tn(1.0, (*oldfeas), 1.0, *oldKda, res_d);
      // new alpha is: - Kaa^-1 . (feas + Kda . old_d), here: - Kaa^-1 . feas
      Core::LinAlg::multiply(1.0, (*alpha), -1.0, *oldKaainv, *oldfeas);
    }  // if (iseas)
  }  // if (*isdefault_step_ptr_)
  /* if it is no default step, we can correct the update and the current eas
   * state without the need for any matrix-vector products. */
  else
  {
    // The first step has to be a default step!
    if (old_step_length_ < 0.0) FOUR_C_THROW("The old step length was not defined!");
    /* if this is no full step, we have to adjust the length of the
     * enhanced assumed strain incremental step. */
    if (iseas_)
    {
      /* undo the previous step:
       *            alpha_new = alpha_old - old_step * alpha_inc
       * and update the solution variable with the new step length:
       *            alpha_new = alpha_new + new_step * alpha_inc */
      for (int i = 0; i < Wall1::neas_; ++i)
        (*alpha)(i, 0) += (step_length - old_step_length_) * (*eas_inc)(i, 0);
    }  // if (nhyb_)
  }  // else
  // save the old step length
  old_step_length_ = step_length;

  // Check if the eas incr is tested and if yes, calculate the element
  // contribution to the norm
  if (iseas_)
    str_params_interface().sum_into_my_update_norm(NOX::Nln::StatusTest::quantity_eas, w1_neas(),
        (*eas_inc)[0], (*alpha)[0], step_length, owner());

  // the element internal stuff should be up-to-date for now...
}

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                            mgit 03/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_nlnstiffmass(const std::vector<int>& lm,
    const std::vector<double>& disp, const std::vector<double>& residual,
    std::vector<Core::LinAlg::SerialDenseVector>& myknots,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseMatrix* elestress,
    Core::LinAlg::SerialDenseMatrix* elestrain, std::shared_ptr<const Core::Mat::Material> material,
    Teuchos::ParameterList& params,  ///< algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress, const Inpar::Solid::StrainType iostrain)
{
  const int numnode = num_node();
  const int numdf = 2;
  const int nd = numnode * numdf;

  // general arrays
  Core::LinAlg::SerialDenseVector funct(numnode);
  Core::LinAlg::SerialDenseMatrix deriv;
  deriv.shape(2, numnode);
  Core::LinAlg::SerialDenseMatrix xjm;
  xjm.shape(2, 2);
  Core::LinAlg::SerialDenseMatrix boplin;
  boplin.shape(4, 2 * numnode);
  Core::LinAlg::SerialDenseVector F;
  F.size(4);
  Core::LinAlg::SerialDenseVector strain;
  strain.size(4);
  double det;
  Core::LinAlg::SerialDenseMatrix xrefe(2, numnode);
  Core::LinAlg::SerialDenseMatrix xcure(2, numnode);
  const int numeps = 4;
  Core::LinAlg::SerialDenseMatrix b_cure;
  b_cure.shape(numeps, nd);
  Core::LinAlg::SerialDenseMatrix stress;
  stress.shape(4, 4);
  Core::LinAlg::SerialDenseMatrix C;
  C.shape(4, 4);

  // for EAS, in any case declare variables, sizes etc. only in eascase
  Core::LinAlg::SerialDenseMatrix* alpha = nullptr;      // EAS alphas
  Core::LinAlg::SerialDenseMatrix F_enh;                 // EAS matrix F_enh
  Core::LinAlg::SerialDenseMatrix F_tot;                 // EAS vector F_tot
  Core::LinAlg::SerialDenseMatrix p_stress;              // first piola-kirchhoff stress vector
  Core::LinAlg::SerialDenseMatrix xjm0;                  // Jacobian Matrix (origin)
  Core::LinAlg::SerialDenseVector F0;                    // Deformation Gradient (origin)
  Core::LinAlg::SerialDenseMatrix boplin0;               // B operator (origin)
  Core::LinAlg::SerialDenseMatrix W0;                    // W operator (origin)
  Core::LinAlg::SerialDenseMatrix G;                     // G operator
  Core::LinAlg::SerialDenseMatrix Z;                     // Z operator
  Core::LinAlg::SerialDenseMatrix FCF;                   // FCF^T
  Core::LinAlg::SerialDenseMatrix Kda;                   // EAS matrix Kda
  Core::LinAlg::SerialDenseMatrix Kaa;                   // EAS matrix Kaa
  Core::LinAlg::SerialDenseVector feas;                  // EAS portion of internal forces
  double detJ0;                                          // detJ(origin)
  Core::LinAlg::SerialDenseMatrix* oldfeas = nullptr;    // EAS history
  Core::LinAlg::SerialDenseMatrix* oldKaainv = nullptr;  // EAS history
  Core::LinAlg::SerialDenseMatrix* oldKda = nullptr;     // EAS history

  // arrays for structure with ale (fractional step strategy)
  Core::LinAlg::SerialDenseMatrix xmat;
  Core::LinAlg::SerialDenseMatrix boplinmat;
  Core::LinAlg::SerialDenseVector Fmat;
  Core::LinAlg::SerialDenseVector FFmatinv;

  // ------------------------------------ check calculation of mass matrix
  double density = 0.0;
  if (massmatrix) density = material->density();

  /*------- get integration data ---------------------------------------- */
  const Core::FE::CellType distype = shape();

  // gaussian points
  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

  /*----------------------------------------------------- geometry update */
  for (int k = 0; k < numnode; ++k)
  {
    xrefe(0, k) = nodes()[k]->x()[0];
    xrefe(1, k) = nodes()[k]->x()[1];
    xcure(0, k) = xrefe(0, k) + disp[k * numdf + 0];
    xcure(1, k) = xrefe(1, k) + disp[k * numdf + 1];
  }

  /*--------------------------------- get node weights for nurbs elements */
  Core::LinAlg::SerialDenseVector weights(numnode);
  if (distype == Core::FE::CellType::nurbs4 || distype == Core::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnode; ++inode)
    {
      auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);

      weights(inode) = cp->w();
    }
  }

  if (iseas_)
  {
    // allocate EAS quantities
    F_enh.shape(4, 1);
    F_tot.shape(4, 3);
    p_stress.shape(4, 1);
    xjm0.shape(2, 2);
    F0.size(4);
    boplin0.shape(4, 2 * numnode);
    W0.shape(4, 2 * numnode);
    G.shape(4, Wall1::neas_);
    Z.shape(2 * numnode, Wall1::neas_);
    FCF.shape(4, 4);
    Kda.shape(2 * numnode, Wall1::neas_);
    Kaa.shape(Wall1::neas_, Wall1::neas_);
    feas.size(Wall1::neas_);

    /*
    ** EAS Update of alphas:
    ** the current alphas are (re-)evaluated out of
    ** Kaa and Kda of previous step to avoid additional element call.
    ** This corresponds to the (innermost) element update loop
    ** in the nonlinear FE-Skript page 120 (load-control alg. with EAS)
    */
    alpha = &easdata_.alpha;  // get alpha of previous iteration

    // get stored EAS history
    oldfeas = &easdata_.feas;
    oldKaainv = &easdata_.invKaa;
    oldKda = &easdata_.Kda;
    if (!alpha || !oldKaainv || !oldKda || !oldfeas) FOUR_C_THROW("Missing EAS history-data");
    // FixMe deprecated implementation
    if (not is_params_interface())
    {
      // we need the (residual) displacement at the previous step
      Core::LinAlg::SerialDenseVector res_d(2 * numnode);
      for (int i = 0; i < (2 * numnode); ++i)
      {
        res_d(i) = residual[i];
      }

      // add Kda . res_d to feas
      Core::LinAlg::multiply_tn(1.0, (*oldfeas), 1.0, *oldKda, res_d);
      // new alpha is: - Kaa^-1 . (feas + Kda . old_d), here: - Kaa^-1 . feas
      Core::LinAlg::multiply(1.0, (*alpha), -1.0, *oldKaainv, *oldfeas);
    }  // if (not IsInterface())
    /* end of EAS Update ******************/

    /* evaluation of EAS variables (which are constant for the following):
    ** -> M defining interpolation of enhanced strains alpha, evaluated at GPs
    ** -> determinant of Jacobi matrix at element origin (r=s=t=0.0)
    ** -> T0^{-T}
    */
    w1_eassetup(boplin0, F0, xjm0, detJ0, xrefe, xcure, distype);
  }

  /*=================================================== integration loops */
  for (int ip = 0; ip < intpoints.nquad; ++ip)
  {
    /*================================== gaussian point and weight at it */
    const double e1 = intpoints.qxg[ip][0];
    const double e2 = intpoints.qxg[ip][1];
    const double wgt = intpoints.qwgt[ip];

    // get values of shape functions and derivatives in the gausspoint
    if (distype != Core::FE::CellType::nurbs4 && distype != Core::FE::CellType::nurbs9)
    {
      // shape functions and their derivatives for polynomials
      Core::FE::shape_function_2d(funct, e1, e2, distype);
      Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);
    }
    else
    {
      // nurbs version
      Core::LinAlg::SerialDenseVector gp(2);
      gp(0) = e1;
      gp(1) = e2;

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(funct, deriv, gp, myknots, weights, distype);
    }

    /*--------------------------------------- compute jacobian Matrix */
    w1_jacobianmatrix(xrefe, deriv, xjm, &det, numnode);

    /*------------------------------------ integration factor  -------*/
    double fac = wgt * det * thickness_;

    /*------------------------------compute mass matrix if imass-----*/
    if (massmatrix)
    {
      double facm = fac * density;
      for (int a = 0; a < numnode; a++)
      {
        for (int b = 0; b < numnode; b++)
        {
          (*massmatrix)(2 * a, 2 * b) += facm * funct(a) * funct(b);         /* a,b even */
          (*massmatrix)(2 * a + 1, 2 * b + 1) += facm * funct(a) * funct(b); /* a,b odd  */
        }
      }
    }

    /*----------------------------------- calculate operator Blin  ---*/
    w1_boplin(boplin, deriv, xjm, det, numnode);
    // cout.precision(16);
    /*------------ calculate defgrad F^u, Green-Lagrange-strain E^u --*/
    w1_defgrad(F, strain, xrefe, xcure, boplin, numnode);

    /*-calculate defgrad F in matrix notation and Blin in current conf.*/
    w1_boplin_cure(b_cure, boplin, F, numeps, nd);

    // EAS technology: "enhance the deformation gradient"  ---- --- EAS
    if (iseas_ == true)
    {
      /*-----calculate the enhanced deformation gradient and--------------------
      -----alsoe the operators G, W0 and Z------------------------------------*/

      w1_call_defgrad_enh(F_enh, xjm0, xjm, detJ0, det, F0, *alpha, e1, e2, G, W0, boplin0, Z);

      /*-----total deformation gradient, Green-Lagrange-strain E^F -----------*/
      w1_call_defgrad_tot(F_enh, F_tot, F, strain);
      /* call material law----------------------------------------------------*/
      w1_call_matgeononl(strain, stress, C, numeps, material, params, ip);

      // return gp strains (only in case of strain output)
      switch (iostrain)
      {
        case Inpar::Solid::strain_gl:
        {
          if (elestrain == nullptr) FOUR_C_THROW("no strain data available");
          (*elestrain)(ip, 0) = strain(0);
          (*elestrain)(ip, 1) = strain(1);
          (*elestrain)(ip, 2) = 0.0;
          (*elestrain)(ip, 3) = strain(3);
        }
        break;
        case Inpar::Solid::strain_none:
          break;
        case Inpar::Solid::strain_ea:
        default:
          FOUR_C_THROW("requested strain type not supported");
          break;
      }

      // return gp stresses (only in case of stress output)
      switch (iostress)
      {
        case Inpar::Solid::stress_2pk:
        {
          if (elestress == nullptr) FOUR_C_THROW("no stress data available");
          (*elestress)(ip, 0) = stress(0, 0);
          (*elestress)(ip, 1) = stress(1, 1);
          (*elestress)(ip, 2) = 0.0;
          (*elestress)(ip, 3) = stress(0, 2);
        }
        break;
        case Inpar::Solid::stress_cauchy:
        {
          if (elestress == nullptr) FOUR_C_THROW("no stress data available");
          stress_cauchy(ip, F_tot(0, 0), F_tot(1, 1), F_tot(0, 2), F_tot(1, 2), stress, elestress);
        }
        break;
        case Inpar::Solid::stress_none:
          break;
        default:
          FOUR_C_THROW("requested stress type not supported");
          break;
      }

      /*-----first piola-kirchhoff stress vector------------------------------*/
      w1_stress_eas(stress, F_tot, p_stress);

      /*-----stiffness matrix kdd---------------------------------------------*/
      if (stiffmatrix) w1_kdd(boplin, W0, F_tot, C, stress, FCF, *stiffmatrix, fac);
      /*-----matrix kda-------------------------------------------------------*/
      w1_kda(FCF, W0, boplin, stress, G, Z, Kda, p_stress, fac);
      /*-----matrix kaa-------------------------------------------------------*/
      w1_kaa(FCF, stress, G, Kaa, fac);
      /*-----nodal forces ----------------------------------------------------*/
      if (force) w1_fint_eas(W0, boplin, G, p_stress, *force, feas, fac);
    }
    else
    {
      w1_call_matgeononl(strain, stress, C, numeps, material, params, ip);

      // return gp strains (only in case of strain output)
      switch (iostrain)
      {
        case Inpar::Solid::strain_gl:
        {
          if (elestrain == nullptr) FOUR_C_THROW("no strain data available");
          (*elestrain)(ip, 0) = strain(0);
          (*elestrain)(ip, 1) = strain(1);
          (*elestrain)(ip, 2) = 0.0;
          (*elestrain)(ip, 3) = strain(3);
        }
        break;
        case Inpar::Solid::strain_none:
          break;
        case Inpar::Solid::strain_ea:
        default:
          FOUR_C_THROW("requested strain type not supported");
          break;
      }

      // return gp stresses (only in case of stress output)
      switch (iostress)
      {
        case Inpar::Solid::stress_2pk:
        {
          if (elestress == nullptr) FOUR_C_THROW("no stress data available");
          (*elestress)(ip, 0) = stress(0, 0);
          (*elestress)(ip, 1) = stress(1, 1);
          (*elestress)(ip, 2) = 0.0;
          (*elestress)(ip, 3) = stress(0, 2);
        }
        break;
        case Inpar::Solid::stress_cauchy:
        {
          if (elestress == nullptr) FOUR_C_THROW("no stress data available");
          stress_cauchy(ip, F[0], F[1], F[2], F[3], stress, elestress);
        }
        break;
        case Inpar::Solid::stress_none:
          break;
        default:
          FOUR_C_THROW("requested stress type not supported");
          break;
      }

      /*---------------------- geometric part of stiffness matrix kg ---*/
      if (stiffmatrix) w1_kg(*stiffmatrix, boplin, stress, fac, nd, numeps);

      /*------------------ elastic+displacement stiffness matrix keu ---*/
      if (stiffmatrix) w1_keu(*stiffmatrix, b_cure, C, fac, nd, numeps);

      /*--------------- nodal forces fi from integration of stresses ---*/
      if (force) w1_fint(stress, b_cure, *force, fac, nd);
    }

  }  // for (int ip=0; ip<totngp; ++ip)


  // EAS technology: ------------------------------------------------------ EAS
  // subtract EAS matrices from disp-based Kdd to "soften" element

  if (force != nullptr && stiffmatrix != nullptr)
  {
    if (iseas_)
    {
      // we need the inverse of Kaa
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
      Teuchos::SerialDenseSolver<ordinalType, scalarType> solve_for_inverseKaa;
      solve_for_inverseKaa.setMatrix(Teuchos::rcpFromRef(Kaa));
      solve_for_inverseKaa.invert();


      Core::LinAlg::SerialDenseMatrix KdaKaa(
          2 * num_node(), Wall1::neas_);  // temporary Kda.Kaa^{-1}
      Core::LinAlg::multiply(1.0, KdaKaa, 1.0, Kda, Kaa);


      // EAS-stiffness matrix is: Kdd - Kda^T . Kaa^-1 . Kad  with Kad=Kda^T
      if (stiffmatrix) Core::LinAlg::multiply_nt(1.0, (*stiffmatrix), -1.0, KdaKaa, Kda);

      // EAS-internal force is: fint - Kda^T . Kaa^-1 . feas
      if (force) Core::LinAlg::multiply(1.0, *force, -1.0, KdaKaa, feas);

      // store current EAS data in history
      for (int i = 0; i < Wall1::neas_; ++i)
        for (int j = 0; j < Wall1::neas_; ++j) (*oldKaainv)(i, j) = Kaa(i, j);

      for (int i = 0; i < (2 * num_node()); ++i)
      {
        for (int j = 0; j < Wall1::neas_; ++j)
        {
          (*oldKda)(i, j) = Kda(i, j);
          (*oldfeas)(j, 0) = feas(j);
        }
      }
    }
  }
  // -------------------------------------------------------------------- EAS
}

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                            popp 09/11|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_linstiffmass(const std::vector<int>& lm,
    const std::vector<double>& disp, const std::vector<double>& residual,
    std::vector<Core::LinAlg::SerialDenseVector>& myknots,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseMatrix* elestress,
    Core::LinAlg::SerialDenseMatrix* elestrain, std::shared_ptr<const Core::Mat::Material> material,
    Teuchos::ParameterList& params, const Inpar::Solid::StressType iostress,
    const Inpar::Solid::StrainType iostrain)
{
  const int numnode = num_node();
  const int numdf = 2;
  const int nd = numnode * numdf;

  // general arrays
  Core::LinAlg::SerialDenseVector funct(numnode);
  Core::LinAlg::SerialDenseMatrix deriv;
  deriv.shape(2, numnode);
  Core::LinAlg::SerialDenseMatrix xjm;
  xjm.shape(2, 2);
  Core::LinAlg::SerialDenseMatrix boplin;
  boplin.shape(4, 2 * numnode);
  Core::LinAlg::SerialDenseVector F;
  F.size(4);
  Core::LinAlg::SerialDenseVector strain;
  strain.size(4);
  double det;
  Core::LinAlg::SerialDenseMatrix xrefe(2, numnode);
  Core::LinAlg::SerialDenseMatrix xcure(2, numnode);
  const int numeps = 4;
  Core::LinAlg::SerialDenseMatrix b_cure;
  b_cure.shape(numeps, nd);
  Core::LinAlg::SerialDenseMatrix stress;
  stress.shape(4, 4);
  Core::LinAlg::SerialDenseMatrix C;
  C.shape(4, 4);

  // ------------------------------------ check calculation of mass matrix
  double density = 0.0;
  if (massmatrix) density = material->density();

  /*------- get integration data ---------------------------------------- */
  const Core::FE::CellType distype = shape();

  // gaussian points
  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

  /*----------------------------------------------------- geometry update */
  for (int k = 0; k < numnode; ++k)
  {
    xrefe(0, k) = nodes()[k]->x()[0];
    xrefe(1, k) = nodes()[k]->x()[1];
    xcure(0, k) = xrefe(0, k) + disp[k * numdf + 0];
    xcure(1, k) = xrefe(1, k) + disp[k * numdf + 1];
  }

  /*--------------------------------- get node weights for nurbs elements */
  Core::LinAlg::SerialDenseVector weights(numnode);
  if (distype == Core::FE::CellType::nurbs4 || distype == Core::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnode; ++inode)
    {
      auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);

      weights(inode) = cp->w();
    }
  }

  /*=================================================== integration loops */
  for (int ip = 0; ip < intpoints.nquad; ++ip)
  {
    /*================================== gaussian point and weight at it */
    const double e1 = intpoints.qxg[ip][0];
    const double e2 = intpoints.qxg[ip][1];
    const double wgt = intpoints.qwgt[ip];

    // get values of shape functions and derivatives in the gausspoint
    if (distype != Core::FE::CellType::nurbs4 && distype != Core::FE::CellType::nurbs9)
    {
      // shape functions and their derivatives for polynomials
      Core::FE::shape_function_2d(funct, e1, e2, distype);
      Core::FE::shape_function_2d_deriv1(deriv, e1, e2, distype);
    }
    else
    {
      // nurbs version
      Core::LinAlg::SerialDenseVector gp(2);
      gp(0) = e1;
      gp(1) = e2;

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(funct, deriv, gp, myknots, weights, distype);
    }

    /*--------------------------------------- compute jacobian Matrix */
    w1_jacobianmatrix(xrefe, deriv, xjm, &det, numnode);

    /*------------------------------------ integration factor  -------*/
    double fac = wgt * det * thickness_;

    /*------------------------------compute mass matrix if imass-----*/
    if (massmatrix)
    {
      double facm = fac * density;
      for (int a = 0; a < numnode; a++)
      {
        for (int b = 0; b < numnode; b++)
        {
          (*massmatrix)(2 * a, 2 * b) += facm * funct(a) * funct(b);         /* a,b even */
          (*massmatrix)(2 * a + 1, 2 * b + 1) += facm * funct(a) * funct(b); /* a,b odd  */
        }
      }
    }

    /*----------------------------------- calculate operator Blin  ---*/
    w1_boplin(boplin, deriv, xjm, det, numnode);

    /*-------------------------deformation gradient and GL strains ---*/
    w1_defgrad(F, strain, xrefe, xcure, boplin, numnode);

    /*--------------redefine strains -> linear engineering strains ---*/
    strain[0] = 0.5 * (F[0] + F[0]) - 1.0;
    strain[1] = 0.5 * (F[1] + F[1]) - 1.0;
    strain[2] = 0.5 * (F[2] + F[3]);
    strain[3] = strain[2];

    // material call
    w1_call_matgeononl(strain, stress, C, numeps, material, params, ip);

    // return gp strains (only in case of strain output)
    switch (iostrain)
    {
      case Inpar::Solid::strain_gl:
      {
        if (elestrain == nullptr) FOUR_C_THROW("no strain data available");
        (*elestrain)(ip, 0) = strain(0);
        (*elestrain)(ip, 1) = strain(1);
        (*elestrain)(ip, 2) = 0.0;
        (*elestrain)(ip, 3) = strain(3);
      }
      break;
      case Inpar::Solid::strain_none:
        break;
      case Inpar::Solid::strain_ea:
      default:
        FOUR_C_THROW("requested strain type not supported");
        break;
    }

    // return gp stresses (only in case of stress output)
    switch (iostress)
    {
      case Inpar::Solid::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("no stress data available");
        (*elestress)(ip, 0) = stress(0, 0);
        (*elestress)(ip, 1) = stress(1, 1);
        (*elestress)(ip, 2) = 0.0;
        (*elestress)(ip, 3) = stress(0, 2);
      }
      break;
      case Inpar::Solid::stress_cauchy:
      {
        if (elestress == nullptr) FOUR_C_THROW("no stress data available");
        stress_cauchy(ip, F[0], F[1], F[2], F[3], stress, elestress);
      }
      break;
      case Inpar::Solid::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not supported");
        break;
    }

    /*-------------------------------- linear stiffness matrix keu ---*/
    if (stiffmatrix) w1_keu(*stiffmatrix, boplin, C, fac, nd, numeps);

    /*--------------- nodal forces fi from integration of stresses ---*/
    if (force) w1_fint(stress, boplin, *force, fac, nd);

  }  // for (int ip=0; ip<totngp; ++ip)
}

/*----------------------------------------------------------------------*
 |  jacobian matrix (private)                                  mgit 04/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_jacobianmatrix(const Core::LinAlg::SerialDenseMatrix& xrefe,
    const Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& xjm, double* det,
    const int iel)
{
  xjm.putScalar(0.0);

  for (int k = 0; k < iel; k++)
  {
    xjm(0, 0) += deriv(0, k) * xrefe(0, k);
    xjm(0, 1) += deriv(0, k) * xrefe(1, k);
    xjm(1, 0) += deriv(1, k) * xrefe(0, k);
    xjm(1, 1) += deriv(1, k) * xrefe(1, k);
  }

  /*------------------------------------------ determinant of jacobian ---*/
  *det = xjm[0][0] * xjm[1][1] - xjm[1][0] * xjm[0][1];

  if (*det < 0.0) FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT %8.5f in ELEMENT %d\n", *det, id());
  /*----------------------------------------------------------------------*/
}  // Discret::Elements::Wall1::w1_jacobianmatrix

/*----------------------------------------------------------------------*
 |  Matrix boplin in reference configuration (private)         mgit 04/07|
 *----------------------------------------------------------------------*/

void Discret::Elements::Wall1::w1_boplin(Core::LinAlg::SerialDenseMatrix& boplin,
    Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& xjm, double& det,
    const int iel)
{
  double inv_det;
  double xji[2][2];
  /*---------------------------------------------- inverse of jacobian ---*/
  inv_det = 1.0 / det;
  xji[0][0] = xjm(1, 1) * inv_det;
  xji[0][1] = -xjm(0, 1) * inv_det;
  xji[1][0] = -xjm(1, 0) * inv_det;
  xji[1][1] = xjm(0, 0) * inv_det;
  /*----------------------------- get operator boplin of global derivatives -*/
  /*-------------- some comments, so that even fluid people are able to
   understand this quickly :-)
   the Boplin looks like
       | Nk,x    0   |
       |   0    Nk,y |
       | Nk,y    0   |
       |  0     Nk,x |
  */
  for (int inode = 0; inode < iel; inode++)
  {
    int dnode = inode * 2;

    boplin(0, dnode + 0) = deriv(0, inode) * xji[0][0] + deriv(1, inode) * xji[0][1];
    boplin(1, dnode + 1) = deriv(0, inode) * xji[1][0] + deriv(1, inode) * xji[1][1];
    boplin(2, dnode + 0) = boplin(1, dnode + 1);
    boplin(3, dnode + 1) = boplin(0, dnode + 0);
  } /* end of loop over nodes */
}

/* Discret::Elements::Wall1::w1_boplin */

/*----------------------------------------------------------------------*
 | Deformation gradient F and Green-Langrange strain (private)  mgit 04/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_defgrad(Core::LinAlg::SerialDenseVector& F,
    Core::LinAlg::SerialDenseVector& strain, const Core::LinAlg::SerialDenseMatrix& xrefe,
    const Core::LinAlg::SerialDenseMatrix& xcure, Core::LinAlg::SerialDenseMatrix& boplin,
    const int iel)
{
  /*------------------calculate defgrad --------- (Summenschleife->+=) ---*
  defgrad looks like:

        |  1 + Ux,X  |
        |  1 + Uy,Y  |
        |      Ux,Y  |
        |      Uy,X  |
  */

  F.putScalar(0.0);

  F[0] = 1;
  F[1] = 1;
  for (int inode = 0; inode < iel; inode++)
  {
    F[0] += boplin(0, 2 * inode) * (xcure(0, inode) - xrefe(0, inode));      // F_11
    F[1] += boplin(1, 2 * inode + 1) * (xcure(1, inode) - xrefe(1, inode));  // F_22
    F[2] += boplin(2, 2 * inode) * (xcure(0, inode) - xrefe(0, inode));      // F_12
    F[3] += boplin(3, 2 * inode + 1) * (xcure(1, inode) - xrefe(1, inode));  // F_21
  } /* end of loop over nodes */

  /*-----------------------calculate Green-Lagrange strain E -------------*/
  strain[0] = 0.5 * (F[0] * F[0] + F[3] * F[3] - 1.0);  // E_11
  strain[1] = 0.5 * (F[2] * F[2] + F[1] * F[1] - 1.0);  // E_22
  strain[2] = 0.5 * (F[0] * F[2] + F[3] * F[1]);        // E_12
  strain[3] = strain[2];                                // E_21

  /*-----------------------linear engineering strain eps -----------------*/
  /* (choose 2PK stresses for stress output, when using linear strains!)  */
  // strain[0] = 0.5 * (F[0] + F[0]) - 1.0;
  // strain[1] = 0.5 * (F[1] + F[1]) - 1.0;
  // strain[2] = 0.5 * (F[2] + F[3]);
  // strain[3] = strain[2];
} /* Discret::Elements::Wall1::w1_defgrad */

/*----------------------------------------------------------------------*
 | Deformation gradient F in matrix notation and B in
 reference configuration (private)                             mgit 04/07|
 *----------------------------------------------------------------------*/

void Discret::Elements::Wall1::w1_boplin_cure(Core::LinAlg::SerialDenseMatrix& b_cure,
    const Core::LinAlg::SerialDenseMatrix& boplin, const Core::LinAlg::SerialDenseVector& F,
    const int numeps, const int nd)
{
  Core::LinAlg::SerialDenseMatrix Fmatrix;
  Fmatrix.shape(4, 4);


  /*---------------------------write Vector F as a matrix Fmatrix*/

  Fmatrix(0, 0) = F[0];
  Fmatrix(0, 2) = 0.5 * F[2];
  Fmatrix(0, 3) = 0.5 * F[2];
  Fmatrix(1, 1) = F[1];
  Fmatrix(1, 2) = 0.5 * F[3];
  Fmatrix(1, 3) = 0.5 * F[3];
  Fmatrix(2, 1) = F[2];
  Fmatrix(2, 2) = 0.5 * F[0];
  Fmatrix(2, 3) = 0.5 * F[0];
  Fmatrix(3, 0) = F[3];
  Fmatrix(3, 2) = 0.5 * F[1];
  Fmatrix(3, 3) = 0.5 * F[1];

  /*-------------------------------------------------int_b_cure operator*/
  b_cure.putScalar(0.0);
  for (int i = 0; i < numeps; i++)
    for (int j = 0; j < nd; j++)
      for (int k = 0; k < numeps; k++) b_cure(i, j) += Fmatrix(k, i) * boplin(k, j);
  /*----------------------------------------------------------------*/
}

/*----------------------------------------------------------------------*
| geometric stiffness part (total lagrange)                   mgit 05/07|
*----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_kg(Core::LinAlg::SerialDenseMatrix& estif,
    const Core::LinAlg::SerialDenseMatrix& boplin, const Core::LinAlg::SerialDenseMatrix& stress,
    const double fac, const int nd, const int numeps)
{
  /*---------------------------------------------- perform B^T * SIGMA * B*/
  for (int i = 0; i < nd; i++)
  {
    for (int j = 0; j < nd; j++)
    {
      for (int r = 0; r < numeps; r++)
      {
        for (int m = 0; m < numeps; m++)
          estif(i, j) += boplin(r, i) * stress(r, m) * boplin(m, j) * fac;
      }
    }
  }
}  // Discret::Elements::Wall1::w1_kg

/*----------------------------------------------------------------------*
| elastic and initial displacement stiffness (total lagrange)  mgit 05/07
*----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_keu(Core::LinAlg::SerialDenseMatrix& estif,
    const Core::LinAlg::SerialDenseMatrix& b_cure, const Core::LinAlg::SerialDenseMatrix& C,
    const double fac, const int nd, const int numeps)
{
  /*------------- perform B_cure^T * D * B_cure, whereas B_cure = F^T * B */
  for (int i = 0; i < nd; i++)
  {
    for (int j = 0; j < nd; j++)
    {
      for (int k = 0; k < numeps; k++)
      {
        for (int m = 0; m < numeps; m++)
        {
          estif(i, j) += b_cure(k, i) * C(k, m) * b_cure(m, j) * fac;
        }
      }
    }
  }
}  // Discret::Elements::Wall1::w1_keu


/*----------------------------------------------------------------------*
 | evaluate internal element forces for large def (total Lagr) mgit 05/07  |
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_fint(const Core::LinAlg::SerialDenseMatrix& stress,
    const Core::LinAlg::SerialDenseMatrix& b_cure, Core::LinAlg::SerialDenseVector& intforce,
    const double fac, const int nd)

{
  Core::LinAlg::SerialDenseVector st;
  st.size(4);

  st[0] = fac * stress(0, 0);
  st[1] = fac * stress(1, 1);
  st[2] = fac * stress(0, 2);
  st[3] = fac * stress(0, 2);

  for (int i = 0; i < nd; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      intforce[i] += b_cure(j, i) * st[j];
    }
  }
}  // Discret::Elements::Wall1::w1_fint


/*-----------------------------------------------------------------------------*
| lump mass matrix                                                  bborn 07/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_lumpmass(Core::LinAlg::SerialDenseMatrix* emass)
{
  // lump mass matrix
  if (emass != nullptr)
  {
    // we assume #elemat2 is a square matrix
    for (int c = 0; c < (*emass).numCols(); ++c)  // parse columns
    {
      double d = 0.0;
      for (int r = 0; r < (*emass).numRows(); ++r)  // parse rows
      {
        d += (*emass)(r, c);  // accumulate row entries
        (*emass)(r, c) = 0.0;
      }
      (*emass)(c, c) = d;  // apply sum of row entries on diagonal
    }
  }
}  // w1_lumpmass

/*-----------------------------------------------------------------------------*
| deliver Cauchy stress                                             bborn 08/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::stress_cauchy(const int ip, const double& F11, const double& F22,
    const double& F12, const double& F21, const Core::LinAlg::SerialDenseMatrix& stress,
    Core::LinAlg::SerialDenseMatrix* elestress)
{
  // Question: Is this true for plane stress and/or plane strain mode?

  double detf = F11 * F22 - F12 * F21;
  // Def.grad. tensor in Cartesian matrix notation
  Core::LinAlg::SerialDenseMatrix defgrad(2, 2);
  defgrad(0, 0) = F11;
  defgrad(0, 1) = F12;
  defgrad(1, 0) = F21;
  defgrad(1, 1) = F22;
  // PK2 stress tensor in Cartesian matrix notation
  Core::LinAlg::SerialDenseMatrix pk2stress(2, 2);
  pk2stress(0, 0) = stress(0, 0);
  pk2stress(0, 1) = stress(0, 2);
  pk2stress(1, 0) = stress(0, 2);
  pk2stress(1, 1) = stress(1, 1);

  // PK1 stress tensor in Cartesian matrix notation
  Core::LinAlg::SerialDenseMatrix pk1stress(2, 2);
  Core::LinAlg::multiply_nt(0.0, pk1stress, 1.0 / detf, pk2stress, defgrad);

  // Cauchy stress tensor in Cartesian matrix notation
  Core::LinAlg::SerialDenseMatrix cauchystress(2, 2);
  Core::LinAlg::multiply(cauchystress, defgrad, pk1stress);

  // copy results to array for output
  (*elestress)(ip, 0) = cauchystress(0, 0);
  (*elestress)(ip, 1) = cauchystress(1, 1);
  (*elestress)(ip, 2) = 0.0;
  (*elestress)(ip, 3) = cauchystress(0, 1);
}  // stress_cauchy


/*-----------------------------------------------------------------------------*
| deliver Cauchy stress                                             bborn 08/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::energy(Teuchos::ParameterList& params, const std::vector<int>& lm,
    const std::vector<double>& dis, Core::LinAlg::SerialDenseVector* energies,
    std::shared_ptr<const Core::Mat::Material> material)
{
  // constants
  // element properties
  const int numnode = num_node();
  const int edof = numnode * Wall1::noddof_;
  const Core::FE::CellType distype = shape();
  // Gaussian points
  const Core::FE::IntegrationPoints2D intpoints(gaussrule_);

  // internal/strain energy
  double internal_energy = 0.0;

  // general arrays
  Core::LinAlg::SerialDenseVector shpfct(numnode);  // shape functions at Gauss point
  Core::LinAlg::SerialDenseMatrix shpdrv(
      Wall1::numdim_, numnode);  // parametric derivatives of shape funct. at Gauss point
  Core::LinAlg::SerialDenseMatrix Xjm(
      Wall1::numdim_, Wall1::numdim_);  // material-to-parameter-space Jacobian
  double Xjdet;                         // determinant of #Xjm
  Core::LinAlg::SerialDenseMatrix boplin(4, edof);
  Core::LinAlg::SerialDenseVector Fuv(4);  // disp-based def.grad. vector at t_{n}
  Core::LinAlg::SerialDenseVector Ev(4);   // Green-Lagrange strain vector at t_{n}
  Core::LinAlg::SerialDenseMatrix Xe(
      Wall1::numdim_, numnode);  // material/initial element co-ordinates
  Core::LinAlg::SerialDenseMatrix xe(
      Wall1::numdim_, numnode);  // spatial/current element co-ordinates at t_{n}
  Core::LinAlg::SerialDenseMatrix bop(Wall1::numstr_, edof);  // non-linear B-op at t_{n}

  Core::LinAlg::SerialDenseMatrix massmatrix(lm.size(), lm.size());

  // for EAS, in any case declare variables, sizes etc. only allocated in EAS version
  Core::LinAlg::SerialDenseMatrix* alphao = nullptr;  // EAS alphas at t_{n}
  Core::LinAlg::SerialDenseMatrix Fenhv;              // EAS matrix Fenhv
  Core::LinAlg::SerialDenseMatrix Fm;                 // total def.grad. matrix at t_{n}
  Core::LinAlg::SerialDenseMatrix Xjm0;               // Jacobian Matrix (origin)
  double Xjdet0;                                      // determinant of #Xjm0
  Core::LinAlg::SerialDenseVector Fuv0;               // deformation gradient at origin at t_{n}
  Core::LinAlg::SerialDenseMatrix boplin0;            // B-operator (origin)
  Core::LinAlg::SerialDenseMatrix W0;                 // W-operator (origin) at t_{n}
  Core::LinAlg::SerialDenseMatrix G;                  // G-operator at t_{n}
  Core::LinAlg::SerialDenseMatrix Z;                  // Z-operator

  // element co-ordinates
  for (int k = 0; k < numnode; ++k)
  {
    Xe(0, k) = nodes()[k]->x()[0];
    Xe(1, k) = nodes()[k]->x()[1];
    xe(0, k) = Xe(0, k) + dis[k * Wall1::noddof_ + 0];
    xe(1, k) = Xe(1, k) + dis[k * Wall1::noddof_ + 1];
  }

  // set-up EAS parameters
  if (iseas_)
  {
    // allocate EAS quantities
    Fenhv.shape(4, 1);
    Fm.shape(4, 3);
    Xjm0.shape(2, 2);
    Fuv0.size(4);
    boplin0.shape(4, edof);
    W0.shape(4, edof);
    G.shape(4, Wall1::neas_);
    Z.shape(edof, Wall1::neas_);

    // get alpha of last converged state
    alphao = &easdata_.alphao;

    // derivatives at origin
    Core::FE::shape_function_2d_deriv1(shpdrv, 0.0, 0.0, distype);
    // material-to-parameter space Jacobian at origin
    w1_jacobianmatrix(Xe, shpdrv, Xjm0, &Xjdet0, numnode);
    // calculate linear B-operator at origin
    w1_boplin(boplin0, shpdrv, Xjm0, Xjdet0, numnode);
    // displ.-based def.grad. at origin
    w1_defgrad(Fuv0, Ev, Xe, xe, boplin0, numnode);  // at t_{n}
  }

  // integration loops over element domain
  for (int ip = 0; ip < intpoints.nquad; ++ip)
  {
    // Gaussian point and weight at it
    const double xi1 = intpoints.qxg[ip][0];
    const double xi2 = intpoints.qxg[ip][1];
    const double wgt = intpoints.qwgt[ip];

    // shape functions and their derivatives
    Core::FE::shape_function_2d(shpfct, xi1, xi2, distype);
    Core::FE::shape_function_2d_deriv1(shpdrv, xi1, xi2, distype);

    // compute Jacobian matrix
    w1_jacobianmatrix(Xe, shpdrv, Xjm, &Xjdet, numnode);

    // integration factor
    double fac = wgt * Xjdet * thickness_;

    // calculate linear B-operator
    w1_boplin(boplin, shpdrv, Xjm, Xjdet, numnode);

    // calculate defgrad F^u, Green-Lagrange-strain E^u
    w1_defgrad(Fuv, Ev, Xe, xe, boplin, numnode);  // at t_{n}

    // calculate non-linear B-operator in current configuration
    w1_boplin_cure(
        bop, boplin, Fuv, Wall1::numstr_, edof);  // at t_{n} // CHECK THIS: NOT SURE IF bopo NEEDED

    // EAS: The deformation gradient is enhanced
    if (iseas_)
    {
      // calculate the enhanced deformation gradient and
      // also the operators G, W0 and Z
      w1_call_defgrad_enh(
          Fenhv, Xjm0, Xjm, Xjdet0, Xjdet, Fuv0, *alphao, xi1, xi2, G, W0, boplin0, Z);  // at t_{n}

      // total deformation gradient F, and total Green-Lagrange-strain E
      w1_call_defgrad_tot(Fenhv, Fm, Fuv, Ev);  // at t_{n}
    }

    internal_energy += fac * energy_internal(material, params, Ev, ip);
  }  // end loop Gauss points


  if (is_params_interface())  // new structural time integration
  {
    str_params_interface().add_contribution_to_energy_type(internal_energy, Solid::internal_energy);
  }
  else if (energies)  // old structural time integration
  {
    // check length of elevec1
    if ((*energies).length() < 1) FOUR_C_THROW("The given result vector is too short.");

    (*energies)(0) += internal_energy;
  }
}

FOUR_C_NAMESPACE_CLOSE
