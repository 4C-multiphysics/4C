// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_w1_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  preevaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1Scatra::pre_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  const int numnode = num_node();

  if (la.size() > 1)
  {
    //  dofs per node of second dofset
    const int numdofpernode = num_dof_per_node(1, *(nodes()[0]), discretization.name());

    if (la[1].size() != numnode * numdofpernode)
      FOUR_C_THROW("calc_struct_nlnstiff: Location vector length for velocities does not match!");

    if (discretization.has_state(1, "scalarfield"))
    {
      // check if you can get the scalar state
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp =
          discretization.get_state(1, "scalarfield");

      if (phinp == nullptr) FOUR_C_THROW("pre_evaluate: Cannot get state vector 'phinp' ");

      // extract local values of the global vectors
      std::vector<double> myphi = Core::FE::extract_values(*phinp, la[1].lm_);

      double meanphi = 0.0;
      for (int i = 0; i < numnode; ++i)
      {
        meanphi += (myphi)[i] / numnode;
      }
      params.set<double>("scalar", meanphi);
    }
    // Get pointer for scatra material in the same element
    std::shared_ptr<Core::FE::Discretization> scatradis = nullptr;
    scatradis = Global::Problem::instance()->get_dis("scatra");
    Core::Elements::Element* scatraele = scatradis->g_element(id());
    std::shared_ptr<Core::Mat::Material> scatramat =
        std::dynamic_pointer_cast<Core::Mat::Material>(scatraele->material());
    params.set<std::shared_ptr<Core::Mat::Material>>("scatramat", scatramat);
  }
  Core::LinAlg::Tensor<double, 2> xrefe{};
  for (int i = 0; i < numnode; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(0) += x[0] / numnode;
    xrefe(1) += x[1] / numnode;
  }
  params.set("elecenter_coords_ref", xrefe);
}
/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
int Discret::Elements::Wall1Scatra::my_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  return 0;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
int Discret::Elements::Wall1Scatra::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  set_params_interface_ptr(params);

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())
  {
    act = params_interface().get_action_type();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none") FOUR_C_THROW("No action supplied");
  }

  // what should the element do
  switch (act)
  {
      //==================================================================================
      // coupling terms in force-vector and stiffness matrix
      //  case Wall1_Scatra::calc_struct_multidofsetcoupling:
      //  {
      //
      //    MyEvaluate(params,
      //                      discretization,
      //                      la,
      //                      elemat1,
      //                      elemat2,
      //                      elevec1,
      //                      elevec2,
      //                      elevec3);
      //  }
      //  break;
      //  case Wall1_Scatra::postprocess_stress:
      //  {
      //    Wall1::evaluate(params,
      //                          discretization,
      //                          la[0].lm_,
      //                          elemat1,
      //                          elemat2,
      //                          elevec1,
      //                          elevec2,
      //                          elevec3);
      //  }
      //  break;
    /*case Wall1_Scatra::calc_struct_update_istep:
    {
      So3Ele::evaluate(params,
                        discretization,
                        la[0].lm_,
                        elemat1,
                        elemat2,
                        elevec1,
                        elevec2,
                        elevec3);
    }
    break;*/
    //==================================================================================
    default:
    {
      // in some cases we need to write/change some data before evaluating

      pre_evaluate(params, discretization, la);

      Wall1::evaluate(
          params, discretization, la[0].lm_, elemat1, elemat2, elevec1, elevec2, elevec3);

      my_evaluate(params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);

      break;
    }
  }  // action

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
