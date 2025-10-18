// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc_cardiac_monodomain.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::evaluate_action(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const ScaTra::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  //(for now) only first dof set considered
  const std::vector<int>& lm = la[0].lm_;

  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::time_update_material:
    {
      std::vector<std::shared_ptr<Mat::Myocard>> updatemat;
      updatemat.reserve(my::numscal_);

      // access the general material
      std::shared_ptr<Core::Mat::Material> material = ele->material();

      // first, determine the materials which need a time update, i.e. myocard materials
      if (material->material_type() == Core::Materials::m_matlist)
      {
        const std::shared_ptr<Mat::MatList> actmat =
            std::dynamic_pointer_cast<Mat::MatList>(material);
        if (actmat->num_mat() < my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

        for (int k = 0; k < my::numscal_; ++k)
        {
          const int matid = actmat->mat_id(k);
          std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

          if (singlemat->material_type() == Core::Materials::m_myocard)
          {
            // reference to Teuchos::rcp not possible here, since the material
            // is required to be not const for this application
            updatemat.push_back(std::dynamic_pointer_cast<Mat::Myocard>(singlemat));
          }
        }
      }

      if (material->material_type() == Core::Materials::m_myocard)
      {  // reference to Teuchos::rcp not possible here, since the material is required to be
        // not const for this application
        updatemat.push_back(std::dynamic_pointer_cast<Mat::Myocard>(material));
      }

      if (updatemat.size() > 0)  // found at least one material to be updated
      {
        // all materials in the matlist should be of the same kind
        if (updatemat.size() != (unsigned)my::numscal_)
          FOUR_C_THROW("Number of materials to be updated is not equal to number of scalars!");

        // get time-step length
        const double dt = my::scatraparatimint_->dt();

        // extract local values from the global vectors
        std::shared_ptr<const Core::LinAlg::Vector<double>> phinp =
            discretization.get_state("phinp");
        if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
        Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, lm);

        my::eval_shape_func_and_derivs_at_ele_center();

        for (unsigned i = 0; i < updatemat.size(); i++)
        {
          const double csnp = my::funct_.dot(my::ephinp_[i]);  // be careful, we assume k==i here
          updatemat[i]->update(csnp, dt);
        }
      }

      break;
    }

    case ScaTra::Action::get_material_internal_state:
    {
      // NOTE: add integral values only for elements which are NOT ghosted!
      if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
      {
        // access the general material
        std::shared_ptr<Core::Mat::Material> material = ele->material();
        std::shared_ptr<Core::LinAlg::MultiVector<double>> material_internal_state =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "material_internal_state");

        if (material->material_type() == Core::Materials::m_myocard)
        {
          std::shared_ptr<Mat::Myocard> material =
              std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
          for (int k = 0; k < material_internal_state->num_vectors(); ++k)
          {
            material_internal_state->replace_global_value(
                ele->id(), k, material->get_internal_state(k));
          }
        }
        params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
            "material_internal_state", material_internal_state);
      }

      break;
    }

    case ScaTra::Action::set_material_internal_state:
    {
      // NOTE: add integral values only for elements which are NOT ghosted!
      if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
      {
        // access the general material
        std::shared_ptr<Core::Mat::Material> material = ele->material();
        std::shared_ptr<Core::LinAlg::Vector<double>> material_internal_state_component =
            params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>(
                "material_internal_state_component");

        if (material->material_type() == Core::Materials::m_myocard)
        {
          std::shared_ptr<Mat::Myocard> material =
              std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
          int k = params.get<int>("k");
          material->set_internal_state(k, (*material_internal_state_component)[ele->id()]);
        }
      }
    }

    break;

    case ScaTra::Action::get_material_ionic_currents:
    {
      // NOTE: add integral values only for elements which are NOT ghosted!
      if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
      {
        // access the general material
        std::shared_ptr<Core::Mat::Material> material = ele->material();
        std::shared_ptr<Core::LinAlg::MultiVector<double>> material_ionic_currents =
            params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                "material_ionic_currents");

        if (material->material_type() == Core::Materials::m_myocard)
        {
          std::shared_ptr<Mat::Myocard> material =
              std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
          for (int k = 0; k < material_ionic_currents->num_vectors(); ++k)
          {
            material_ionic_currents->replace_global_value(
                ele->id(), k, material->get_ionic_currents(k));
          }
        }
        params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
            "material_ionic_currents", material_ionic_currents);
      }

      break;
    }

    default:
    {
      my::evaluate_action(
          ele, params, discretization, action, la, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }
  }  // switch(action)

  return 0;
}


// template classes
// 1D elements
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad4, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::hex8, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tet10, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::pyramid5, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
