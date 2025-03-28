// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_anisotropy_extension_default.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_mat_anisotropy_extension.hpp"
#include "4C_mat_service.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

template <unsigned int numfib>
Mat::DefaultAnisotropyExtension<numfib>::DefaultAnisotropyExtension(const int init_mode,
    const double gamma, const bool adapt_angle,
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& stucturalTensorStrategy,
    std::array<int, numfib> fiber_ids)
    : FiberAnisotropyExtension<numfib>(stucturalTensorStrategy),
      init_mode_(init_mode),
      gamma_(gamma),
      adapt_angle_(adapt_angle),
      fiber_ids_(fiber_ids)
{
  if (init_mode_ == INIT_MODE_NODAL_FIBERS || init_mode_ == INIT_MODE_NODAL_EXTERNAL)
  {
    this->set_fiber_location(FiberLocation::GPFibers);
  }
  else
  {
    this->set_fiber_location(FiberLocation::ElementFibers);
  }
}

template <unsigned int numfib>
void Mat::DefaultAnisotropyExtension<numfib>::pack_anisotropy(
    Core::Communication::PackBuffer& data) const
{
  // Call base packing
  Mat::FiberAnisotropyExtension<numfib>::pack_anisotropy(data);

  add_to_pack(data, initialized_);
}

template <unsigned int numfib>
void Mat::DefaultAnisotropyExtension<numfib>::unpack_anisotropy(
    Core::Communication::UnpackBuffer& buffer)
{
  // Call base unpacking
  Mat::FiberAnisotropyExtension<numfib>::unpack_anisotropy(buffer);

  extract_from_pack(buffer, initialized_);
}

template <unsigned int numfib>
void Mat::DefaultAnisotropyExtension<numfib>::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Matrix<3, 3>& locsys, const Core::LinAlg::Matrix<3, 3>& defgrd)
{
  Core::LinAlg::Matrix<3, 1> ca1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ca2(Core::LinAlg::Initialization::zero);

  // Fiber direction derived from local cosy
  if (init_mode_ == INIT_MODE_ELEMENT_EXTERNAL || init_mode_ == INIT_MODE_ELEMENT_FIBERS)
  {
    // alignment angles gamma_i are read from first entry of then unnecessary vectors a1 and a2
    if ((gamma_ < -90) || (gamma_ > 90)) FOUR_C_THROW("Fiber angle not in [-90,90]");
    // convert
    double gamma = (gamma_ * M_PI) / 180.;

    if (adapt_angle_ && newgamma != -1.0)
    {
      if (gamma * newgamma < 0.0)
      {
        gamma = -1.0 * newgamma;
      }
      else
      {
        gamma = newgamma;
      }
    }

    for (int i = 0; i < 3; ++i)
    {
      // a1 = cos gamma e3 + sin gamma e2
      ca1(i) = std::cos(gamma) * locsys(i, 2) + std::sin(gamma) * locsys(i, 1);
      // a2 = cos gamma e3 - sin gamma e2
      ca2(i) = std::cos(gamma) * locsys(i, 2) - std::sin(gamma) * locsys(i, 1);
    }
  }
  else
  {
    FOUR_C_THROW(
        "Setting the fiber vectors is only possible for external element fibers mode or using a "
        "coordinate system.");
  }

  // pull back in reference configuration
  Core::LinAlg::Matrix<3, 1> a1_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> a2_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> idefgrd(Core::LinAlg::Initialization::zero);
  idefgrd.invert(defgrd);


  std::array<Core::LinAlg::Matrix<3, 1>, numfib> fibers;

  if (numfib >= 1)
  {
    fibers[0].multiply(idefgrd, ca1);
    fibers[0].scale(1.0 / fibers[0].norm2());
  }
  if (numfib >= 2)
  {
    fibers[1].multiply(idefgrd, ca2);
    fibers[1].scale(1.0 / fibers[1].norm2());
  }
  if (numfib >= 3)
  {
    FOUR_C_THROW(
        "This kind of initialization method is not implemented for materials that need more than 2 "
        "fibers.");
  }

  this->set_fibers(BaseAnisotropyExtension::GPDEFAULT, fibers);
}

template <unsigned int numfib>
void Mat::DefaultAnisotropyExtension<numfib>::set_fiber_vecs(
    const Core::LinAlg::Matrix<3, 1>& fibervec)
{
  std::array<Core::LinAlg::Matrix<3, 1>, numfib> fibers;
  fibers[0].update(fibervec);

  if (numfib >= 2)
  {
    FOUR_C_THROW("This method can only be called for materials with one fiber!");
  }

  this->set_fibers(BaseAnisotropyExtension::GPDEFAULT, fibers);
}

template <unsigned int numfib>
bool Mat::DefaultAnisotropyExtension<numfib>::do_element_fiber_initialization()
{
  switch (init_mode_)
  {
    case INIT_MODE_ELEMENT_EXTERNAL:
      do_external_fiber_initialization();
      return true;
    case INIT_MODE_ELEMENT_FIBERS:

      // check, whether a coordinate system is given
      if (this->get_anisotropy()->has_element_cylinder_coordinate_system())
      {
        // initialize fiber vector with local coordinate system
        Core::LinAlg::Matrix<3, 3> locsys(Core::LinAlg::Initialization::zero);
        const Core::LinAlg::Matrix<3, 3> Id = Core::LinAlg::identity_matrix<3>();
        this->get_anisotropy()
            ->get_element_cylinder_coordinate_system()
            .evaluate_local_coordinate_system(locsys);

        this->set_fiber_vecs(-1.0, locsys, Id);
      }
      else if (this->get_anisotropy()->get_number_of_element_fibers() > 0)
      {
        // initialize fibers from global given fibers
        std::array<Core::LinAlg::Matrix<3, 1>, numfib> fibers;
        for (unsigned int i = 0; i < numfib; ++i)
        {
          fibers.at(i) = this->get_anisotropy()->get_element_fibers().at(fiber_ids_.at(i));
        }
        this->set_fibers(BaseAnisotropyExtension::GPDEFAULT, fibers);
      }
      else
      {
        FOUR_C_THROW("Could not find element coordinate system or element fibers!");
      }

      return true;
    default:
      return false;
  }
}

template <unsigned int numfib>
bool Mat::DefaultAnisotropyExtension<numfib>::do_gp_fiber_initialization()
{
  switch (init_mode_)
  {
    case INIT_MODE_NODAL_EXTERNAL:
      do_external_fiber_initialization();
      return true;
    case INIT_MODE_NODAL_FIBERS:

      // check, whether a coordinate system is given
      if (this->get_anisotropy()->has_gp_cylinder_coordinate_system())
      {
        FOUR_C_THROW(
            "Gauss-point fibers defined via Gauss-point cylinder coordinate systems is not yet "
            "defined");
      }
      else if (this->get_anisotropy()->get_number_of_gauss_point_fibers() > 0)
      {
        // initialize fibers from global given fibers
        int gp = 0;
        for (const auto& fiberList : this->get_anisotropy()->get_gauss_point_fibers())
        {
          std::array<Core::LinAlg::Matrix<3, 1>, numfib> fibers;

          int i = 0;
          for (int id : fiber_ids_)
          {
            fibers.at(i) = fiberList.at(id);
            ++i;
          }
          this->set_fibers(gp, fibers);
          ++gp;
        }
      }
      else
      {
        FOUR_C_THROW("Could not find Gauss-point coordinate systems or Gauss-point fibers!");
      }

      return true;
    default:
      return false;
  }
}

template <unsigned int numfib>
void Mat::DefaultAnisotropyExtension<numfib>::do_external_fiber_initialization()
{
  const Core::LinAlg::Matrix<3, 3> Id = Core::LinAlg::identity_matrix<3>();
  set_fiber_vecs(-1.0, Id, Id);
}


// explicit instantiations of template classes
template class Mat::DefaultAnisotropyExtension<1u>;
template class Mat::DefaultAnisotropyExtension<2u>;
FOUR_C_NAMESPACE_CLOSE
