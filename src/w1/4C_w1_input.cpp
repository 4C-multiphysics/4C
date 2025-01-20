// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elasthyper.hpp"
#include "4C_w1.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::Wall1::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // set discretization type
  set_dis_type(Core::FE::string_to_cell_type(distype));

  thickness_ = container.get<double>("THICK");
  if (thickness_ <= 0) FOUR_C_THROW("WALL element thickness needs to be < 0");

  std::vector<int> ngp = container.get<std::vector<int>>("GP");



  if ((num_node() == 4) and ((ngp[0] < 2) or (ngp[1] < 2)))
    FOUR_C_THROW("Insufficient number of Gauss points");
  else if ((num_node() == 8) and ((ngp[0] < 3) or (ngp[1] < 3)))
    FOUR_C_THROW("Insufficient number of Gauss points");
  else if ((num_node() == 9) and ((ngp[0] < 3) or (ngp[1] < 3)))
    FOUR_C_THROW("Insufficient number of Gauss points");
  else if ((num_node() == 6) and (ngp[0] < 3))
    FOUR_C_THROW("Insufficient number of Gauss points");

  gaussrule_ = get_gaussrule(ngp.data());

  // read number of material model
  int material_id = container.get_or<int>("MAT", 0);
  set_material(0, Mat::factory(material_id));

  std::shared_ptr<Core::Mat::Material> mat = material();

  {
    const Core::FE::IntegrationPoints2D intpoints(gaussrule_);
    const int numgp = intpoints.nquad;
    solid_material()->setup(numgp, container);
  }

  std::string buffer;
  // reduced dimension assumption
  buffer = container.get<std::string>("STRESS_STRAIN");

  if (buffer == "plane_stress")
    wtype_ = plane_stress;
  else if (buffer == "plane_strain")
    wtype_ = plane_strain;
  else
    FOUR_C_THROW("Illegal strain/stress type '%s'", buffer.c_str());

  // kinematics type
  buffer = container.get<std::string>("KINEM");
  // geometrically linear
  if (buffer == "linear") kintype_ = Inpar::Solid::KinemType::linear;
  // geometrically non-linear with Total Lagrangean approach
  else if (buffer == "nonlinear")
    kintype_ = Inpar::Solid::KinemType::nonlinearTotLag;
  else
    FOUR_C_THROW("Illegal KINEM type '%s'", buffer.c_str());

  // EAS type
  buffer = container.get<std::string>("EAS");

  if (buffer == "none")
  {
    iseas_ = false;
  }
  else if (buffer == "full")
  {
    iseas_ = true;

    if (num_node() == 9)
      FOUR_C_THROW("eas-technology not necessary with 9 nodes");
    else if (num_node() == 8)
      FOUR_C_THROW("eas-technology not necessary with 8 nodes");
    else if (num_node() == 3)
      FOUR_C_THROW("eas-technology not implemented for tri3 elements");
    else if (num_node() == 6)
      FOUR_C_THROW("eas-technology not implemented for tri6 elements");
    else
    {
      // EAS enhanced deformation gradient parameters
      Core::LinAlg::SerialDenseMatrix alpha(
          Wall1::neas_, 1);  // if you change '4' here, then do it for alphao as well
      Core::LinAlg::SerialDenseMatrix alphao(Wall1::neas_, 1);

      // EAS portion of internal forces, also called enhancement vector s or Rtilde
      Core::LinAlg::SerialDenseVector feas(Wall1::neas_);
      // EAS matrix K_{alpha alpha}, also called Dtilde
      Core::LinAlg::SerialDenseMatrix invKaa(Wall1::neas_, Wall1::neas_);
      // EAS matrix K_{d alpha}
      Core::LinAlg::SerialDenseMatrix Kda(2 * num_node(), Wall1::neas_);
      // EAS matrix K_{alpha d} // ONLY NEEDED FOR GENERALISED ENERGY-MOMENTUM METHOD
      Core::LinAlg::SerialDenseMatrix Kad(Wall1::neas_, 2 * num_node());
      // EAS increment over last Newton step
      Core::LinAlg::SerialDenseMatrix eas_inc(Wall1::neas_, 1);

      // save EAS data into element container easdata_
      easdata_.alpha = alpha;
      easdata_.alphao = alphao;
      easdata_.feas = feas;
      easdata_.invKaa = invKaa;
      easdata_.Kda = Kda;
      easdata_.Kad = Kad;  // ONLY NEEDED FOR GENERALISED ENERGY-MOMENTUM METHOD
      easdata_.eas_inc = eas_inc;
    }
  }
  else
  {
    FOUR_C_THROW("Illegal EAS model");
  }

  // EAS type
  if (iseas_)
  {
    eastype_ = eas_q1e4;
  }
  else
  {
    eastype_ = eas_vague;
  }

  stresstype_ = w1_xy;

  // check for invalid combinations
  if (kintype_ == Inpar::Solid::KinemType::linear && iseas_ == true)
    FOUR_C_THROW("ERROR: No EAS for geometrically linear WALL element");

  // validate kinematics of solid material
  solid_material()->valid_kinematics(kintype_);

  // Validate that materials doesn't use extended update call.
  if (solid_material()->uses_extended_update())
    FOUR_C_THROW("This element currently does not support the extended update call.");

  return true;
}

/*----------------------------------------------------------------------*
 |  Get gaussrule on dependance of gausspoints                     mgit |
 *----------------------------------------------------------------------*/
Core::FE::GaussRule2D Discret::Elements::Wall1::get_gaussrule(int* ngp)
{
  Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::undefined;

  switch (shape())
  {
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      if ((ngp[0] == 2) && (ngp[1] == 2))
      {
        rule = Core::FE::GaussRule2D::quad_4point;
      }
      else if ((ngp[0] == 3) && (ngp[1] == 3))
      {
        rule = Core::FE::GaussRule2D::quad_9point;
      }
      else
        FOUR_C_THROW("Unknown number of Gauss points for quad element");
      break;
    }
    case Core::FE::CellType::nurbs4:
    case Core::FE::CellType::nurbs9:
    {
      if ((ngp[0] == 2) && (ngp[1] == 2))
      {
        rule = Core::FE::GaussRule2D::quad_4point;
      }
      else if ((ngp[0] == 3) && (ngp[1] == 3))
      {
        rule = Core::FE::GaussRule2D::quad_9point;
      }
      else if ((ngp[0] == 4) && (ngp[1] == 4))
      {
        rule = Core::FE::GaussRule2D::quad_16point;
      }
      else if ((ngp[0] == 5) && (ngp[1] == 5))
      {
        rule = Core::FE::GaussRule2D::quad_25point;
      }
      else if ((ngp[0] == 10) && (ngp[1] == 10))
      {
        rule = Core::FE::GaussRule2D::quad_100point;
      }
      else
        FOUR_C_THROW("Unknown number of Gauss points for nurbs element");
      break;
    }
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      if ((ngp[0] == 1) && (ngp[1] == 0))
      {
        rule = Core::FE::GaussRule2D::tri_1point;
      }
      else if ((ngp[0] == 3) && (ngp[1] == 0))
      {
        rule = Core::FE::GaussRule2D::tri_3point;
      }
      else if ((ngp[0] == 6) && (ngp[1] == 0))
      {
        rule = Core::FE::GaussRule2D::tri_6point;
      }
      else
        FOUR_C_THROW("Unknown number of Gauss points for tri element");
      break;
    }
    default:
      FOUR_C_THROW("Unknown distype");
      break;
  }
  return rule;
}

FOUR_C_NAMESPACE_CLOSE
