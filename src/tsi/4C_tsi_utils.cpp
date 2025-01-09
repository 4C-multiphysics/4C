// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_utils.hpp"

#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_element_center.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_solid_scatra_3D_ele.hpp"
#include "4C_thermo_element.hpp"

#include <Epetra_MpiComm.h>


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | remove flag thermo from condition                         dano 12/11 |
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> TSI::Utils::ThermoStructureCloneStrategy::conditions_to_copy()
    const
{
  return {{"ThermoDirichlet", "Dirichlet"}, {"ThermoPointNeumann", "PointNeumann"},
      {"ThermoLineNeumann", "LineNeumann"}, {"ThermoSurfaceNeumann", "SurfaceNeumann"},
      {"ThermoVolumeNeumann", "VolumeNeumann"}, {"ThermoConvections", "ThermoConvections"},
      {"LinePeriodic", "LinePeriodic"}, {"SurfacePeriodic", "SurfacePeriodic"},
      {"ThermoInitfield", "Initfield"}, {"MortarMulti", "MortarMulti"}};
}


/*----------------------------------------------------------------------*
 | check material of cloned element                          dano 12/11 |
 *----------------------------------------------------------------------*/
void TSI::Utils::ThermoStructureCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  //  Core::Materials::MaterialType mtype =
  //  Global::Problem::instance()->Materials()->ParameterById(matid)->Type(); if ((mtype !=
  //  Core::Materials::m_thermo_fourier))
  //    FOUR_C_THROW("Material with ID %d is not admissible for thermo elements",matid);

}  // check_material_type()


/*----------------------------------------------------------------------*
 | set element data for cloned element                                  |
 *----------------------------------------------------------------------*/
void TSI::Utils::ThermoStructureCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbs)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // initialise kinematic type to geo_linear.
  // kintype is passed to the cloned thermo element
  Inpar::Solid::KinemType kintype = Inpar::Solid::KinemType::linear;
  // if oldele is a so3_base element or a so3_Plast element
  if (const auto* const so_base = dynamic_cast<Discret::Elements::SoBase*>(oldele))
  {
    kintype = so_base->kinematic_type();
  }
  else if (const auto* const solid_ele = dynamic_cast<Discret::Elements::SolidScatra*>(oldele))
  {
    kintype = solid_ele->get_solid_element_properties().kintype;
  }
  else
  {
    FOUR_C_THROW("Unsupported solid element type!");
  }

  // note: set_material() was reimplemented by the thermo element!

  std::shared_ptr<Thermo::Element> therm = std::dynamic_pointer_cast<Thermo::Element>(newele);
  if (therm != nullptr)
  {
    // cloning to same material id -> use the same material instance
    if (oldele->material()->parameter()->id() == matid)
      therm->set_material(0, oldele->material());
    else
      therm->set_material(0, Mat::factory(matid));
    therm->set_dis_type(oldele->shape());  // set distype as well!
    therm->set_kinematic_type(kintype);    // set kintype in cloned thermal element
  }
  else
  {
    FOUR_C_THROW(
        "unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*newele).c_str());
  }
  return;
}  // set_element_data()


/*----------------------------------------------------------------------*
 | cloned element has to be a THERMO element                 dano 12/11 |
 *----------------------------------------------------------------------*/
bool TSI::Utils::ThermoStructureCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // we only support thermo elements here
  eletype.push_back("THERMO");

  return true;  // yes, we copy EVERY element (no submeshes)
}  // determine_ele_type()


/*----------------------------------------------------------------------*
 | setup TSI                                                 dano 12/11 |
 *----------------------------------------------------------------------*/
void TSI::Utils::setup_tsi(MPI_Comm comm)
{
  // access the structure discretization, make sure it is filled
  std::shared_ptr<Core::FE::Discretization> structdis;
  structdis = Global::Problem::instance()->get_dis("structure");
  // set degrees of freedom in the discretization
  if (!structdis->filled() or !structdis->have_dofs())
  {
    structdis->fill_complete();
    Epetra_Map nc = *(structdis->node_col_map());
    Epetra_Map nr = *(structdis->node_row_map());
    structdis->redistribute(nr, nc);
  }

  // access the thermo discretization
  std::shared_ptr<Core::FE::Discretization> thermdis;
  thermdis = Global::Problem::instance()->get_dis("thermo");
  if (!thermdis->filled()) thermdis->fill_complete();

  // access the problem-specific parameter list
  const Teuchos::ParameterList& tsidyn = Global::Problem::instance()->tsi_dynamic_params();

  bool matchinggrid = tsidyn.get<bool>("MATCHINGGRID");

  // we use the structure discretization as layout for the temperature discretization
  if (structdis->num_global_nodes() == 0) FOUR_C_THROW("Structure discretization is empty!");

  // create thermo elements if the temperature discretization is empty
  if (thermdis->num_global_nodes() == 0)
  {
    if (!matchinggrid)
      FOUR_C_THROW(
          "MATCHINGGRID is set to 'no' in TSI DYNAMIC section, but thermo discretization is "
          "empty!");

    Core::FE::clone_discretization<TSI::Utils::ThermoStructureCloneStrategy>(
        *structdis, *thermdis, Global::Problem::instance()->cloning_material_map());
    thermdis->fill_complete();

    // connect degrees of freedom for periodic boundary conditions
    {
      Core::Conditions::PeriodicBoundaryConditions pbc_struct(structdis);

      if (pbc_struct.has_pbc())
      {
        pbc_struct.update_dofs_for_periodic_boundary_conditions();
      }
    }

    // connect degrees of freedom for periodic boundary conditions
    {
      Core::Conditions::PeriodicBoundaryConditions pbc(thermdis);

      if (pbc.has_pbc())
      {
        pbc.update_dofs_for_periodic_boundary_conditions();
      }
    }

    // TSI must know the other discretization
    // build a proxy of the structure discretization for the temperature field
    std::shared_ptr<Core::DOFSets::DofSetInterface> structdofset = structdis->get_dof_set_proxy();
    // build a proxy of the temperature discretization for the structure field
    std::shared_ptr<Core::DOFSets::DofSetInterface> thermodofset = thermdis->get_dof_set_proxy();

    // check if ThermoField has 2 discretizations, so that coupling is possible
    if (thermdis->add_dof_set(structdofset) != 1)
      FOUR_C_THROW("unexpected dof sets in thermo field");
    if (structdis->add_dof_set(thermodofset) != 1)
      FOUR_C_THROW("unexpected dof sets in structure field");

    structdis->fill_complete(true, true, true);
    thermdis->fill_complete(true, true, true);

    TSI::Utils::set_material_pointers_matching_grid(*structdis, *thermdis);
  }
  else
  {
    if (matchinggrid)
      FOUR_C_THROW(
          "MATCHINGGRID is set to 'yes' in TSI DYNAMIC section, but thermo discretization is not "
          "empty!");

    // first call fill_complete for single discretizations.
    // This way the physical dofs are numbered successively
    structdis->fill_complete();
    thermdis->fill_complete();

    // build auxiliary dofsets, i.e. pseudo dofs on each discretization
    const int ndofpernode_thermo = 1;
    const int ndofperelement_thermo = 0;
    const int ndofpernode_struct = Global::Problem::instance()->n_dim();
    const int ndofperelement_struct = 0;
    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
    dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
        ndofpernode_thermo, ndofperelement_thermo, 0, true);
    if (structdis->add_dof_set(dofsetaux) != 1)
      FOUR_C_THROW("unexpected dof sets in structure field");
    dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
        ndofpernode_struct, ndofperelement_struct, 0, true);
    if (thermdis->add_dof_set(dofsetaux) != 1) FOUR_C_THROW("unexpected dof sets in thermo field");

    // call assign_degrees_of_freedom also for auxiliary dofsets
    // note: the order of fill_complete() calls determines the gid numbering!
    // 1. structure dofs
    // 2. thermo dofs
    // 3. structure auxiliary dofs
    // 4. thermo auxiliary dofs
    structdis->fill_complete(true, false, false);
    thermdis->fill_complete(true, false, false);
  }

}  // setup_tsi()


/*----------------------------------------------------------------------*
 | print TSI-logo                                            dano 03/10 |
 *----------------------------------------------------------------------*/
void TSI::Utils::set_material_pointers_matching_grid(
    const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis)
{
  const int numelements = targetdis.num_my_col_elements();

  for (int i = 0; i < numelements; ++i)
  {
    Core::Elements::Element* targetele = targetdis.l_col_element(i);
    const int gid = targetele->id();

    Core::Elements::Element* sourceele = sourcedis.g_element(gid);

    // for coupling we add the source material to the target element and vice versa
    targetele->add_material(sourceele->material());
    sourceele->add_material(targetele->material());
  }
}

/*----------------------------------------------------------------------*
 |  assign material to discretization A                       vuong 09/14|
 *----------------------------------------------------------------------*/
void TSI::Utils::TSIMaterialStrategy::assign_material2_to1(
    const Coupling::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele1,
    const std::vector<int>& ids_2, std::shared_ptr<Core::FE::Discretization> dis1,
    std::shared_ptr<Core::FE::Discretization> dis2)
{
  // call default assignment
  Coupling::VolMortar::Utils::DefaultMaterialStrategy::assign_material2_to1(
      volmortar, ele1, ids_2, dis1, dis2);

  // done
  return;
};


/*----------------------------------------------------------------------*
|  assign material to discretization B                       vuong 09/14|
 *----------------------------------------------------------------------*/
void TSI::Utils::TSIMaterialStrategy::assign_material1_to2(
    const Coupling::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele2,
    const std::vector<int>& ids_1, std::shared_ptr<Core::FE::Discretization> dis1,
    std::shared_ptr<Core::FE::Discretization> dis2)
{
  // if no corresponding element found -> leave
  if (ids_1.empty()) return;

  // call default assignment
  Coupling::VolMortar::Utils::DefaultMaterialStrategy::assign_material1_to2(
      volmortar, ele2, ids_1, dis1, dis2);

  // initialise kinematic type to geo_linear.
  // kintype is passed to the corresponding thermo element
  Inpar::Solid::KinemType kintype = Inpar::Solid::KinemType::linear;

  // default strategy: take material of element with closest center in reference coordinates
  Core::Elements::Element* ele1 = nullptr;
  double mindistance = 1e10;
  {
    std::vector<double> centercoords2 = Core::FE::element_center_refe_coords(*ele2);

    for (unsigned i = 0; i < ids_1.size(); ++i)
    {
      Core::Elements::Element* actele1 = dis1->g_element(ids_1[i]);
      std::vector<double> centercoords1 = Core::FE::element_center_refe_coords(*actele1);

      Core::LinAlg::Matrix<3, 1> diffcoords(true);

      for (int j = 0; j < 3; ++j) diffcoords(j, 0) = centercoords1[j] - centercoords2[j];

      if (diffcoords.norm2() - mindistance < 1e-16)
      {
        mindistance = diffcoords.norm2();
        ele1 = actele1;
      }
    }
  }

  // if Aele is a so3_base element
  if (const auto* const so_base = dynamic_cast<Discret::Elements::SoBase*>(ele1))
  {
    kintype = so_base->kinematic_type();
  }
  else if (const auto* const so_base = dynamic_cast<Discret::Elements::SolidScatra*>(ele1))
  {
    kintype = so_base->get_solid_element_properties().kintype;
  }
  else
  {
    FOUR_C_THROW("Unsupported solid element type!");
  }

  Thermo::Element* therm = dynamic_cast<Thermo::Element*>(ele2);
  if (therm != nullptr)
  {
    therm->set_kinematic_type(kintype);  // set kintype in cloned thermal element
  }

  // done
  return;
}



/*----------------------------------------------------------------------*
 | print TSI-logo                                            dano 03/10 |
 *----------------------------------------------------------------------*/
void TSI::printlogo()
{
  // more at http://www.ascii-art.de under entry "rockets"
  std::cout << "Welcome to Thermo-Structure-Interaction " << std::endl;
  std::cout << "         !\n"
            << "         !\n"
            << "         ^\n"
            << "        / \\\n"
            << "       /___\\\n"
            << "      |=   =|\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      |     |\n"
            << "      | TSI |\n"
            << "      |     |\n"
            << "     /|##!##|\\\n"
            << "    / |##!##| \\\n"
            << "   /  |##!##|  \\\n"
            << "  |  / ^ | ^ \\  |\n"
            << "  | /  ( | )  \\ |\n"
            << "  |/   ( | )   \\|\n"
            << "      ((   ))\n"
            << "     ((  :  ))\n"
            << "     ((  :  ))\n"
            << "      ((   ))\n"
            << "       (( ))\n"
            << "        ( )\n"
            << "         .\n"
            << "         .\n"
            << "         .\n"
            << "\n"
            << std::endl;
}  // printlogo()


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
