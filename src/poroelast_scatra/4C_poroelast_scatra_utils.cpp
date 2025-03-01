// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_scatra_utils.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_poroelast_base.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_poroelast_scatra_base.hpp"
#include "4C_poroelast_scatra_monolithic.hpp"
#include "4C_poroelast_scatra_part_1wc.hpp"
#include "4C_poroelast_scatra_part_2wc.hpp"
#include "4C_poroelast_scatra_utils_clonestrategy.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_so3_poro_p1_scatra_eletypes.hpp"
#include "4C_so3_poro_scatra_eletypes.hpp"
#include "4C_solid_poro_3D_ele_pressure_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based_p1.hpp"
#include "4C_w1_poro_p1_scatra_eletypes.hpp"
#include "4C_w1_poro_scatra_eletypes.hpp"

#include <Epetra_Time.h>

FOUR_C_NAMESPACE_OPEN


bool PoroElastScaTra::Utils::is_poro_scatra_element(const Core::Elements::Element* actele)
{
  // checks if element is a poro scatra element (new elements need to be listed here)
  return actele->element_type() == Discret::Elements::SoHex8PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::SolidPoroPressureBasedType::instance() or
         actele->element_type() ==
             Discret::Elements::SolidPoroPressureVelocityBasedType::instance() or
         actele->element_type() == Discret::Elements::SoTet4PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::SoTet10PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::SoHex27PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::SoNurbs27PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::WallTri3PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::WallQuad4PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::WallQuad9PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::WallNurbs4PoroScatraType::instance() or
         actele->element_type() == Discret::Elements::WallNurbs9PoroScatraType::instance() or
         is_poro_p1_scatra_element(actele);
}

bool PoroElastScaTra::Utils::is_poro_p1_scatra_element(const Core::Elements::Element* actele)
{
  // checks if element is a porop1 scatra element (new elements need to be listed here)
  return actele->element_type() ==
             Discret::Elements::SolidPoroPressureVelocityBasedP1Type::instance() or
         actele->element_type() == Discret::Elements::SoHex8PoroP1ScatraType::instance() or
         actele->element_type() == Discret::Elements::SoTet4PoroP1ScatraType::instance() or
         actele->element_type() == Discret::Elements::WallQuad4PoroP1ScatraType::instance() or
         actele->element_type() == Discret::Elements::WallTri3PoroP1ScatraType::instance() or
         actele->element_type() == Discret::Elements::WallQuad9PoroP1ScatraType::instance();
}

std::shared_ptr<PoroElastScaTra::PoroScatraBase>
PoroElastScaTra::Utils::create_poro_scatra_algorithm(
    const Teuchos::ParameterList& timeparams, MPI_Comm comm)
{
  Global::Problem* problem = Global::Problem::instance();

  // create an empty PoroScatraBase instance
  std::shared_ptr<PoroElastScaTra::PoroScatraBase> algo = nullptr;

  // Parameter reading
  const Teuchos::ParameterList& params = problem->poro_scatra_control_params();
  const auto coupling =
      Teuchos::getIntegralValue<Inpar::PoroScaTra::SolutionSchemeOverFields>(params, "COUPALGO");

  switch (coupling)
  {
    case Inpar::PoroScaTra::Monolithic:
    {
      algo = std::make_shared<PoroElastScaTra::PoroScatraMono>(comm, timeparams);
      break;
    }
    case Inpar::PoroScaTra::Part_ScatraToPoro:
    {
      algo = std::make_shared<PoroElastScaTra::PoroScatraPart1WCScatraToPoro>(comm, timeparams);
      break;
    }
    case Inpar::PoroScaTra::Part_PoroToScatra:
    {
      algo = std::make_shared<PoroElastScaTra::PoroScatraPart1WCPoroToScatra>(comm, timeparams);
      break;
    }
    case Inpar::PoroScaTra::Part_TwoWay:
    {
      algo = std::make_shared<PoroElastScaTra::PoroScatraPart2WC>(comm, timeparams);
      break;
    }
    default:
      break;
  }

  if (!algo)
  {
    FOUR_C_THROW("Creation of the Poroelast Scatra Algorithm failed.");
  }

  // setup solver (if needed)
  algo->setup_solver();

  return algo;
}

std::shared_ptr<Core::LinAlg::MapExtractor> PoroElastScaTra::Utils::build_poro_scatra_splitter(
    Core::FE::Discretization& dis)
{
  std::shared_ptr<Core::LinAlg::MapExtractor> porositysplitter = nullptr;

  // Loop through all elements on processor
  int locporop1 = std::count_if(
      dis.my_col_element_range().begin(), dis.my_col_element_range().end(), is_poro_scatra_element);

  // Was at least one PoroP1 found on one processor?
  int glonumporop1 = 0;
  Core::Communication::max_all(&locporop1, &glonumporop1, 1, dis.get_comm());
  // Yes, it was. Go ahead for all processors (even if they do not carry any PoroP1 elements)
  if (glonumporop1 > 0)
  {
    porositysplitter = std::make_shared<Core::LinAlg::MapExtractor>();
    const int ndim = Global::Problem::instance()->n_dim();
    Core::LinAlg::create_map_extractor_from_discretization(dis, ndim, *porositysplitter);
  }

  return porositysplitter;
}

void PoroElastScaTra::Utils::create_volume_ghosting(Core::FE::Discretization& idiscret)
{
  // We get the discretizations from the global problem, as the contact does not have
  // both structural and porofluid discretization, but we should guarantee consistent ghosting!

  Global::Problem* problem = Global::Problem::instance();

  std::vector<std::shared_ptr<Core::FE::Discretization>> voldis;
  voldis.push_back(problem->get_dis("structure"));
  voldis.push_back(problem->get_dis("porofluid"));
  voldis.push_back(problem->get_dis("scatra"));

  const Epetra_Map* ielecolmap = idiscret.element_col_map();

  for (auto& voldi : voldis)
  {
    // 1 Ghost all Volume Element + Nodes,for all ghosted mortar elements!
    std::vector<int> rdata;

    // Fill rdata with existing colmap

    const Epetra_Map* elecolmap = voldi->element_col_map();
    const std::shared_ptr<Epetra_Map> allredelecolmap =
        Core::LinAlg::allreduce_e_map(*voldi->element_row_map());

    for (int i = 0; i < elecolmap->NumMyElements(); ++i)
    {
      int gid = elecolmap->GID(i);
      rdata.push_back(gid);
    }

    // Find elements, which are ghosted on the interface but not in the volume discretization
    for (int i = 0; i < ielecolmap->NumMyElements(); ++i)
    {
      int gid = ielecolmap->GID(i);

      Core::Elements::Element* ele = idiscret.g_element(gid);
      if (!ele) FOUR_C_THROW("ERROR: Cannot find element with gid %", gid);
      auto* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele);

      int volgid = 0;
      if (!faceele)
        FOUR_C_THROW("Cast to FaceElement failed!");
      else
        volgid = faceele->parent_element_id();

      // Ghost the parent element additionally
      if (elecolmap->LID(volgid) == -1 &&
          allredelecolmap->LID(volgid) !=
              -1)  // Volume discretization has not Element on this proc but on another
        rdata.push_back(volgid);
    }

    // re-build element column map
    Epetra_Map newelecolmap(-1, static_cast<int>(rdata.size()), rdata.data(), 0,
        Core::Communication::as_epetra_comm(voldi->get_comm()));
    rdata.clear();

    // redistribute the volume discretization according to the
    // new (=old) element column layout & and ghost also nodes!
    voldi->extended_ghosting(newelecolmap, true, true, true, false);  // no check!!!
  }

  // 2 Material pointers need to be reset after redistribution.
  PoroElast::Utils::set_material_pointers_matching_grid(*voldis[0], *voldis[1]);
  PoroElast::Utils::set_material_pointers_matching_grid(*voldis[0], *voldis[2]);
  PoroElast::Utils::set_material_pointers_matching_grid(*voldis[1], *voldis[2]);

  // 3 Reconnect Face Element -- Porostructural Parent Element Pointers!
  PoroElast::Utils::reconnect_parent_pointers(idiscret, *voldis[0], &(*voldis[1]));

  // 4 In case we use
  std::shared_ptr<Core::FE::DiscretizationFaces> facediscret =
      std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(voldis[1]);
  if (facediscret != nullptr) facediscret->fill_complete_faces(true, true, true, true);
}

FOUR_C_NAMESPACE_CLOSE
