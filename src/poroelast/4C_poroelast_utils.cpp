// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_utils.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_general_element_center.hpp"
#include "4C_fluid_ele_poro.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_poroelast_base.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_poroelast_monolithicfluidsplit.hpp"
#include "4C_poroelast_monolithicmeshtying.hpp"
#include "4C_poroelast_monolithicsplit_nopenetration.hpp"
#include "4C_poroelast_monolithicstructuresplit.hpp"
#include "4C_poroelast_partitioned.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"
#include "4C_so3_poro_eletypes.hpp"
#include "4C_so3_poro_p1_eletypes.hpp"
#include "4C_solid_poro_3D_ele_pressure_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based_p1.hpp"
#include "4C_w1_poro_eletypes.hpp"
#include "4C_w1_poro_p1_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

bool PoroElast::Utils::is_poro_element(const Core::Elements::Element* actele)
{
  // all poro elements need to be listed here
  return actele->element_type() == Discret::Elements::SoHex8PoroType::instance() or
         actele->element_type() == Discret::Elements::SolidPoroPressureBasedType::instance() or
         actele->element_type() ==
             Discret::Elements::SolidPoroPressureVelocityBasedType::instance() or
         actele->element_type() == Discret::Elements::SoTet4PoroType::instance() or
         actele->element_type() == Discret::Elements::SoTet10PoroType::instance() or
         actele->element_type() == Discret::Elements::SoHex27PoroType::instance() or
         actele->element_type() == Discret::Elements::SoNurbs27PoroType::instance() or
         actele->element_type() == Discret::Elements::WallTri3PoroType::instance() or
         actele->element_type() == Discret::Elements::WallQuad4PoroType::instance() or
         actele->element_type() == Discret::Elements::WallQuad9PoroType::instance() or
         actele->element_type() == Discret::Elements::WallNurbs4PoroType::instance() or
         actele->element_type() == Discret::Elements::WallNurbs9PoroType::instance() or
         is_poro_p1_element(actele);
}

bool PoroElast::Utils::is_poro_p1_element(const Core::Elements::Element* actele)
{
  // all poro-p1 elements need to be listed here
  return actele->element_type() ==
             Discret::Elements::SolidPoroPressureVelocityBasedP1Type::instance() or
         actele->element_type() == Discret::Elements::SoHex8PoroP1Type::instance() or
         actele->element_type() == Discret::Elements::SoTet4PoroP1Type::instance() or
         actele->element_type() == Discret::Elements::WallQuad4PoroP1Type::instance() or
         actele->element_type() == Discret::Elements::WallTri3PoroP1Type::instance() or
         actele->element_type() == Discret::Elements::WallQuad9PoroP1Type::instance();
}

std::shared_ptr<PoroElast::PoroBase> PoroElast::Utils::create_poro_algorithm(
    const Teuchos::ParameterList& timeparams, MPI_Comm comm, bool setup_solver,
    std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter)
{
  Global::Problem* problem = Global::Problem::instance();

  // access the problem-specific parameter list
  const Teuchos::ParameterList& poroelastdyn = problem->poroelast_dynamic_params();

  //  problem->mortar_coupling_params()
  const auto coupling = Teuchos::getIntegralValue<Inpar::PoroElast::SolutionSchemeOverFields>(
      poroelastdyn, "COUPALGO");

  // create an empty Poroelast::Algorithm instance
  std::shared_ptr<PoroElast::PoroBase> poroalgo = nullptr;

  switch (coupling)
  {
    case Inpar::PoroElast::Monolithic:
    {
      // create an PoroElast::Monolithic instance
      poroalgo = std::make_shared<PoroElast::Monolithic>(comm, timeparams, porosity_splitter);
      break;
    }  // monolithic case
    case Inpar::PoroElast::Monolithic_structuresplit:
    {
      // create an PoroElast::MonolithicStructureSplit instance
      poroalgo = std::make_shared<PoroElast::MonolithicStructureSplit>(
          comm, timeparams, porosity_splitter);
      break;
    }
    case Inpar::PoroElast::Monolithic_fluidsplit:
    {
      // create an PoroElast::MonolithicFluidSplit instance
      poroalgo =
          std::make_shared<PoroElast::MonolithicFluidSplit>(comm, timeparams, porosity_splitter);
      break;
    }
    case Inpar::PoroElast::Monolithic_nopenetrationsplit:
    {
      // create an PoroElast::MonolithicSplitNoPenetration instance
      poroalgo = std::make_shared<PoroElast::MonolithicSplitNoPenetration>(
          comm, timeparams, porosity_splitter);
      break;
    }
    case Inpar::PoroElast::Partitioned:
    {
      // create an PoroElast::Partitioned instance
      poroalgo = std::make_shared<PoroElast::Partitioned>(comm, timeparams, porosity_splitter);
      break;
    }
    case Inpar::PoroElast::Monolithic_meshtying:
    {
      // create an PoroElast::MonolithicMeshtying instance
      poroalgo =
          std::make_shared<PoroElast::MonolithicMeshtying>(comm, timeparams, porosity_splitter);
      break;
    }
    default:
      FOUR_C_THROW("Unknown solutiontype for poroelasticity: %d", coupling);
      break;
  }

  // setup solver (if needed)
  if (setup_solver) poroalgo->setup_solver();

  return poroalgo;
}


std::shared_ptr<Core::LinAlg::MapExtractor> PoroElast::Utils::build_poro_splitter(
    Core::FE::Discretization& dis)
{
  std::shared_ptr<Core::LinAlg::MapExtractor> porositysplitter = nullptr;

  // Loop through all elements on processor
  int locporop1 = std::count_if(
      dis.my_col_element_range().begin(), dis.my_col_element_range().end(), is_poro_p1_element);

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

void PoroElast::Utils::set_material_pointers_matching_grid(
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

void PoroElast::Utils::create_volume_ghosting(Core::FE::Discretization& idiscret)
{
  //**********************************************************************
  // Prerequisites of this function:
  // All Contact Elements need a set parent_id_ (member of faceelement!) before
  // calling CreateInterfaceGhosting as this id will be communicated to all
  // processors! Otherwise any information which connects face and volume
  // element is lost! (Parent Element Pointer is not communicated)
  //**********************************************************************

  // We get the discretizations from the global problem, as the contact does not have
  // both structural and porofluid discretization, but we should guarantee consistent ghosting!

  Global::Problem* problem = Global::Problem::instance();

  std::vector<std::shared_ptr<Core::FE::Discretization>> voldis;
  voldis.push_back(problem->get_dis("structure"));
  voldis.push_back(problem->get_dis("porofluid"));

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

  // 3 Reconnect Face Element -- Porostructural Parent Element Pointers!
  PoroElast::Utils::reconnect_parent_pointers(idiscret, *voldis[0], &(*voldis[1]));

  // 4 In case we use
  std::shared_ptr<Core::FE::DiscretizationFaces> facediscret =
      std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(voldis[1]);
  if (facediscret != nullptr) facediscret->fill_complete_faces(true, true, true, true);
}

void PoroElast::Utils::reconnect_parent_pointers(Core::FE::Discretization& idiscret,
    Core::FE::Discretization& voldiscret, Core::FE::Discretization* voldiscret2)
{
  const Epetra_Map* ielecolmap = idiscret.element_col_map();
  const Epetra_Map* elecolmap = voldiscret.element_col_map();

  for (int i = 0; i < ielecolmap->NumMyElements(); ++i)
  {
    int gid = ielecolmap->GID(i);

    Core::Elements::Element* ele = idiscret.g_element(gid);
    if (!ele) FOUR_C_THROW("ERROR: Cannot find element with gid %", gid);

    auto* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele);

    if (!faceele) FOUR_C_THROW("Cast to FaceElement failed!");
    set_slave_and_master(voldiscret, voldiscret2, elecolmap, faceele);
  }
}

void PoroElast::Utils::set_slave_and_master(const Core::FE::Discretization& voldiscret,
    const Core::FE::Discretization* voldiscret2, const Epetra_Map* elecolmap,
    Core::Elements::FaceElement* faceele)
{
  int volgid = faceele->parent_element_id();

  if (elecolmap->LID(volgid) == -1)  // Volume discretization has not Element
    FOUR_C_THROW("create_volume_ghosting: Element %d does not exist on this Proc!", volgid);

  Core::Elements::Element* vele = voldiscret.g_element(volgid);
  if (!vele) FOUR_C_THROW("ERROR: Cannot find element with gid %", volgid);
  faceele->set_parent_master_element(vele, faceele->face_parent_number());

  if (voldiscret2)
  {
    const Epetra_Map* elecolmap2 = voldiscret2->element_col_map();
    if (elecolmap2->LID(volgid) == -1)  // Volume discretization has not Element
      faceele->set_parent_slave_element(nullptr, -1);
    else
    {
      vele = voldiscret2->g_element(volgid);
      if (!vele) FOUR_C_THROW("ERROR: Cannot find element with gid %", volgid);
      faceele->set_parent_slave_element(vele, faceele->face_parent_number());
    }
  }
}

void PoroElast::print_logo()
{
  std::cout << "This is a Porous Media problem" << std::endl;
  std::cout << "       .--..--..--..--..--..--. " << std::endl;
  std::cout << "      .'  \\  (`._   (_)     _   \\ " << std::endl;
  std::cout << "     .'    |  '._)         (_)  | " << std::endl;
  std::cout << "     \\ _.')\\      .----..---.   / " << std::endl;
  std::cout << "     |(_.'  |    /    .-\\-.  \\  | " << std::endl;
  std::cout << "     \\     0|    |   ( O| O) | o| " << std::endl;
  std::cout << "      |  _  |  .--.____.'._.-.  | " << std::endl;
  std::cout << "      \\ (_) | o         -` .-`  | " << std::endl;
  std::cout << "       |    \\   |`-._ _ _ _ _\\ / " << std::endl;
  std::cout << "       \\    |   |  `. |_||_|   | " << std::endl;
  std::cout << "       | o  |    \\_      \\     |     -.   .-. " << std::endl;
  std::cout << "       |.-.  \\     `--..-'   O |     `.`-' .' " << std::endl;
  std::cout << "     _.'  .' |     `-.-'      /-.__   ' .-' " << std::endl;
  std::cout << "   .' `-.` '.|='=.='=.='=.='=|._/_ `-'.' " << std::endl;
  std::cout << "   `-._  `.  |________/\\_____|    `-.' " << std::endl;
  std::cout << "      .'   ).| '=' '='\\/ '=' | " << std::endl;
  std::cout << "      `._.`  '---------------' " << std::endl;
  std::cout << "            //___\\   //___\\ " << std::endl;
  std::cout << "              ||       || " << std::endl;
  std::cout << "              ||_.-.   ||_.-. " << std::endl;
  std::cout << "             (_.--__) (_.--__) " << std::endl;
}

double PoroElast::Utils::calculate_vector_norm(
    const enum Inpar::PoroElast::VectorNorm norm, const Core::LinAlg::Vector<double>& vect)
{
  // L1 norm
  // norm = sum_0^i vect[i]
  if (norm == Inpar::PoroElast::norm_l1)
  {
    double vectnorm;
    vect.Norm1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  // norm = sqrt{sum_0^i vect[i]^2 }
  else if (norm == Inpar::PoroElast::norm_l2)
  {
    double vectnorm;
    vect.Norm2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  // norm = sqrt{sum_0^i vect[i]^2 }/ sqrt{length_vect}
  else if (norm == Inpar::PoroElast::norm_rms)
  {
    double vectnorm;
    vect.Norm2(&vectnorm);
    return vectnorm / sqrt((double)vect.GlobalLength());
  }
  // infinity/maximum norm
  // norm = max( vect[i] )
  else if (norm == Inpar::PoroElast::norm_inf)
  {
    double vectnorm;
    vect.NormInf(&vectnorm);
    return vectnorm;
  }
  // norm = sum_0^i vect[i]/length_vect
  else if (norm == Inpar::PoroElast::norm_l1_scaled)
  {
    double vectnorm;
    vect.Norm1(&vectnorm);
    return vectnorm / ((double)vect.GlobalLength());
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }
}

void PoroElast::Utils::PoroMaterialStrategy::assign_material2_to1(
    const Coupling::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele1,
    const std::vector<int>& ids_2, std::shared_ptr<Core::FE::Discretization> dis1,
    std::shared_ptr<Core::FE::Discretization> dis2)
{
  // call default assignment
  Coupling::VolMortar::Utils::DefaultMaterialStrategy::assign_material2_to1(
      volmortar, ele1, ids_2, dis1, dis2);

  // default strategy: take material of element with closest center in reference coordinates
  Core::Elements::Element* ele2 = nullptr;
  double mindistance = 1e10;
  {
    std::vector<double> centercoords1 = Core::FE::element_center_refe_coords(*ele1);

    for (int id_2 : ids_2)
    {
      Core::Elements::Element* actele2 = dis2->g_element(id_2);
      std::vector<double> centercoords2 = Core::FE::element_center_refe_coords(*actele2);

      Core::LinAlg::Matrix<3, 1> diffcoords(true);

      for (int j = 0; j < 3; ++j) diffcoords(j, 0) = centercoords1[j] - centercoords2[j];

      if (diffcoords.norm2() - mindistance < 1e-16)
      {
        mindistance = diffcoords.norm2();
        ele2 = actele2;
      }
    }
  }

  // if Bele is a fluid element
  auto* fluid = dynamic_cast<Discret::Elements::FluidPoro*>(ele2);
  if (fluid != nullptr)
  {
    // Copy Initial Porosity from StructPoro Material to FluidPoro Material
    static_cast<Mat::PAR::FluidPoro*>(fluid->material()->parameter())
        ->set_initial_porosity(
            std::static_pointer_cast<Mat::StructPoro>(ele1->material())->init_porosity());
  }
  else
  {
    FOUR_C_THROW(
        "ERROR: Unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*ele2).c_str());
  }
}

void PoroElast::Utils::PoroMaterialStrategy::assign_material1_to2(
    const Coupling::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele2,
    const std::vector<int>& ids_1, std::shared_ptr<Core::FE::Discretization> dis1,
    std::shared_ptr<Core::FE::Discretization> dis2)
{
  // call default assignment
  Coupling::VolMortar::Utils::DefaultMaterialStrategy::assign_material1_to2(
      volmortar, ele2, ids_1, dis1, dis2);

  // if no corresponding element found -> leave
  if (ids_1.empty()) return;

  // default strategy: take material of element with closest center in reference coordinates
  Core::Elements::Element* ele1 = nullptr;
  double mindistance = 1e10;
  {
    std::vector<double> centercoords2 = Core::FE::element_center_refe_coords(*ele2);

    for (int id_1 : ids_1)
    {
      Core::Elements::Element* actele1 = dis1->g_element(id_1);
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
  auto* so_base = dynamic_cast<Discret::Elements::SoBase*>(ele1);

  // if Bele is a fluid element
  auto* fluid = dynamic_cast<Discret::Elements::FluidPoro*>(ele2);
  if (fluid != nullptr)
  {
    if (so_base)
    {
      fluid->set_kinematic_type(so_base->kinematic_type());
    }
    else
      FOUR_C_THROW("ERROR: ele1 is not a solid element");
  }
  else
  {
    FOUR_C_THROW(
        "ERROR: Unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*ele2).c_str());
  }
}

FOUR_C_NAMESPACE_CLOSE
