// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_data_read.hpp"

#include "4C_comm_utils.hpp"
#include "4C_contact_constitutivelaw_bundle.hpp"
#include "4C_contact_constitutivelaw_constitutivelaw_definition.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset_independent.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_inpar_validconditions.hpp"
#include "4C_inpar_validcontactconstitutivelaw.hpp"
#include "4C_inpar_validmaterials.hpp"
#include "4C_io.hpp"
#include "4C_io_elementreader.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_io_meshreader.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_mat_elchphase.hpp"
#include "4C_mat_materialdefinition.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_newman_multiscale.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_scatra_multiscale.hpp"
#include "4C_particle_engine_particlereader.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_xfem_discretization.hpp"
#include "4C_xfem_discretization_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

FOUR_C_NAMESPACE_OPEN

void Global::read_fields(Global::Problem& problem, Core::IO::InputFile& input, const bool read_mesh)
{
  std::shared_ptr<Core::FE::Discretization> structdis = nullptr;
  std::shared_ptr<Core::FE::Discretization> fluiddis = nullptr;
  std::shared_ptr<Core::FE::Discretization> xfluiddis = nullptr;
  std::shared_ptr<Core::FE::Discretization> aledis = nullptr;
  std::shared_ptr<Core::FE::Discretization> structaledis = nullptr;
  std::shared_ptr<Core::FE::Discretization> thermdis = nullptr;
  std::shared_ptr<Core::FE::Discretization> lubricationdis = nullptr;
  std::shared_ptr<Core::FE::Discretization> scatradis = nullptr;
  std::shared_ptr<Core::FE::Discretization> scatra_micro_dis = nullptr;
  std::shared_ptr<Core::FE::Discretization> cellscatradis = nullptr;
  std::shared_ptr<Core::FE::Discretization> fluidscatradis = nullptr;
  std::shared_ptr<Core::FE::Discretization> structscatradis = nullptr;
  std::shared_ptr<Core::FE::Discretization> artscatradis = nullptr;
  std::shared_ptr<Core::FE::Discretization> arterydis = nullptr;  //_1D_ARTERY_
  std::shared_ptr<Core::FE::Discretization> airwaydis = nullptr;
  std::shared_ptr<Core::FE::Discretization> optidis = nullptr;
  std::shared_ptr<Core::FE::Discretization> porofluiddis = nullptr;  // fpsi, poroelast
  std::shared_ptr<Core::FE::Discretization> elemagdis = nullptr;
  std::shared_ptr<Core::FE::Discretization> celldis = nullptr;
  std::shared_ptr<Core::FE::Discretization> pboxdis = nullptr;

  // decide which kind of spatial representation is required
  const Core::FE::ShapeFunctionType distype = problem.spatial_approximation_type();
  auto output_control = problem.output_control_file();

  // the basic mesh reader. now add desired node and element readers to it!
  Core::IO::MeshReader meshreader(input, "NODE COORDS",
      {.mesh_paritioning_parameters = Problem::instance()->mesh_partitioning_params(),
          .geometric_search_parameters = Problem::instance()->geometric_search_params(),
          .io_parameters = Problem::instance()->io_params()});

  MPI_Comm comm = problem.get_communicators()->local_comm();
  switch (problem.get_problem_type())
  {
    case Core::ProblemType::fsi:
    case Core::ProblemType::fsi_redmodels:
    {
      if (distype == Core::FE::ShapeFunctionType::nurbs)
      {
        structdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
            "structure", comm, problem.n_dim());
        fluiddis =
            std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("fluid", comm, problem.n_dim());
        aledis =
            std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("ale", comm, problem.n_dim());
      }
      else if (problem.fluid_dynamic_params().sublist("WALL MODEL").get<bool>("X_WALL"))
      {
        structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
        fluiddis = std::make_shared<XFEM::DiscretizationXWall>("fluid", comm, problem.n_dim());
        aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
      }
      else
      {
        structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
        fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
        if (problem.x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
          xfluiddis = std::make_shared<XFEM::DiscretizationXFEM>("xfluid", comm, problem.n_dim());
        aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      if (xfluiddis != nullptr)
        xfluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(xfluiddis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("fluid", fluiddis);
      if (xfluiddis != nullptr) problem.add_dis("xfluid", xfluiddis);
      problem.add_dis("ale", aledis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));

      if (xfluiddis != nullptr)
      {
        meshreader.add_element_reader(Core::IO::ElementReader(xfluiddis, input, "FLUID ELEMENTS"));
      }
      else
        meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));

      meshreader.add_element_reader(Core::IO::ElementReader(aledis, input, "ALE ELEMENTS"));

      break;
    }
    case Core::ProblemType::gas_fsi:
    case Core::ProblemType::thermo_fsi:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          FOUR_C_THROW("Nurbs discretization not possible for fs3i!");
          break;
        }
        default:
        {
          structdis =
              std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
          fluiddis =
              std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
          aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
          fluidscatradis =
              std::make_shared<Core::FE::Discretization>("scatra1", comm, problem.n_dim());
          structscatradis =
              std::make_shared<Core::FE::Discretization>("scatra2", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));
      fluidscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          fluidscatradis, output_control, distype));
      structscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          structscatradis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("fluid", fluiddis);
      problem.add_dis("ale", aledis);
      problem.add_dis("scatra1", fluidscatradis);
      problem.add_dis("scatra2", structscatradis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(fluidscatradis, input, "TRANSPORT ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(structscatradis, input, "TRANSPORT2 ELEMENTS"));

#ifdef EXTENDEDPARALLELOVERLAP
      structdis->CreateExtendedOverlap(false, false, false);
#endif

      break;
    }
    case Core::ProblemType::biofilm_fsi:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          FOUR_C_THROW("Nurbs discretization not possible for biofilm problems!");
          break;
        }
        default:
        {
          structdis =
              std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
          fluiddis =
              std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
          aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
          structaledis =
              std::make_shared<Core::FE::Discretization>("structale", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));
      structaledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structaledis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("fluid", fluiddis);
      problem.add_dis("ale", aledis);
      problem.add_dis("structale", structaledis);


      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));

#ifdef EXTENDEDPARALLELOVERLAP
      structdis->CreateExtendedOverlap(false, false, false);
#endif

      // fluid scatra field
      fluidscatradis = std::make_shared<Core::FE::Discretization>("scatra1", comm, problem.n_dim());
      // create discretization writer - in constructor set into and owned by corresponding discret
      fluidscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          fluidscatradis, output_control, distype));
      problem.add_dis("scatra1", fluidscatradis);

      // structure scatra field
      structscatradis =
          std::make_shared<Core::FE::Discretization>("scatra2", comm, problem.n_dim());
      // create discretization writer - in constructor set into and owned by corresponding discret
      structscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          structscatradis, output_control, distype));
      problem.add_dis("scatra2", structscatradis);

      break;
    }
    case Core::ProblemType::fsi_xfem:
    case Core::ProblemType::fluid_xfem:
    {
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      problem.add_dis("structure", structdis);
      meshreader.add_advanced_reader(structdis, input, "STRUCTURE",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.structural_dynamic_params(), "GEOMETRY"),
          nullptr);

      if (problem.x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
      {
        fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
        fluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
        problem.add_dis("fluid", fluiddis);

        xfluiddis = std::make_shared<XFEM::DiscretizationXFEM>("xfluid", comm, problem.n_dim());
        xfluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(xfluiddis, output_control, distype));
        problem.add_dis("xfluid", xfluiddis);

        meshreader.add_element_reader(
            Core::IO::ElementReader(xfluiddis, input, "FLUID ELEMENTS", "FLUID"));
      }
      else
      {
        fluiddis = std::make_shared<XFEM::DiscretizationXFEM>("fluid", comm, problem.n_dim());
        fluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
        problem.add_dis("fluid", fluiddis);

        meshreader.add_advanced_reader(fluiddis, input, "FLUID",
            Teuchos::getIntegralValue<Core::IO::GeometryType>(
                problem.fluid_dynamic_params(), "GEOMETRY"),
            nullptr);
      }

      aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));
      problem.add_dis("ale", aledis);
      meshreader.add_element_reader(Core::IO::ElementReader(aledis, input, "ALE ELEMENTS"));
      break;
    }
    case Core::ProblemType::fpsi_xfem:
    {
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      fluiddis = std::make_shared<XFEM::DiscretizationXFEM>("fluid", comm, problem.n_dim());
      porofluiddis =
          std::make_shared<Core::FE::DiscretizationFaces>("porofluid", comm, problem.n_dim());
      aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("porofluid", porofluiddis);
      problem.add_dis("fluid", fluiddis);
      problem.add_dis("ale", aledis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));

      meshreader.add_advanced_reader(fluiddis, input, "FLUID",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.fluid_dynamic_params(), "GEOMETRY"),
          nullptr);

      meshreader.add_element_reader(Core::IO::ElementReader(aledis, input, "ALE ELEMENTS"));

      break;
    }
    case Core::ProblemType::ale:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          aledis =
              std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("ale", comm, problem.n_dim());
          break;
        }
        default:
        {
          aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));

      problem.add_dis("ale", aledis);

      meshreader.add_element_reader(Core::IO::ElementReader(aledis, input, "ALE ELEMENTS"));

      break;
    }
    case Core::ProblemType::fluid:
    case Core::ProblemType::fluid_redmodels:
    {
      if (distype == Core::FE::ShapeFunctionType::hdg)
      {
        fluiddis = std::make_shared<Core::FE::DiscretizationHDG>("fluid", comm, problem.n_dim());

        // create discretization writer - in constructor set into and owned by corresponding discret
        fluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      }
      else if (distype == Core::FE::ShapeFunctionType::nurbs)
      {
        fluiddis =
            std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("fluid", comm, problem.n_dim());

        // create discretization writer - in constructor set ingto and owned by corresponding
        // discret
        fluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      }
      else if (problem.fluid_dynamic_params().sublist("WALL MODEL").get<bool>("X_WALL"))
      {
        fluiddis = std::make_shared<XFEM::DiscretizationXWall>("fluid", comm, problem.n_dim());

        // create discretization writer - in constructor set into and owned by corresponding discret
        fluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      }
      else
      {
        // fluiddis  = Teuchos::rcp(new Core::FE::Discretization("fluid",reader.Comm()));
        fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());

        // create discretization writer - in constructor set into and owned by corresponding discret
        fluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      }

      problem.add_dis("fluid", fluiddis);

      meshreader.add_advanced_reader(fluiddis, input, "FLUID",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.fluid_dynamic_params(), "GEOMETRY"),
          nullptr);

      break;
    }
    case Core::ProblemType::lubrication:
    {
      // create empty discretizations
      lubricationdis =
          std::make_shared<Core::FE::Discretization>("lubrication", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      lubricationdis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          lubricationdis, output_control, distype));

      problem.add_dis("lubrication", lubricationdis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(lubricationdis, input, "LUBRICATION ELEMENTS"));

      break;
    }
    case Core::ProblemType::cardiac_monodomain:
    case Core::ProblemType::scatra:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          fluiddis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "fluid", comm, problem.n_dim());
          scatradis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "scatra", comm, problem.n_dim());
          break;
        }
        case Core::FE::ShapeFunctionType::hdg:
        {
          fluiddis =
              std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
          scatradis =
              std::make_shared<Core::FE::DiscretizationHDG>("scatra", comm, problem.n_dim());
          break;
        }
        default:
        {
          fluiddis =
              std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
          scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));


      problem.add_dis("fluid", fluiddis);
      problem.add_dis("scatra", scatradis);

      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));

      break;
    }
    case Core::ProblemType::sti:
    {
      // safety checks
      if (distype == Core::FE::ShapeFunctionType::nurbs)
        FOUR_C_THROW("Scatra-thermo interaction does not work for nurbs discretizations yet!");

      // create empty discretizations for scalar and thermo fields
      scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());
      thermdis = std::make_shared<Core::FE::Discretization>("thermo", comm, problem.n_dim());

      // create discretization writers
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));
      thermdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(thermdis, output_control, distype));

      // add empty discretizations to global problem
      problem.add_dis("scatra", scatradis);
      problem.add_dis("thermo", thermdis);

      // add element input to node input
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));

      break;
    }
    case Core::ProblemType::fluid_ale:
    {
      if (distype == Core::FE::ShapeFunctionType::hdg)
      {
        fluiddis = std::make_shared<Core::FE::DiscretizationHDG>("fluid", comm, problem.n_dim());
        aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
      }
      else if (distype == Core::FE::ShapeFunctionType::nurbs)
      {
        fluiddis =
            std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("fluid", comm, problem.n_dim());
        aledis =
            std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("ale", comm, problem.n_dim());
      }
      else if (problem.fluid_dynamic_params().sublist("WALL MODEL").get<bool>("X_WALL"))
      {
        fluiddis = std::make_shared<XFEM::DiscretizationXWall>("fluid", comm, problem.n_dim());
        aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
      }
      else
      {
        fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
        if (problem.x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
          xfluiddis = std::make_shared<XFEM::DiscretizationXFEM>("xfluid", comm, problem.n_dim());
        aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
      }


      // create discretization writer - in constructor set into and owned by corresponding discret
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      if (xfluiddis != nullptr)
      {
        xfluiddis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(xfluiddis, output_control, distype));
      }
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));


      problem.add_dis("fluid", fluiddis);
      if (xfluiddis != nullptr)
      {
        problem.add_dis("xfluid", xfluiddis);  // xfem discretization on slot 1
      }
      problem.add_dis("ale", aledis);

      if (xfluiddis != nullptr)
      {
        meshreader.add_element_reader(Core::IO::ElementReader(xfluiddis, input, "FLUID ELEMENTS"));
      }
      else
        meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));

      meshreader.add_element_reader(Core::IO::ElementReader(aledis, input, "ALE ELEMENTS"));

      break;
    }
    case Core::ProblemType::tsi:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          structdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "structure", comm, problem.n_dim());
          thermdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "thermo", comm, problem.n_dim());
          break;
        }
        default:
        {
          structdis =
              std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
          thermdis = std::make_shared<Core::FE::Discretization>("thermo", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      thermdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(thermdis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("thermo", thermdis);

      meshreader.add_advanced_reader(structdis, input, "STRUCTURE",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.structural_dynamic_params(), "GEOMETRY"),
          nullptr);
      meshreader.add_advanced_reader(thermdis, input, "THERMO",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.thermal_dynamic_params(), "GEOMETRY"),
          nullptr);

      break;
    }
    case Core::ProblemType::thermo:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          thermdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "thermo", comm, problem.n_dim());
          break;
        }
        default:
        {
          thermdis = std::make_shared<Core::FE::Discretization>("thermo", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      thermdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(thermdis, output_control, distype));

      problem.add_dis("thermo", thermdis);

      meshreader.add_element_reader(Core::IO::ElementReader(thermdis, input, "THERMO ELEMENTS"));

      break;
    }

    case Core::ProblemType::structure:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          structdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "structure", comm, problem.n_dim());
          break;
        }
        default:
        {
          structdis =
              std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));

      problem.add_dis("structure", structdis);

      meshreader.add_advanced_reader(structdis, input, "STRUCTURE",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.structural_dynamic_params(), "GEOMETRY"),
          nullptr);

      break;
    }

    case Core::ProblemType::polymernetwork:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      pboxdis = std::make_shared<Core::FE::Discretization>("boundingbox", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      pboxdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(pboxdis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("boundingbox", pboxdis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(pboxdis, input, "PERIODIC BOUNDINGBOX ELEMENTS"));

      break;
    }

    case Core::ProblemType::loma:
    {
      // create empty discretizations
      fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
      scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      problem.add_dis("fluid", fluiddis);
      problem.add_dis("scatra", scatradis);

      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));

      break;
    }

    case Core::ProblemType::fluid_xfem_ls:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      if (problem.get_problem_type() == Core::ProblemType::fluid_xfem_ls)
        fluiddis = std::make_shared<XFEM::DiscretizationXFEM>("fluid", comm, problem.n_dim());
      else
        fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
      scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("fluid", fluiddis);
      problem.add_dis("scatra", scatradis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_advanced_reader(fluiddis, input, "FLUID",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.fluid_dynamic_params(), "GEOMETRY"),
          nullptr);
      // meshreader.AddElementReader(Teuchos::rcp(new Core::IO::ElementReader(fluiddis, input,
      // "FLUID ELEMENTS")));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));
      break;
    }

    case Core::ProblemType::elch:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          fluiddis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "fluid", comm, problem.n_dim());
          scatradis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "scatra", comm, problem.n_dim());
          aledis =
              std::make_shared<Core::FE::Nurbs::NurbsDiscretization>("ale", comm, problem.n_dim());
          scatra_micro_dis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "scatra_micro", comm, problem.n_dim());
          break;
        }
        default:
        {
          fluiddis =
              std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
          scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());
          aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());
          scatra_micro_dis =
              std::make_shared<Core::FE::Discretization>("scatra_micro", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));
      scatra_micro_dis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          scatra_micro_dis, output_control, distype));

      problem.add_dis("fluid", fluiddis);
      problem.add_dis("scatra", scatradis);
      problem.add_dis("ale", aledis);
      problem.add_dis("scatra_micro", scatra_micro_dis);

      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(aledis, input, "ALE ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatra_micro_dis, input, "TRANSPORT2 ELEMENTS"));

      break;
    }
    case Core::ProblemType::art_net:  // _1D_ARTERY_
    {
      // create empty discretizations
      arterydis = std::make_shared<Core::FE::Discretization>("artery", comm, problem.n_dim());

      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          FOUR_C_THROW("Nurbs discretization not possible for artery");
          break;
        }
        default:
        {
          scatradis =
              std::make_shared<Core::FE::Discretization>("artery_scatra", comm, problem.n_dim());
          break;
        }
      }

      problem.add_dis("artery", arterydis);
      problem.add_dis("artery_scatra", scatradis);

      // create discretization writer - in constructor set into and owned by corresponding discret
      arterydis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(arterydis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      meshreader.add_element_reader(Core::IO::ElementReader(arterydis, input, "ARTERY ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));

      break;
    }
    case Core::ProblemType::red_airways:  // _reduced D airways
    {
      // create empty discretizations
      airwaydis = std::make_shared<Core::FE::Discretization>("red_airway", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      airwaydis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(airwaydis, output_control, distype));

      problem.add_dis("red_airway", airwaydis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(airwaydis, input, "REDUCED D AIRWAYS ELEMENTS"));

      break;
    }
    case Core::ProblemType::poroelast:
    case Core::ProblemType::poromultiphase:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          structdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "structure", comm, problem.n_dim());
          porofluiddis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "porofluid", comm, problem.n_dim());
          break;
        }
        default:
        {
          structdis =
              std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
          porofluiddis =
              std::make_shared<Core::FE::Discretization>("porofluid", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("porofluid", porofluiddis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(porofluiddis, input, "FLUID ELEMENTS"));

      if (problem.poro_multi_phase_dynamic_params().get<bool>("ARTERY_COUPLING"))
      {
        arterydis = std::make_shared<Core::FE::Discretization>("artery", comm, problem.n_dim());
        arterydis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(arterydis, output_control, distype));
        problem.add_dis("artery", arterydis);
        meshreader.add_element_reader(Core::IO::ElementReader(arterydis, input, "ARTERY ELEMENTS"));
      }

      break;
    }
    case Core::ProblemType::poromultiphasescatra:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          structdis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "structure", comm, problem.n_dim());
          porofluiddis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "porofluid", comm, problem.n_dim());
          scatradis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "scatra", comm, problem.n_dim());
          break;
        }
        default:
        {
          structdis =
              std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
          porofluiddis =
              std::make_shared<Core::FE::Discretization>("porofluid", comm, problem.n_dim());
          scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("porofluid", porofluiddis);
      problem.add_dis("scatra", scatradis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(porofluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));

      if (problem.poro_multi_phase_scatra_dynamic_params().get<bool>("ARTERY_COUPLING"))
      {
        arterydis = std::make_shared<Core::FE::Discretization>("artery", comm, problem.n_dim());
        arterydis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(arterydis, output_control, distype));
        problem.add_dis("artery", arterydis);
        meshreader.add_element_reader(Core::IO::ElementReader(arterydis, input, "ARTERY ELEMENTS"));

        artscatradis =
            std::make_shared<Core::FE::Discretization>("artery_scatra", comm, problem.n_dim());
        artscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
            artscatradis, output_control, distype));
        problem.add_dis("artery_scatra", artscatradis);
        meshreader.add_element_reader(
            Core::IO::ElementReader(artscatradis, input, "TRANSPORT ELEMENTS"));
      }

      break;
    }
    case Core::ProblemType::porofluidmultiphase:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          porofluiddis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(
              "porofluid", comm, problem.n_dim());
          break;
        }
        default:
        {
          porofluiddis =
              std::make_shared<Core::FE::Discretization>("porofluid", comm, problem.n_dim());
          break;
        }
      }

      // create discretization writer - in constructor set into and owned by corresponding discret
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));

      problem.add_dis("porofluid", porofluiddis);

      meshreader.add_element_reader(Core::IO::ElementReader(porofluiddis, input, "FLUID ELEMENTS"));

      if (problem.poro_fluid_multi_phase_dynamic_params().get<bool>("ARTERY_COUPLING"))
      {
        arterydis = std::make_shared<Core::FE::Discretization>("artery", comm, problem.n_dim());
        arterydis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(arterydis, output_control, distype));
        problem.add_dis("artery", arterydis);
        meshreader.add_element_reader(Core::IO::ElementReader(arterydis, input, "ARTERY ELEMENTS"));
      }
      break;
    }
    case Core::ProblemType::fpsi:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      porofluiddis = std::make_shared<Core::FE::Discretization>("porofluid", comm, problem.n_dim());
      fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
      aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("porofluid", porofluiddis);
      problem.add_dis("fluid", fluiddis);
      problem.add_dis("ale", aledis);

      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));

      break;
    }
    case Core::ProblemType::fbi:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("fluid", fluiddis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_advanced_reader(fluiddis, input, "FLUID",
          Teuchos::getIntegralValue<Core::IO::GeometryType>(
              problem.fluid_dynamic_params(), "GEOMETRY"),
          nullptr);

      break;
    }
    case Core::ProblemType::immersed_fsi:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("fluid", fluiddis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));

      break;
    }
    case Core::ProblemType::fps3i:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      porofluiddis = std::make_shared<Core::FE::Discretization>("porofluid", comm, problem.n_dim());
      fluiddis = std::make_shared<Core::FE::DiscretizationFaces>("fluid", comm, problem.n_dim());
      aledis = std::make_shared<Core::FE::Discretization>("ale", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));
      fluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(fluiddis, output_control, distype));
      aledis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(aledis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("porofluid", porofluiddis);
      problem.add_dis("fluid", fluiddis);
      problem.add_dis("ale", aledis);


      meshreader.add_element_reader(Core::IO::ElementReader(fluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));

      // fluid scatra field
      fluidscatradis = std::make_shared<Core::FE::Discretization>("scatra1", comm, problem.n_dim());
      // create discretization writer - in constructor set into and owned by corresponding discret
      fluidscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          fluidscatradis, output_control, distype));
      problem.add_dis("scatra1", fluidscatradis);

      // poro structure scatra field
      structscatradis =
          std::make_shared<Core::FE::Discretization>("scatra2", comm, problem.n_dim());
      // create discretization writer - in constructor set into and owned by corresponding discret
      structscatradis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          structscatradis, output_control, distype));
      problem.add_dis("scatra2", structscatradis);

      break;
    }
    case Core::ProblemType::poroscatra:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      porofluiddis = std::make_shared<Core::FE::Discretization>("porofluid", comm, problem.n_dim());
      scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      porofluiddis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(porofluiddis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("porofluid", porofluiddis);
      problem.add_dis("scatra", scatradis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(Core::IO::ElementReader(porofluiddis, input, "FLUID ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));
      break;
    }
    case Core::ProblemType::ehl:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      lubricationdis =
          std::make_shared<Core::FE::Discretization>("lubrication", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      lubricationdis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
          lubricationdis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("lubrication", lubricationdis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(lubricationdis, input, "LUBRICATION ELEMENTS"));

      break;
    }
    case Core::ProblemType::ssi:
    case Core::ProblemType::ssti:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("scatra", scatradis);

      // consider case of additional scatra manifold
      if (problem.ssi_control_params().sublist("MANIFOLD").get<bool>("ADD_MANIFOLD"))
      {
        auto scatra_manifold_dis =
            std::make_shared<Core::FE::Discretization>("scatra_manifold", comm, problem.n_dim());
        scatra_manifold_dis->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
            scatra_manifold_dis, output_control, distype));
        problem.add_dis("scatra_manifold", scatra_manifold_dis);
      }

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));

      if (problem.get_problem_type() == Core::ProblemType::ssti)
      {
        thermdis = std::make_shared<Core::FE::Discretization>("thermo", comm, problem.n_dim());
        thermdis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(thermdis, output_control, distype));
        problem.add_dis("thermo", thermdis);
        meshreader.add_element_reader(
            Core::IO::ElementReader(thermdis, input, "TRANSPORT ELEMENTS"));
      }

      break;
    }
    case Core::ProblemType::particle:
    case Core::ProblemType::pasi:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));

      problem.add_dis("structure", structdis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));

      break;
    }
    case Core::ProblemType::level_set:
    {
      // create empty discretizations
      scatradis = std::make_shared<Core::FE::Discretization>("scatra", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      scatradis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(scatradis, output_control, distype));

      problem.add_dis("scatra", scatradis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(scatradis, input, "TRANSPORT ELEMENTS"));
      break;
    }
    case Core::ProblemType::np_support:
    {
      // no discretizations and nodes needed for supporting procs
      break;
    }
    case Core::ProblemType::elemag:
    {
      // create empty discretizations
      elemagdis = std::make_shared<Core::FE::DiscretizationHDG>("elemag", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      elemagdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(elemagdis, output_control, distype));

      problem.add_dis("elemag", elemagdis);

      std::set<std::string> elemagelementtypes;
      elemagelementtypes.insert("ELECTROMAGNETIC");
      elemagelementtypes.insert("ELECTROMAGNETICDIFF");

      meshreader.add_element_reader(Core::IO::ElementReader(
          elemagdis, input, "ELECTROMAGNETIC ELEMENTS", elemagelementtypes));

      break;
    }
    case Core::ProblemType::redairways_tissue:
    {
      // create empty discretizations
      structdis = std::make_shared<Core::FE::Discretization>("structure", comm, problem.n_dim());
      airwaydis = std::make_shared<Core::FE::Discretization>("red_airway", comm, problem.n_dim());

      // create discretization writer - in constructor set into and owned by corresponding discret
      structdis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(structdis, output_control, distype));
      airwaydis->set_writer(
          std::make_shared<Core::IO::DiscretizationWriter>(airwaydis, output_control, distype));

      problem.add_dis("structure", structdis);
      problem.add_dis("red_airway", airwaydis);

      meshreader.add_element_reader(
          Core::IO::ElementReader(structdis, input, "STRUCTURE ELEMENTS"));
      meshreader.add_element_reader(
          Core::IO::ElementReader(airwaydis, input, "REDUCED D AIRWAYS ELEMENTS"));
    }
    break;
    default:
      FOUR_C_THROW("Unknown problem type: %d", problem.get_problem_type());
      break;
  }

  // add artery or airways discretizations only for the following problem types
  switch (problem.get_problem_type())
  {
    case Core::ProblemType::fsi_redmodels:
    case Core::ProblemType::fluid_ale:
    case Core::ProblemType::fluid_redmodels:
    {
      if (distype == Core::FE::ShapeFunctionType::polynomial)
      {
        // create empty discretizations
        arterydis = std::make_shared<Core::FE::Discretization>("artery", comm, problem.n_dim());
        // create discretization writer - in constructor set into and owned by corresponding discret
        arterydis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(arterydis, output_control, distype));
        problem.add_dis("artery", arterydis);
        meshreader.add_element_reader(Core::IO::ElementReader(arterydis, input, "ARTERY ELEMENTS"));

        airwaydis = std::make_shared<Core::FE::Discretization>("red_airway", comm, problem.n_dim());
        // create discretization writer - in constructor set into and owned by corresponding discret
        airwaydis->set_writer(
            std::make_shared<Core::IO::DiscretizationWriter>(airwaydis, output_control, distype));
        problem.add_dis("red_airway", airwaydis);
        meshreader.add_element_reader(
            Core::IO::ElementReader(airwaydis, input, "REDUCED D AIRWAYS ELEMENTS"));
      }
    }
    break;
    default:
      break;
  }

  if (read_mesh)  // now read and allocate!
  {
    // we read nodes and elements for the desired fields as specified above
    meshreader.read_and_partition();

    // care for special applications
    switch (problem.get_problem_type())
    {
      case Core::ProblemType::elch:
      case Core::ProblemType::fsi:
      case Core::ProblemType::fsi_redmodels:
      case Core::ProblemType::scatra:
      case Core::ProblemType::structure:
      {
        // read microscale fields from second, third, ... input file if necessary
        // (in case of multi-scale material models)
        read_micro_fields(problem, input.file_for_section("MATERIALS").parent_path());
        break;
      }
      case Core::ProblemType::np_support:
      {
        // read microscale fields from second, third, ... inputfile for supporting processors
        read_microfields_np_support(problem);
        break;
      }
      default:
        break;
    }
  }  // if(read_mesh)
}

void Global::read_micro_fields(Global::Problem& problem, const std::filesystem::path& input_path)
{
  // check whether micro material is specified
  const int id_struct = Global::Problem::instance()->materials()->first_id_by_type(
      Core::Materials::m_struct_multiscale);
  const int id_scatra = Global::Problem::instance()->materials()->first_id_by_type(
      Core::Materials::m_scatra_multiscale);
  const int id_elch = Global::Problem::instance()->materials()->first_id_by_type(
      Core::Materials::m_newman_multiscale);

  // return if no multiscale material is used
  if (id_struct == -1 and id_scatra == -1 and id_elch == -1) return;

  // safety check
  if ((id_struct != -1 and id_scatra != -1) or (id_struct != -1 and id_elch != -1) or
      (id_scatra != -1 and id_elch != -1))
    FOUR_C_THROW("Cannot have more than one multi-scale material!");

  // store name of macro-scale discretization in string
  std::string macro_dis_name("");
  if (id_struct != -1)
    macro_dis_name = "structure";
  else
    macro_dis_name = "scatra";

  // fetch communicators
  MPI_Comm lcomm = problem.get_communicators()->local_comm();
  MPI_Comm gcomm = problem.get_communicators()->global_comm();

  Global::Problem* macro_problem = Global::Problem::instance();
  std::shared_ptr<Core::FE::Discretization> macro_dis = macro_problem->get_dis(macro_dis_name);

  // repartition macro problem for a good distribution of elements with micro material
  if (macro_dis_name == "structure")
  {
    // do weighted repartitioning to obtain new row/column maps
    const Teuchos::ParameterList rebalanceParams;
    std::shared_ptr<const Epetra_CrsGraph> nodeGraph = macro_dis->build_node_graph();
    const auto& [nodeWeights, edgeWeights] = Core::Rebalance::build_weights(*macro_dis);
    const auto& [rownodes, colnodes] =
        Core::Rebalance::rebalance_node_maps(*nodeGraph, rebalanceParams, nodeWeights, edgeWeights);

    // rebuild the discretization with new maps
    macro_dis->redistribute(*rownodes, *colnodes);
  }

  // make sure that we read the micro discretizations only on the processors on
  // which elements with the corresponding micro material are evaluated

  std::set<int> my_multimat_IDs;

  // take care also of ghosted elements! -> ElementColMap!
  for (int i = 0; i < macro_dis->element_col_map()->NumMyElements(); ++i)
  {
    Core::Elements::Element* actele = macro_dis->l_col_element(i);
    std::shared_ptr<Core::Mat::Material> actmat = actele->material();

    if (id_elch != -1 and actmat->material_type() == Core::Materials::m_elchmat)
    {
      // extract wrapped material
      auto elchmat = std::dynamic_pointer_cast<const Mat::ElchMat>(actmat);
      auto elchphase = std::dynamic_pointer_cast<const Mat::ElchPhase>(
          elchmat->phase_by_id(elchmat->phase_id(0)));
      actmat = elchphase->mat_by_id(elchphase->mat_id(0));
    }

    if ((actmat->material_type() == Core::Materials::m_struct_multiscale and
            macro_dis_name == "structure") or
        (actmat->material_type() == Core::Materials::m_scatra_multiscale and
            macro_dis_name == "scatra") or
        (actmat->material_type() == Core::Materials::m_newman_multiscale and
            macro_dis_name == "scatra"))
    {
      Core::Mat::PAR::Parameter* actparams = actmat->parameter();
      my_multimat_IDs.insert(actparams->id());
    }
  }

  // check which macro procs have an element with micro material
  int foundmicromat = 0;
  int foundmicromatmyrank = -1;
  if (my_multimat_IDs.size() != 0)
  {
    foundmicromat = 1;
    foundmicromatmyrank = Core::Communication::my_mpi_rank(lcomm);
  }

  // find out how many procs have micro material
  int nummicromat = 0;
  Core::Communication::sum_all(&foundmicromat, &nummicromat, 1, lcomm);
  // broadcast number of procs that have micro material
  Core::Communication::broadcast(&nummicromat, 1, 0, gcomm);

  // every proc needs to know which procs have micro material in order to distribute colors
  // array is filled with either its local proc id or -1 when no micro mat was found
  std::vector<int> foundmyranks;
  foundmyranks.resize(Core::Communication::num_mpi_ranks(lcomm), -1);
  Core::Communication::gather_all(&foundmicromatmyrank, foundmyranks.data(), 1, lcomm);

  // determine color of macro procs with any contribution to micro material, only important for
  // procs with micro material color starts with 0 and is incremented for each group
  int color = -1;
  if (foundmicromat == 1)
  {
    for (int foundmyrank : foundmyranks)
    {
      if (foundmyrank != -1) ++color;
      if (foundmyrank == foundmicromatmyrank) break;
    }
  }
  else
  {
    color = MPI_UNDEFINED;
  }

  // do the splitting of the communicator (macro proc must always be proc in subcomm with lowest
  // key
  // --> 0 is inserted here)
  MPI_Comm mpi_local_comm;
  MPI_Comm_split(gcomm, color, 0 /*important here*/, &mpi_local_comm);

  // sort out macro procs that do not have micro material
  if (foundmicromat == 1)
  {
    // create the sub communicator that includes one macro proc and some supporting procs
    MPI_Comm subgroupcomm = mpi_local_comm;
    problem.get_communicators()->set_sub_comm(subgroupcomm);

    // find out how many micro problems have to be solved on this macro proc
    int microcount = 0;
    for (const auto& material_map : problem.materials()->map())
    {
      int matid = material_map.first;
      if (my_multimat_IDs.find(matid) != my_multimat_IDs.end()) microcount++;
    }
    // and broadcast it to the corresponding group of procs
    Core::Communication::broadcast(&microcount, 1, 0, subgroupcomm);

    for (const auto& material_map : problem.materials()->map())
    {
      int matid = material_map.first;

      if (my_multimat_IDs.find(matid) != my_multimat_IDs.end())
      {
        std::shared_ptr<Core::Mat::Material> mat = Mat::factory(matid);

        // initialize variables storing micro-scale information
        int microdisnum(-1);
        std::string micro_dis_name = "";
        std::string micro_inputfile_name("");
        Global::Problem* micro_problem(nullptr);

        // structure case
        if (macro_dis_name == "structure")
        {
          // access multi-scale structure material
          auto* micromat = static_cast<Mat::MicroMaterial*>(mat.get());

          // extract and broadcast number of micro-scale discretization
          microdisnum = micromat->micro_dis_num();
          Core::Communication::broadcast(&microdisnum, 1, 0, subgroupcomm);

          // set name of micro-scale discretization
          micro_dis_name = "structure";

          // extract name of micro-scale input file
          micro_inputfile_name = micromat->micro_input_file_name();

          // instantiate micro-scale problem
          micro_problem = Global::Problem::instance(microdisnum);
        }

        // scalar transport case
        else
        {
          // access multi-scale scalar transport material
          Mat::ScatraMicroMacroCoupling* micromat = nullptr;
          if (id_scatra != -1)
            micromat = dynamic_cast<Mat::ScatraMultiScale*>(mat.get());
          else if (id_elch != -1)
            micromat = dynamic_cast<Mat::NewmanMultiScale*>(mat.get());
          else
            FOUR_C_THROW("How the heck did you get here?!");

          // extract and broadcast number of micro-scale discretization
          microdisnum = micromat->micro_dis_num();
          Core::Communication::broadcast(&microdisnum, 1, 0, subgroupcomm);

          // set unique name of micro-scale discretization
          std::stringstream name;
          name << "scatra_multiscale_" << microdisnum;
          micro_dis_name = name.str();

          // extract name of micro-scale input file
          micro_inputfile_name = micromat->micro_input_file_name();

          // instantiate micro-scale problem
          micro_problem = Global::Problem::instance(microdisnum);
        }


        if (micro_inputfile_name[0] != '/')
        {
          micro_inputfile_name = input_path / micro_inputfile_name;
        }

        // broadcast micro input file name
        int length = static_cast<int>(micro_inputfile_name.length());
        Core::Communication::broadcast(&length, 1, 0, subgroupcomm);
        Core::Communication::broadcast(
            (const_cast<char*>(micro_inputfile_name.c_str())), length, 0, subgroupcomm);

        // start with actual reading
        Core::IO::InputFile micro_reader(micro_inputfile_name, subgroupcomm);

        std::shared_ptr<Core::FE::Discretization> dis_micro =
            std::make_shared<Core::FE::Discretization>(
                micro_dis_name, subgroupcomm, problem.n_dim());

        // replace standard dofset inside micro discretization by independent dofset
        // to avoid inconsistent dof numbering in non-nested parallel settings with more than one
        // micro discretization
        if (problem.get_communicators()->np_type() ==
            Core::Communication::NestedParallelismType::no_nested_parallelism)
          dis_micro->replace_dof_set(std::make_shared<Core::DOFSets::IndependentDofSet>());

        // create discretization writer - in constructor set into and owned by corresponding
        // discret
        dis_micro->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(dis_micro,
            micro_problem->output_control_file(), micro_problem->spatial_approximation_type()));

        micro_problem->add_dis(micro_dis_name, dis_micro);

        read_parameter(*micro_problem, micro_reader);

        // read materials of microscale
        // CAUTION: materials for microscale cannot be read until
        // micro_reader is activated, since else materials will again be
        // read from macroscale inputfile. Besides, materials MUST be read
        // before elements are read since elements establish a connection
        // to the corresponding material! Thus do not change position of
        // function calls!
        problem.materials()->set_read_from_problem(microdisnum);

        read_materials(*micro_problem, micro_reader);

        Core::IO::MeshReader micromeshreader(micro_reader, "NODE COORDS",
            {.mesh_paritioning_parameters = Problem::instance()->mesh_partitioning_params(),
                .geometric_search_parameters = Problem::instance()->geometric_search_params(),
                .io_parameters = Problem::instance()->io_params()});

        if (micro_dis_name == "structure")
        {
          micromeshreader.add_element_reader(
              Core::IO::ElementReader(dis_micro, micro_reader, "STRUCTURE ELEMENTS"));
        }
        else
          micromeshreader.add_element_reader(
              Core::IO::ElementReader(dis_micro, micro_reader, "TRANSPORT ELEMENTS"));

        micromeshreader.read_and_partition();


        {
          Core::Utils::FunctionManager function_manager;
          global_legacy_module_callbacks().AttachFunctionDefinitions(function_manager);
          function_manager.read_input(micro_reader);
          micro_problem->set_function_manager(std::move(function_manager));
        }

        read_result(*micro_problem, micro_reader);
        read_conditions(*micro_problem, micro_reader);

        // At this point, everything for the microscale is read,
        // subsequent reading is only for macroscale
        dis_micro->fill_complete();

        // broadcast restart information
        int restart_step = problem.restart();
        Core::Communication::broadcast(&restart_step, 1, 0, subgroupcomm);
        problem.set_restart_step(restart_step);

        // set the problem number from which to call materials again to zero
        // (i.e. macro problem), cf. Mat::factory!
        problem.materials()->reset_read_from_problem();
      }
    }
    problem.materials()->reset_read_from_problem();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_microfields_np_support(Global::Problem& problem)
{
  MPI_Comm lcomm = problem.get_communicators()->local_comm();
  MPI_Comm gcomm = problem.get_communicators()->global_comm();

  // receive number of procs that have micro material
  int nummicromat = 0;
  Core::Communication::broadcast(&nummicromat, 1, 0, gcomm);

  // prepare the supporting procs for a splitting of gcomm

  // groups should be equally sized
  // in a first step every macro proc that needs support gets procpergroup supporting procs
  int procpergroup = int(floor((Core::Communication::num_mpi_ranks(lcomm)) / nummicromat));
  std::vector<int> supgrouplayout(nummicromat, procpergroup);
  // remaining procs are added to the groups in the beginning
  int remainingProcs = Core::Communication::num_mpi_ranks(lcomm) - procpergroup * nummicromat;
  for (int k = 0; k < remainingProcs; ++k)
  {
    supgrouplayout[k]++;
  }

  // secondly: colors are distributed
  // color starts with 0 and is incremented for each group
  int color = -1;
  int gsum = 0;
  do
  {
    color++;
    gsum += supgrouplayout[color];
  } while (gsum <= Core::Communication::my_mpi_rank(lcomm));

  // do the splitting of the communicator
  MPI_Comm mpi_local_comm;
  MPI_Comm_split(gcomm, color, Core::Communication::my_mpi_rank(gcomm), &mpi_local_comm);

  // create the sub communicator that includes one macro proc and some supporting procs
  MPI_Comm subgroupcomm = mpi_local_comm;
  problem.get_communicators()->set_sub_comm(subgroupcomm);

  // number of micro problems for this sub group
  int microcount = 0;
  Core::Communication::broadcast(&microcount, 1, 0, subgroupcomm);

  for (int n = 0; n < microcount; n++)
  {
    // broadcast microdis number
    int microdisnum = -1;
    Core::Communication::broadcast(&microdisnum, 1, 0, subgroupcomm);

    Global::Problem* micro_problem = Global::Problem::instance(microdisnum);

    // broadcast micro input file name
    int length = -1;
    std::string micro_inputfile_name;
    Core::Communication::broadcast(&length, 1, 0, subgroupcomm);
    micro_inputfile_name.resize(length);
    Core::Communication::broadcast(
        (const_cast<char*>(micro_inputfile_name.c_str())), length, 0, subgroupcomm);

    // start with actual reading
    Core::IO::InputFile micro_reader(micro_inputfile_name, subgroupcomm);

    std::shared_ptr<Core::FE::Discretization> structdis_micro =
        std::make_shared<Core::FE::Discretization>("structure", subgroupcomm, problem.n_dim());

    // create discretization writer - in constructor set into and owned by corresponding discret
    structdis_micro->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(structdis_micro,
        micro_problem->output_control_file(), micro_problem->spatial_approximation_type()));

    micro_problem->add_dis("structure", structdis_micro);

    read_parameter(*micro_problem, micro_reader);

    // read materials of microscale
    // CAUTION: materials for microscale cannot be read until
    // micro_reader is activated, since else materials will again be
    // read from macroscale inputfile. Besides, materials MUST be read
    // before elements are read since elements establish a connection
    // to the corresponding material! Thus do not change position of
    // function calls!
    problem.materials()->set_read_from_problem(microdisnum);

    read_materials(*micro_problem, micro_reader);

    Core::IO::MeshReader micromeshreader(micro_reader, "NODE COORDS",
        {.mesh_paritioning_parameters = Problem::instance()->mesh_partitioning_params(),
            .geometric_search_parameters = Problem::instance()->geometric_search_params(),
            .io_parameters = Problem::instance()->io_params()});
    micromeshreader.add_element_reader(
        Core::IO::ElementReader(structdis_micro, micro_reader, "STRUCTURE ELEMENTS"));
    micromeshreader.read_and_partition();

    // read conditions of microscale
    // -> note that no time curves and spatial functions can be read!

    read_conditions(*micro_problem, micro_reader);

    // At this point, everything for the microscale is read,
    // subsequent reading is only for macroscale
    structdis_micro->fill_complete();

    // broadcast restart information
    int restart_step = problem.restart();
    Core::Communication::broadcast(&restart_step, 1, 0, subgroupcomm);
    problem.set_restart_step(restart_step);

    // set the problem number from which to call materials again to zero
    // (i.e. macro problem), cf. Mat::factory!
    problem.materials()->reset_read_from_problem();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_parameter(Global::Problem& problem, Core::IO::InputFile& input)
{
  std::shared_ptr<Teuchos::ParameterList> list =
      std::make_shared<Teuchos::ParameterList>("DAT FILE");

  Core::IO::read_parameters_in_section(input, "DISCRETISATION", *list);
  Core::IO::read_parameters_in_section(input, "PROBLEM SIZE", *list);
  Core::IO::read_parameters_in_section(input, "PROBLEM TYPE", *list);
  Core::IO::read_parameters_in_section(input, "BINNING STRATEGY", *list);
  Core::IO::read_parameters_in_section(input, "BOUNDINGVOLUME STRATEGY", *list);
  Core::IO::read_parameters_in_section(input, "IO", *list);
  Core::IO::read_parameters_in_section(input, "IO/EVERY ITERATION", *list);
  Core::IO::read_parameters_in_section(input, "IO/MONITOR STRUCTURE DBC", *list);
  Core::IO::read_parameters_in_section(input, "IO/RUNTIME VTK OUTPUT", *list);
  Core::IO::read_parameters_in_section(input, "IO/RUNTIME VTK OUTPUT/FLUID", *list);
  Core::IO::read_parameters_in_section(input, "IO/RUNTIME VTK OUTPUT/STRUCTURE", *list);
  Core::IO::read_parameters_in_section(input, "IO/RUNTIME VTK OUTPUT/BEAMS", *list);
  Core::IO::read_parameters_in_section(input, "IO/RUNTIME VTP OUTPUT STRUCTURE", *list);
  Core::IO::read_parameters_in_section(input, "STRUCTURAL DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "STRUCTURAL DYNAMIC/TIMEADAPTIVITY", *list);
  Core::IO::read_parameters_in_section(input, "STRUCTURAL DYNAMIC/ERROR EVALUATION", *list);
  Core::IO::read_parameters_in_section(input, "STRUCTURAL DYNAMIC/GENALPHA", *list);
  Core::IO::read_parameters_in_section(input, "STRUCTURAL DYNAMIC/ONESTEPTHETA", *list);
  Core::IO::read_parameters_in_section(
      input, "STRUCTURAL DYNAMIC/TIMEADAPTIVITY/JOINT EXPLICIT", *list);
  Core::IO::read_parameters_in_section(input, "MORTAR COUPLING", *list);
  Core::IO::read_parameters_in_section(input, "MORTAR COUPLING/PARALLEL REDISTRIBUTION", *list);
  Core::IO::read_parameters_in_section(input, "CONTACT DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "CARDIOVASCULAR 0D-STRUCTURE COUPLING", *list);
  Core::IO::read_parameters_in_section(
      input, "CARDIOVASCULAR 0D-STRUCTURE COUPLING/SYS-PUL CIRCULATION PARAMETERS", *list);
  Core::IO::read_parameters_in_section(
      input, "CARDIOVASCULAR 0D-STRUCTURE COUPLING/RESPIRATORY PARAMETERS", *list);
  Core::IO::read_parameters_in_section(input, "BROWNIAN DYNAMICS", *list);
  Core::IO::read_parameters_in_section(input, "BEAM INTERACTION", *list);
  Core::IO::read_parameters_in_section(input, "BEAM INTERACTION/SPHERE BEAM LINK", *list);
  Core::IO::read_parameters_in_section(input, "BEAM INTERACTION/BEAM TO BEAM CONTACT", *list);
  Core::IO::read_parameters_in_section(input, "BEAM INTERACTION/BEAM TO SPHERE CONTACT", *list);
  Core::IO::read_parameters_in_section(
      input, "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT", *list);
  Core::IO::read_parameters_in_section(
      input, "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING", *list);
  Core::IO::read_parameters_in_section(input, "BEAM INTERACTION/BEAM TO SOLID SURFACE", *list);
  Core::IO::read_parameters_in_section(
      input, "BEAM INTERACTION/BEAM TO SOLID SURFACE/RUNTIME VTK OUTPUT", *list);
  Core::IO::read_parameters_in_section(
      input, "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING", *list);
  Core::IO::read_parameters_in_section(
      input, "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/RUNTIME VTK OUTPUT", *list);
  Core::IO::read_parameters_in_section(input, "BEAM INTERACTION/CROSSLINKING", *list);
  Core::IO::read_parameters_in_section(input, "THERMAL DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "THERMAL DYNAMIC/GENALPHA", *list);
  Core::IO::read_parameters_in_section(input, "THERMAL DYNAMIC/ONESTEPTHETA", *list);
  Core::IO::read_parameters_in_section(input, "TSI DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "TSI DYNAMIC/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "TSI DYNAMIC/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "TSI CONTACT", *list);
  Core::IO::read_parameters_in_section(input, "POROELASTICITY DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "POROSCATRA CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "POROFLUIDMULTIPHASE DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "POROFLUIDMULTIPHASE DYNAMIC/ARTERY COUPLING", *list);
  Core::IO::read_parameters_in_section(input, "POROMULTIPHASE DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "POROMULTIPHASE DYNAMIC/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "POROMULTIPHASE DYNAMIC/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "POROMULTIPHASESCATRA DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "POROMULTIPHASESCATRA DYNAMIC/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "POROMULTIPHASESCATRA DYNAMIC/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "ELASTO HYDRO DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "ELASTO HYDRO DYNAMIC/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "ELASTO HYDRO DYNAMIC/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "EMBEDDED MESH COUPLING", *list);
  Core::IO::read_parameters_in_section(input, "SSI CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "SSI CONTROL/ELCH", *list);
  Core::IO::read_parameters_in_section(input, "SSI CONTROL/MANIFOLD", *list);
  Core::IO::read_parameters_in_section(input, "SSI CONTROL/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "SSI CONTROL/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "SSTI CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "SSTI CONTROL/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "SSTI CONTROL/THERMO", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/EDGE-BASED STABILIZATION", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/POROUS-FLOW STABILIZATION", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/TURBULENCE MODEL", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/SUBGRID VISCOSITY", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/WALL MODEL", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/TIMEADAPTIVITY", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/MULTIFRACTAL SUBGRID SCALES", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/TURBULENT INFLOW", *list);
  Core::IO::read_parameters_in_section(input, "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES", *list);
  Core::IO::read_parameters_in_section(input, "LUBRICATION DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "SCALAR TRANSPORT DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "SCALAR TRANSPORT DYNAMIC/NONLINEAR", *list);
  Core::IO::read_parameters_in_section(input, "SCALAR TRANSPORT DYNAMIC/STABILIZATION", *list);
  Core::IO::read_parameters_in_section(input, "SCALAR TRANSPORT DYNAMIC/S2I COUPLING", *list);
  Core::IO::read_parameters_in_section(input, "SCALAR TRANSPORT DYNAMIC/ARTERY COUPLING", *list);
  Core::IO::read_parameters_in_section(input, "SCALAR TRANSPORT DYNAMIC/EXTERNAL FORCE", *list);
  Core::IO::read_parameters_in_section(input, "STI DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "STI DYNAMIC/MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "STI DYNAMIC/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "FS3I DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "FS3I DYNAMIC/PARTITIONED", *list);
  Core::IO::read_parameters_in_section(input, "FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION", *list);
  Core::IO::read_parameters_in_section(input, "ALE DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "FSI DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "FSI DYNAMIC/CONSTRAINT", *list);
  Core::IO::read_parameters_in_section(input, "FSI DYNAMIC/MONOLITHIC SOLVER", *list);
  Core::IO::read_parameters_in_section(input, "FSI DYNAMIC/PARTITIONED SOLVER", *list);
  Core::IO::read_parameters_in_section(input, "FSI DYNAMIC/TIMEADAPTIVITY", *list);
  Core::IO::read_parameters_in_section(input, "FLUID BEAM INTERACTION", *list);
  Core::IO::read_parameters_in_section(
      input, "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING", *list);
  Core::IO::read_parameters_in_section(
      input, "FLUID BEAM INTERACTION/BEAM TO FLUID MESHTYING/RUNTIME VTK OUTPUT", *list);
  Core::IO::read_parameters_in_section(input, "IMMERSED METHOD", *list);
  Core::IO::read_parameters_in_section(input, "IMMERSED METHOD/PARTITIONED SOLVER", *list);
  Core::IO::read_parameters_in_section(input, "FPSI DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "ARTERIAL DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "REDUCED DIMENSIONAL AIRWAYS DYNAMIC", *list);
  Core::IO::read_parameters_in_section(
      input, "COUPLED REDUCED-D AIRWAYS AND TISSUE DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "SEARCH TREE", *list);
  Core::IO::read_parameters_in_section(input, "XFEM GENERAL", *list);
  Core::IO::read_parameters_in_section(input, "CUT GENERAL", *list);
  Core::IO::read_parameters_in_section(input, "XFLUID DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "XFLUID DYNAMIC/GENERAL", *list);
  Core::IO::read_parameters_in_section(input, "XFLUID DYNAMIC/STABILIZATION", *list);
  Core::IO::read_parameters_in_section(input, "XFLUID DYNAMIC/XFPSI MONOLITHIC", *list);
  Core::IO::read_parameters_in_section(input, "LOMA CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "ELCH CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "ELCH CONTROL/DIFFCOND", *list);
  Core::IO::read_parameters_in_section(input, "ELCH CONTROL/SCL", *list);
  Core::IO::read_parameters_in_section(input, "BIOFILM CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "PARTICLE DYNAMIC", *list);
  Core::IO::read_parameters_in_section(
      input, "PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS", *list);
  Core::IO::read_parameters_in_section(input, "PARTICLE DYNAMIC/SPH", *list);
  Core::IO::read_parameters_in_section(input, "PARTICLE DYNAMIC/DEM", *list);
  Core::IO::read_parameters_in_section(input, "PASI DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "LEVEL-SET CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "LEVEL-SET CONTROL/REINITIALIZATION", *list);
  Core::IO::read_parameters_in_section(input, "WEAR", *list);
  Core::IO::read_parameters_in_section(input, "BEAM CONTACT", *list);
  Core::IO::read_parameters_in_section(input, "BEAM CONTACT/RUNTIME VTK OUTPUT", *list);
  Core::IO::read_parameters_in_section(input, "BEAM POTENTIAL", *list);
  Core::IO::read_parameters_in_section(input, "BEAM POTENTIAL/RUNTIME VTK OUTPUT", *list);
  Core::IO::read_parameters_in_section(input, "SEMI-SMOOTH PLASTICITY", *list);
  Core::IO::read_parameters_in_section(input, "ELECTROMAGNETIC DYNAMIC", *list);
  Core::IO::read_parameters_in_section(input, "VOLMORTAR COUPLING", *list);
  Core::IO::read_parameters_in_section(input, "CARDIAC MONODOMAIN CONTROL", *list);
  Core::IO::read_parameters_in_section(input, "MOR", *list);
  Core::IO::read_parameters_in_section(input, "MESH PARTITIONING", *list);
  Core::IO::read_parameters_in_section(input, "MULTI POINT CONSTRAINTS", *list);
  Core::IO::read_parameters_in_section(input, "NURBS", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Direction", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Direction/Newton", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Direction/Newton/Modified", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Direction/Newton/Linear Solver", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Direction/Steepest Descent", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Line Search", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Line Search/Full Step", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Line Search/Backtrack", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Line Search/Polynomial", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Line Search/More'-Thuente", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Pseudo Transient", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Trust Region", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Printing", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Status Test", *list);
  Core::IO::read_parameters_in_section(input, "STRUCT NOX/Solver Options", *list);

  // read in solver sections
  // Note: the maximum number of solver blocks in dat files is hardwired here.
  // If you change this do not forget to edit the corresponding parts in
  // validparameters.cpp, too!
  for (int i = 1; i < 10; i++)
  {
    std::stringstream ss;
    ss << "SOLVER " << i;
    Core::IO::read_parameters_in_section(input, ss.str(), *list);

    // adapt path of XML file if necessary
    Teuchos::ParameterList& sublist = list->sublist(ss.str());
    std::vector<std::string> listOfFileNameParameters = {
        "AMGNXN_XML_FILE", "MUELU_XML_FILE", "TEKO_XML_FILE", "SOLVER_XML_FILE"};

    for (auto& filenameParameter : listOfFileNameParameters)
    {
      auto* xml_filename = sublist.getPtr<std::string>(filenameParameter);
      if (xml_filename != nullptr and *xml_filename != "none")
      {
        // make path relative to input file path if it is not an absolute path
        if ((*xml_filename)[0] != '/')
        {
          auto input_filename = input.file_for_section(ss.str());
          *xml_filename = input_filename.parent_path() / *xml_filename;
        }
      }
    }
  }

  // read in STRUCT NOX/Status Test and modify the xml file name, if there
  // is one.
  if (list->sublist("STRUCT NOX").sublist("Status Test").isParameter("XML File"))
  {
    // adapt path of XML file if necessary
    Teuchos::ParameterList& sublist = list->sublist("STRUCT NOX").sublist("Status Test");
    auto* statustest_xmlfile = sublist.getPtr<std::string>("XML File");
    // make path relative to input file path if it is not an absolute path
    if (((*statustest_xmlfile)[0] != '/') and ((*statustest_xmlfile) != "none"))
    {
      auto input_filename = input.file_for_section("STRUCT NOX/Status Test");
      *statustest_xmlfile = input_filename.parent_path() / *statustest_xmlfile;
      std::cout << "XML file for NOX status test: " << *statustest_xmlfile << std::endl;
    }
  }  // STRUCT NOX/Status Test

  // check for invalid parameters
  problem.set_parameter_list(list);

  //---------------------------------------------------------------------
  // Now we have successfully read the whole input file. It's time to access some data

  // 1) get the problem type
  const Teuchos::ParameterList& type = problem.problem_type_params();
  problem.set_problem_type(Teuchos::getIntegralValue<Core::ProblemType>(type, "PROBLEMTYPE"));

  // 2) get the spatial approximation type
  problem.set_spatial_approximation_type(
      Teuchos::getIntegralValue<Core::FE::ShapeFunctionType>(type, "SHAPEFCT"));

  int restart_step = problem.restart();
  // 3) do the restart business with the four options we support (partially)
  if (restart_step == 0)
  {
    // no restart flag on the command line, so check the restart flag from the input file
    restart_step = type.get<int>("RESTART");
    problem.set_restart_step(restart_step);
  }
  else  // SetRestartStep() has been called before!
  {
    // There is a non-zero restart flag on the command line, so we ignore the input file.
    // The RESTART flag in the input file should be zero or have the same value!
    const int restartflaginfile = type.get<int>("RESTART");
    if ((restartflaginfile > 0) and (restartflaginfile != restart_step))
      FOUR_C_THROW("Restart flags in input file and command line are non-zero and different!");
  }

  // Set restart time based on walltime
  const double restartinterval = problem.io_params().get<double>("RESTARTWALLTIMEINTERVAL");
  const int restartevry = problem.io_params().get<int>("RESTARTEVRY");
  problem.restart_manager()->setup_restart_manager(restartinterval, restartevry);

  // 4) set random seed
  // time is in seconds, therefore we add the global processor id to obtain a unique seed on each
  // proc
  {
    int rs = type.get<int>("RANDSEED");
    if (rs < 0)
      rs = static_cast<int>(time(nullptr)) +
           42 * Core::Communication::my_mpi_rank(
                    Global::Problem::instance(0)->get_communicators()->global_comm());

    srand((unsigned int)rs);  // Set random seed for stdlibrary. This is deprecated, as it does not
    // produce random numbers on some platforms!
    problem.random()->set_rand_seed((unsigned int)rs);  // Use this instead.
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_materials(Global::Problem& problem, Core::IO::InputFile& input)
{
  // create list of known materials
  std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> vm =
      Input::valid_materials();
  std::vector<std::shared_ptr<Mat::MaterialDefinition>>& matlist = *vm;

  std::vector<Core::IO::InputSpec> all_specs;
  std::vector<Core::Materials::MaterialType> all_types;
  for (auto& mat : matlist)
  {
    all_specs.emplace_back(Core::IO::InputSpecBuilders::group(mat->name(), mat->specs()));
    all_types.push_back(mat->type());
  }

  // Whenever one of the materials is read, the lambda function will update this index to the
  // current material index. This lets us access the correct subcontainer for the current material
  // without searching through all of them.
  std::size_t current_index = 0;

  auto all_materials = Core::IO::InputSpecBuilders::one_of(all_specs,
      [&current_index](Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container,
          std::size_t index) { current_index = index; });

  for (const auto& line : input.lines_in_section("MATERIALS"))
  {
    Core::IO::InputParameterContainer container;
    Core::IO::ValueParser parser(
        line, {.user_scope_message = "While reading 'MATERIALS' section: ",
                  .base_path = input.file_for_section("MATERIALS").parent_path()});

    parser.consume("MAT");
    const int mat_id = parser.read<int>();
    FOUR_C_ASSERT_ALWAYS(mat_id >= 0, "Material ID must be non-negative. Found: %d", mat_id);

    if (problem.materials()->id_exists(mat_id))
      FOUR_C_THROW("More than one material with 'MAT %d'", mat_id);

    Core::IO::fully_parse(parser, all_materials, container);

    problem.materials()->insert(
        mat_id, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>(
                    [mat_id, mat_type = all_types[current_index],
                        container = container.group(all_specs[current_index].name())]()
                    { return Mat::make_parameter(mat_id, mat_type, container); }));
  }

  // We have read in all the materials and now we force construction of them all. The LazyPtr
  // ensures that the ordering does not matter. Note that we do not wait any longer for
  // construction, because materials might later be used in code sections that only run on proc 0.
  // Doing anything MPI-parallel inside the material constructors would then fail. Unfortunately,
  // such operations happen in the code base, thus we construct the materials here.
  for (const auto& [id, mat] : problem.materials()->map())
  {
    // This is the point where the material is actually constructed via the side effect that we try
    // to access the material.
    (void)mat.get();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_contact_constitutive_laws(Global::Problem& problem, Core::IO::InputFile& input)
{
  // create list of known contact constitutive laws
  std::shared_ptr<std::vector<std::shared_ptr<CONTACT::CONSTITUTIVELAW::LawDefinition>>> vm =
      Input::valid_contact_constitutive_laws();
  std::vector<std::shared_ptr<CONTACT::CONSTITUTIVELAW::LawDefinition>>& coconstlawlist = *vm;

  // test for each contact constitutive law definition (input file --CONTACT CONSTITUTIVE LAWS
  // section) and store it
  for (auto& m : coconstlawlist)
  {
    // read contact constitutive law from DAT file of type
    m->read(problem, input, *problem.contact_constitutive_laws());
  }

  // check if every contact constitutive law was identified
  const std::string contact_const_laws = "CONTACT CONSTITUTIVE LAWS";
  for (const auto& section_i : input.lines_in_section(contact_const_laws))
  {
    std::stringstream condline{std::string{section_i}};

    std::string coconstlaw;
    std::string number;
    std::string name;
    (condline) >> coconstlaw >> number >> name;
    if ((not(condline)) or (coconstlaw != "LAW"))
      FOUR_C_THROW("invalid contact constitutive law line in '%s'", name.c_str());

    // extract contact constitutive law ID
    int coconstlawid = -1;
    {
      char* ptr;
      coconstlawid = static_cast<int>(strtol(number.c_str(), &ptr, 10));
      if (ptr == number.c_str())
        FOUR_C_THROW("failed to read contact constitutive law object number '%s'", number.c_str());
    }

    // processed?
    if (problem.contact_constitutive_laws()->find(coconstlawid) == -1)
      FOUR_C_THROW("Contact constitutive law 'LAW %d' with name '%s' could not be identified",
          coconstlawid, name.c_str());
  }

  // make fast access parameters
  problem.contact_constitutive_laws()->make_parameters();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_cloning_material_map(Global::Problem& problem, Core::IO::InputFile& input)
{
  const std::vector<Input::LineDefinition> lines = Core::FE::valid_cloning_material_map_lines();

  // perform the actual reading and extract the input parameters
  std::vector<Input::LineDefinition> input_line_vec =
      Core::IO::InputFileUtils::read_all_lines_in_section(input, "CLONING MATERIAL MAP", lines);
  for (const auto& input_line : input_line_vec)
  {
    // extract what was read from the input file
    std::string src_field = input_line.container().get<std::string>("SRC_FIELD");
    int src_matid = input_line.container().get_or<int>("SRC_MAT", -1);
    std::string tar_field = input_line.container().get<std::string>("TAR_FIELD");
    int tar_matid = input_line.container().get_or<int>("TAR_MAT", -1);

    // create the key pair
    std::pair<std::string, std::string> fields(src_field, tar_field);

    // enter the material pairing into the map
    std::pair<int, int> matmap(src_matid, tar_matid);
    problem.cloning_material_map()[fields].insert(matmap);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_result(Global::Problem& problem, Core::IO::InputFile& input)
{
  const auto lines = global_legacy_module_callbacks().valid_result_description_lines();

  // read design nodes <-> nodes, lines <-> nodes, surfaces <-> nodes, volumes <-> nodes
  const auto get_discretization_callback = [](const std::string& name) -> decltype(auto)
  { return *Global::Problem::instance()->get_dis(name); };
  std::vector<std::vector<std::vector<int>>> nodeset(4);
  Core::IO::read_design(input, "DNODE", nodeset[0], get_discretization_callback);
  Core::IO::read_design(input, "DLINE", nodeset[1], get_discretization_callback);
  Core::IO::read_design(input, "DSURF", nodeset[2], get_discretization_callback);
  Core::IO::read_design(input, "DVOL", nodeset[3], get_discretization_callback);
  problem.get_result_test_manager().set_node_set(nodeset);

  problem.get_result_test_manager().set_parsed_lines(
      Core::IO::InputFileUtils::read_all_lines_in_section(input, "RESULT DESCRIPTION", lines));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_conditions(Global::Problem& problem, Core::IO::InputFile& input)
{
  Teuchos::Time time("", true);
  if (Core::Communication::my_mpi_rank(input.get_comm()) == 0)
  {
    Core::IO::cout << "Read/generate conditions                          in....";
    Core::IO::cout.flush();
  }

  //--------------------------------------------- read generic node sets
  const auto get_discretization_callback = [](const std::string& name) -> decltype(auto)
  { return *Global::Problem::instance()->get_dis(name); };

  // read design nodes <-> nodes
  std::vector<std::vector<int>> dnode_fenode;
  Core::IO::read_design(input, "DNODE", dnode_fenode, get_discretization_callback);

  // read design lines <-> nodes
  std::vector<std::vector<int>> dline_fenode;
  Core::IO::read_design(input, "DLINE", dline_fenode, get_discretization_callback);

  // read design surfaces <-> nodes
  std::vector<std::vector<int>> dsurf_fenode;
  Core::IO::read_design(input, "DSURF", dsurf_fenode, get_discretization_callback);

  // read design volumes <-> nodes
  std::vector<std::vector<int>> dvol_fenode;
  Core::IO::read_design(input, "DVOL", dvol_fenode, get_discretization_callback);

  // check for meshfree discretisation to add node set topologies
  std::vector<std::vector<std::vector<int>>*> nodeset(4);
  nodeset[0] = &dnode_fenode;
  nodeset[1] = &dline_fenode;
  nodeset[2] = &dsurf_fenode;
  nodeset[3] = &dvol_fenode;

  // create list of known conditions
  std::shared_ptr<std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>> vc =
      Input::valid_conditions();
  std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist = *vc;

  // test for each condition definition (input file condition section)
  // - read all conditions that match the definition
  // - add the nodal clouds to the conditions
  // - add the conditions to the appropriate discretizations
  //
  // Note that this will reset (un-fill_complete) the discretizations.
  for (auto& condition : condlist)
  {
    std::multimap<int, std::shared_ptr<Core::Conditions::Condition>> cond;

    // read conditions from dat file
    condition->read(input, cond);

    // add nodes to conditions
    std::multimap<int, std::shared_ptr<Core::Conditions::Condition>>::const_iterator curr;
    for (curr = cond.begin(); curr != cond.end(); ++curr)
    {
      switch (curr->second->g_type())
      {
        case Core::Conditions::geometry_type_point:
          if (curr->first < 0 or static_cast<unsigned>(curr->first) >= dnode_fenode.size())
          {
            FOUR_C_THROW(
                "DPoint %d not in range [0:%d[\n"
                "DPoint condition on non existent DPoint?",
                curr->first, dnode_fenode.size());
          }
          curr->second->set_nodes(dnode_fenode[curr->first]);
          break;
        case Core::Conditions::geometry_type_line:
          if (curr->first < 0 or static_cast<unsigned>(curr->first) >= dline_fenode.size())
          {
            FOUR_C_THROW(
                "DLine %d not in range [0:%d[\n"
                "DLine condition on non existent DLine?",
                curr->first, dline_fenode.size());
          }
          curr->second->set_nodes(dline_fenode[curr->first]);
          break;
        case Core::Conditions::geometry_type_surface:
          if (curr->first < 0 or static_cast<unsigned>(curr->first) >= dsurf_fenode.size())
          {
            FOUR_C_THROW(
                "DSurface %d not in range [0:%d[\n"
                "DSurface condition on non existent DSurface?",
                curr->first, dsurf_fenode.size());
          }
          curr->second->set_nodes(dsurf_fenode[curr->first]);
          break;
        case Core::Conditions::geometry_type_volume:
          if (curr->first < 0 or static_cast<unsigned>(curr->first) >= dvol_fenode.size())
          {
            FOUR_C_THROW(
                "DVolume %d not in range [0:%d[\n"
                "DVolume condition on non existent DVolume?",
                curr->first, dvol_fenode.size());
          }
          curr->second->set_nodes(dvol_fenode[curr->first]);
          break;
        default:
          FOUR_C_THROW("geometry type unspecified");
          break;
      }

      // Iterate through all discretizations and sort the appropriate condition
      // into the correct discretization it applies to

      for (const auto& [name, dis] : problem.discretization_range())
      {
        const std::vector<int>* nodes = curr->second->get_nodes();
        if (nodes->size() == 0)
          FOUR_C_THROW("%s condition %d has no nodal cloud", condition->description().c_str(),
              curr->second->id());

        int foundit = 0;
        for (int node : *nodes)
        {
          foundit = dis->have_global_node(node);
          if (foundit) break;
        }
        int found = 0;
        Core::Communication::sum_all(&foundit, &found, 1, dis->get_comm());
        if (found)
        {
          // Insert a copy since we might insert the same condition in many discretizations.
          dis->set_condition(condition->name(), curr->second->copy_without_geometry());
        }
      }
    }
  }

  if (Core::Communication::my_mpi_rank(input.get_comm()) == 0)
  {
    std::cout << time.totalElapsedTime(true) << " secs\n";
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_knots(Global::Problem& problem, Core::IO::InputFile& input)
{
  // get information on the spatial approximation --- we only read knots
  // in the nurbs case
  Core::FE::ShapeFunctionType distype = problem.spatial_approximation_type();

  // Iterate through all discretizations and sort the appropriate condition
  // into the correct discretization it applies to

  for (const auto& [name, dis] : problem.discretization_range())
  {
    if (distype == Core::FE::ShapeFunctionType::nurbs)
    {
      // cast discretisation to nurbs variant to be able
      // to add the knotvector
      auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*dis));

      if (nurbsdis == nullptr)
        FOUR_C_THROW("discretization %s is not a NurbsDiscretization! Panic.", dis->name().c_str());

      // define an empty knot vector object
      std::shared_ptr<Core::FE::Nurbs::Knotvector> disknots = nullptr;

      // read the knotvector data from the input
      Core::IO::read_knots(input, dis->name(), disknots);

      if (disknots == nullptr)
      {
        FOUR_C_THROW("Knotvector read failed in Nurbs discretisation\n");
      }

      // make sure atdis is fillcompleted, to be able to call
      // ElementRowMap() on it
      // do not initialize elements, since this would require knot
      // vector values
      if (!dis->filled())
      {
        dis->fill_complete(false, false, false);
      }

      // the smallest gid in the discretisation determines the access
      // pattern via the element offset
      int smallest_gid_in_dis = dis->element_row_map()->MinAllGID();

      // consistency checks
      disknots->finish_knots(smallest_gid_in_dis);

      // add knots to discretisation
      nurbsdis->set_knot_vector(disknots);
    }
  }  // loop fields
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_particles(Global::Problem& problem, Core::IO::InputFile& input)
{
  // no need to read in particles in case of restart
  if (problem.restart()) return;

  PARTICLEENGINE::read_particles(input, "PARTICLES", problem.particles());
}

FOUR_C_NAMESPACE_CLOSE
