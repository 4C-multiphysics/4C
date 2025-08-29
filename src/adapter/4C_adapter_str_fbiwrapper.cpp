// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_fbiwrapper.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_fsi_str_model_evaluator_partitioned.hpp"
#include "4C_global_data.hpp"
#include "4C_structure_aux.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtk_output.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::FBIStructureWrapper::FBIStructureWrapper(std::shared_ptr<Structure> structure)
    : FSIStructureWrapper(structure)
{
  const bool is_prestress = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
                                Global::Problem::instance()->structural_dynamic_params(),
                                "PRESTRESS") != Inpar::Solid::PreStress::none;
  if (is_prestress)
  {
    FOUR_C_THROW("Prestressing for fluid-beam interaction not tested yet.");
  }
  eletypeextractor_ = std::make_shared<BeamInteraction::Utils::MapExtractor>();
  BeamInteraction::Utils::setup_ele_type_map_extractor(
      *structure_->discretization(), *eletypeextractor_);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FBIStructureWrapper::extract_interface_veln()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> veli =
      std::make_shared<Core::LinAlg::Vector<double>>(veln()->get_map());
  veli->update(1.0, *veln(), 0.0);
  return veli;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIStructureWrapper::extract_interface_velnp()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> veli =
      std::make_shared<Core::LinAlg::Vector<double>>(velnp()->get_map());
  veli->update(1.0, *velnp(), 0.0);
  return veli;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIStructureWrapper::predict_interface_velnp()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> veli =
      std::make_shared<Core::LinAlg::Vector<double>>(veln()->get_map());
  veli->update(1.0, *veln(), 0.0);
  return veli;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FBIStructureWrapper::relaxation_solve(
    std::shared_ptr<Core::LinAlg::Vector<double>> iforce)
{
  FOUR_C_THROW("RelaxationSolve not implemented for immersed fluid-beam interaction\n");
  return nullptr;
}
/*------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------*/
void Adapter::FBIStructureWrapper::rebuild_interface() { FOUR_C_THROW("Not implemented yet"); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIStructureWrapper::predict_interface_dispnp()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> disi =
      std::make_shared<Core::LinAlg::Vector<double>>(dispn()->get_map());
  disi->update(1.0, *dispnp(), 0.0);
  return disi;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIStructureWrapper::extract_interface_dispnp()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> disi =
      std::make_shared<Core::LinAlg::Vector<double>>(dispnp()->get_map());
  disi->update(1.0, *dispnp(), 0.0);
  return disi;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FBIStructureWrapper::extract_interface_dispn()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> disi =
      std::make_shared<Core::LinAlg::Vector<double>>(dispn()->get_map());
  disi->update(1.0, *dispn(), 0.0);
  return disi;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
// Apply interface forces
void Adapter::FBIStructureWrapper::apply_interface_forces(
    std::shared_ptr<Core::LinAlg::Vector<double>> iforce)
{
  fsi_model_evaluator()->get_interface_force_np_ptr()->update(
      1.0, *iforce, 0.0);  // todo This has to be changed for mixed structure
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIStructureWrapper::setup_multi_map_extractor()
{
  fsi_model_evaluator()->setup_multi_map_extractor();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

std::shared_ptr<const Solid::TimeInt::ParamsRuntimeOutput>
Adapter::FBIStructureWrapper::get_io_data()
{
  return fsi_model_evaluator()->get_in_output().get_runtime_output_params();
}

FOUR_C_NAMESPACE_CLOSE
