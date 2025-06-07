// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_multiscale.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::setup()
{
  check_init();

  macro_visualization_params_ = Core::IO::visualization_parameters_factory(
      Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
      *Global::Problem::instance()->output_control_file(), global_state().get_time_n());

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::write_restart(
    Core::IO::DiscretizationWriter& iowriter) const
{
  for (const auto& actele : discret().my_row_element_range())
  {
    std::shared_ptr<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->write_restart();
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();

  const int my_pid = Core::Communication::my_mpi_rank(
      Global::Problem::instance()->get_dis("structure")->get_comm());
  for (const auto& actele : discret().my_col_element_range())
  {
    std::shared_ptr<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      auto* micro = dynamic_cast<Mat::MicroMaterial*>(mat.get());
      const int eleID = actele->id();
      const bool eleowner = my_pid == actele->owner();

      Discret::Elements::Solid* solidele = dynamic_cast<Discret::Elements::Solid*>(actele);
      const int numGaussPoints = solidele->get_gauss_rule().num_points();

      for (int gp = 0; gp < numGaussPoints; ++gp) micro->read_restart(gp, eleID, eleowner);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::post_setup()
{
  for (const auto& actele : discret().my_col_element_range())
  {
    std::shared_ptr<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->post_setup();
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::runtime_pre_output_step_state()
{
  for (const auto& actele : discret().my_row_element_range())
  {
    std::shared_ptr<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      const auto* micro_mat = static_cast<Mat::MicroMaterial*>(mat.get());
      micro_mat->runtime_pre_output_step_state();
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::runtime_output_step_state() const
{
  std::pair<double, int> output_macro_time_and_step;
  if (macro_visualization_params_.every_iteration_ == true)
  {
    output_macro_time_and_step =
        Core::IO::get_time_and_time_step_index_for_output(macro_visualization_params_,
            global_state().get_time_n(), global_state().get_step_n(), eval_data().get_nln_iter());
  }
  else
  {
    output_macro_time_and_step = Core::IO::get_time_and_time_step_index_for_output(
        macro_visualization_params_, global_state().get_time_n(), global_state().get_step_n());
  }

  for (const auto& actele : discret().my_row_element_range())
  {
    std::shared_ptr<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      auto* micro_mat = static_cast<Mat::MicroMaterial*>(mat.get());
      micro_mat->runtime_output_step_state(output_macro_time_and_step);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::post_time_loop()
{
  // stop supporting processors in multi scale simulations
  MultiScale::stop_np_multiscale();
}

FOUR_C_NAMESPACE_CLOSE
