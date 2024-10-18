/*-----------------------------------------------------------*/
/*! \file

\brief Handle multiscale relevant aspects


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_model_evaluator_multiscale.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_stru_multi_microstatic.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::setup()
{
  check_init();

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
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
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

  const int my_pid = Global::Problem::instance()->get_dis("structure")->get_comm().MyPID();
  for (const auto& actele : discret().my_col_element_range())
  {
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      auto* micro = dynamic_cast<Mat::MicroMaterial*>(mat.get());
      int eleID = actele->id();
      const bool eleowner = my_pid == actele->owner();

      Discret::ELEMENTS::Solid* solidele = dynamic_cast<Discret::ELEMENTS::Solid*>(actele);
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
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->post_setup();
    }
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::determine_optional_quantity()
{
  // TODO: Move to RuntimePreOutputStepState
  for (const auto& actele : discret().my_row_element_range())
  {
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->prepare_output();
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Multiscale::output_step_state(
    Core::IO::DiscretizationWriter& iowriter) const
{
  // TODO: Move to RuntimeOutputStepState
  for (const auto& actele : discret().my_row_element_range())
  {
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->output_step_state();
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
