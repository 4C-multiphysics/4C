// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_phasemanager.hpp"

#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_multiphase_reactions.hpp"
#include "4C_mat_fluidporo_multiphase_singlereaction.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_porofluid_pressure_based_ele_calc_utils.hpp"
#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_SerialDenseSolver.hpp>

#include <numeric>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 | factory method                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface>
Discret::Elements::PoroFluidManager::PhaseManagerInterface::create_phase_manager(
    const Discret::Elements::PoroFluidMultiPhaseEleParameter& para, int nsd,
    Core::Materials::MaterialType mattype, const PoroPressureBased::Action& action,
    int totalnumdofpernode, int numfluidphases, int numvolfrac)
{
  std::shared_ptr<PhaseManagerInterface> phasemanager = nullptr;

  // build the standard phase manager
  phasemanager = std::make_shared<PhaseManagerCore>(totalnumdofpernode, numfluidphases, numvolfrac);

  return wrap_phase_manager(para, nsd, mattype, action, phasemanager);
}

/*----------------------------------------------------------------------*
 | factory method                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface>
Discret::Elements::PoroFluidManager::PhaseManagerInterface::wrap_phase_manager(
    const Discret::Elements::PoroFluidMultiPhaseEleParameter& para, int nsd,
    Core::Materials::MaterialType mattype, const PoroPressureBased::Action& action,
    std::shared_ptr<PhaseManagerInterface> corephasemanager)
{
  std::shared_ptr<PhaseManagerInterface> phasemanager = nullptr;

  // determine action
  switch (action)
  {
    // calculate true pressures and saturation
    case PoroPressureBased::calc_pres_and_sat:
    case PoroPressureBased::calc_valid_dofs:
    {
      // no extensions needed
      phasemanager = corephasemanager;
      break;
    }
    // calculate solid pressure
    case PoroPressureBased::calc_solidpressure:
    {
      // we have volume fractions --> we need PhaseManagerDerivAndPorosity because solid pressure is
      // calculated differently
      if (corephasemanager->total_num_dof() > corephasemanager->num_fluid_phases() &&
          corephasemanager->num_fluid_phases() > 0)
      {
        // porosity (includes derivatves) needed
        phasemanager = std::make_shared<PhaseManagerDerivAndPorosity>(corephasemanager);
      }
      else
      {
        // no extensions needed
        phasemanager = corephasemanager;
      }
      break;
    }
    case PoroPressureBased::calc_phase_velocities:
    {
      // derivatives needed
      if (corephasemanager->num_fluid_phases() > 0)
        phasemanager = std::make_shared<PhaseManagerDerivAndPorosity>(corephasemanager);
      else
        phasemanager = corephasemanager;
      // enhance by diffusion tensor
      switch (nsd)
      {
        case 1:
          phasemanager = std::make_shared<PhaseManagerDiffusion<1>>(phasemanager);
          break;
        case 2:
          phasemanager = std::make_shared<PhaseManagerDiffusion<2>>(phasemanager);
          break;
        case 3:
          phasemanager = std::make_shared<PhaseManagerDiffusion<3>>(phasemanager);
          break;
        default:
          FOUR_C_THROW("invalid dimension for creating phase manager!");
      }
      break;
    }
    case PoroPressureBased::recon_flux_at_nodes:
    {
      // derivatives needed
      if (corephasemanager->num_fluid_phases() > 0)
        phasemanager = std::make_shared<PhaseManagerDeriv>(corephasemanager);
      else
        phasemanager = corephasemanager;
      // enhance by diffusion tensor
      switch (nsd)
      {
        case 1:
          phasemanager = std::make_shared<PhaseManagerDiffusion<1>>(phasemanager);
          break;
        case 2:
          phasemanager = std::make_shared<PhaseManagerDiffusion<2>>(phasemanager);
          break;
        case 3:
          phasemanager = std::make_shared<PhaseManagerDiffusion<3>>(phasemanager);
          break;
        default:
          FOUR_C_THROW("invalid dimension for creating phase manager!");
      }
      break;
    }
    // standard evaluate call
    case PoroPressureBased::calc_mat_and_rhs:
    case PoroPressureBased::calc_initial_time_deriv:
    case PoroPressureBased::calc_fluid_struct_coupl_mat:
    case PoroPressureBased::calc_fluid_scatra_coupl_mat:
    case PoroPressureBased::calc_domain_integrals:
    {
      // porosity (includes derivatves) needed
      phasemanager = std::make_shared<PhaseManagerDerivAndPorosity>(corephasemanager);

      // enhance by diffusion tensor
      switch (nsd)
      {
        case 1:
          phasemanager = std::make_shared<PhaseManagerDiffusion<1>>(phasemanager);
          break;
        case 2:
          phasemanager = std::make_shared<PhaseManagerDiffusion<2>>(phasemanager);
          break;
        case 3:
          phasemanager = std::make_shared<PhaseManagerDiffusion<3>>(phasemanager);
          break;
        default:
          FOUR_C_THROW("invalid dimension for creating phase manager!");
      }

      if (mattype == Core::Materials::m_fluidporo_multiphase_reactions)
        // enhance by scalar handling capability
        phasemanager = std::make_shared<PhaseManagerReaction>(phasemanager);

      if (corephasemanager->total_num_dof() >
          corephasemanager->num_fluid_phases() + corephasemanager->num_vol_frac())
      {
        switch (nsd)
        {
          case 1:
            phasemanager = std::make_shared<PhaseManagerVolFrac<1>>(phasemanager);
            break;
          case 2:
            phasemanager = std::make_shared<PhaseManagerVolFrac<2>>(phasemanager);
            break;
          case 3:
            phasemanager = std::make_shared<PhaseManagerVolFrac<3>>(phasemanager);
            break;
          default:
            FOUR_C_THROW("invalid dimension for creating phase manager!");
        }
      }

      break;
    }
    case PoroPressureBased::get_access_from_scatra:
    {
      // porosity (includes derivatives) needed
      phasemanager = std::make_shared<PhaseManagerDerivAndPorosity>(corephasemanager);

      // enhance by diffusion tensor
      switch (nsd)
      {
        case 1:
          phasemanager = std::make_shared<PhaseManagerDiffusion<1>>(phasemanager);
          break;
        case 2:
          phasemanager = std::make_shared<PhaseManagerDiffusion<2>>(phasemanager);
          break;
        case 3:
          phasemanager = std::make_shared<PhaseManagerDiffusion<3>>(phasemanager);
          break;
        default:
          FOUR_C_THROW("invalid dimension for creating phase manager!");
      }

      break;
    }
    case PoroPressureBased::calc_porosity:
    case PoroPressureBased::get_access_from_artcoupling:
    case PoroPressureBased::calc_volfrac_blood_lung:
    case PoroPressureBased::calc_determinant_of_deformationgradient:
    {
      // porosity (includes derivatves) needed
      phasemanager = std::make_shared<PhaseManagerDerivAndPorosity>(corephasemanager);
      break;
    }
    default:
      FOUR_C_THROW("unknown action {} for creating the phase manager!", action);
      break;
  }  // switch(action)


  return phasemanager;
}

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::PoroFluidManager::PhaseManagerCore::PhaseManagerCore(
    int totalnumdofpernode, int numfluidphases, int numvolfrac)
    : totalnumdofpernode_(totalnumdofpernode),
      numfluidphases_(numfluidphases),
      numvolfrac_(numvolfrac),
      genpressure_(numfluidphases, 0.0),
      volfrac_(numvolfrac_, 0.0),
      volfracpressure_(numvolfrac_, 0.0),
      sumaddvolfrac_(0.0),
      pressure_(numfluidphases, 0.0),
      saturation_(numfluidphases, 0.0),
      density_(numfluidphases, 0.0),
      soliddensity_(0.0),
      solidpressure_(0.0),
      invbulkmodulifluid_(numfluidphases, 0.0),
      invbulkmodulussolid_(0.0),
      ele_(nullptr),
      isevaluated_(false),
      issetup_(false),
      initialvolfrac_(0.0),
      volfrac_deformationbased_scaling_parameter_deformation_(0.0),
      volfrac_deformationbased_scaling_parameter_pressure_(0.0)
{
}

/*----------------------------------------------------------------------*
 | copy constructor                                          vuong 08/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::PoroFluidManager::PhaseManagerCore::PhaseManagerCore(const PhaseManagerCore& old)
    : totalnumdofpernode_(old.totalnumdofpernode_),
      numfluidphases_(old.numfluidphases_),
      numvolfrac_(old.numvolfrac_),
      genpressure_(old.genpressure_),
      volfrac_(old.volfrac_),
      volfracpressure_(old.volfracpressure_),
      sumaddvolfrac_(old.sumaddvolfrac_),
      pressure_(old.pressure_),
      saturation_(old.saturation_),
      density_(old.density_),
      soliddensity_(old.soliddensity_),
      solidpressure_(old.solidpressure_),
      invbulkmodulifluid_(old.invbulkmodulifluid_),
      invbulkmodulussolid_(old.invbulkmodulussolid_),
      ele_(nullptr),
      isevaluated_(old.isevaluated_),
      issetup_(old.issetup_),
      initialvolfrac_(old.initialvolfrac_),
      volfrac_deformationbased_scaling_parameter_deformation_(
          old.volfrac_deformationbased_scaling_parameter_deformation_),
      volfrac_deformationbased_scaling_parameter_pressure_(
          old.volfrac_deformationbased_scaling_parameter_pressure_)
{
  return;
}

/*----------------------------------------------------------------------*
 | setup                                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerCore::setup(
    const Core::Elements::Element* ele, const int matnum)
{
  FOUR_C_ASSERT(ele != nullptr, "Element is null pointer for setup of phase manager!");
  // save current element
  ele_ = ele;
  // get material
  const Core::Mat::Material& material = *(ele_->material(matnum));

  // check the material
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase and poro multiphase reactions material valid");

  // cast
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  // safety check
  if (numfluidphases_ != multiphasemat.num_fluid_phases() ||
      totalnumdofpernode_ != multiphasemat.num_mat() || numvolfrac_ != multiphasemat.num_vol_frac())
    FOUR_C_THROW(
        "Number of phases given by the poro multiphase material does not match number "
        "of DOFs ({} phases and {} Fluid DOFs, {} vol fracs and {} volfracs, {} total dofs and {} "
        "Total DOFs)!",
        numfluidphases_, multiphasemat.num_fluid_phases(), numvolfrac_,
        multiphasemat.num_vol_frac(), totalnumdofpernode_, multiphasemat.num_mat());

  density_.resize(numfluidphases_, 0.0);
  volfracdensity_.resize(numvolfrac_, 0.0);
  invbulkmodulifluid_.resize(numfluidphases_, 0.0);

  // access second material in structure element
  FOUR_C_ASSERT(ele_->num_material() > 1, "no second material defined for element ");
  FOUR_C_ASSERT(ele_->material(1)->material_type() == Core::Materials::m_structporo,
      "invalid structure material for poroelasticity");

  // cast second material to poro material
  std::shared_ptr<Mat::StructPoro> structmat =
      std::static_pointer_cast<Mat::StructPoro>(ele_->material(1));

  invbulkmodulussolid_ = structmat->inv_bulk_modulus();
  soliddensity_ = structmat->density_solid_phase();

  for (int iphase = 0; iphase < numfluidphases_; iphase++)
  {
    // get the single phase material
    const Mat::FluidPoroSinglePhase& singlephasemat =
        PoroPressureBased::ElementUtils::get_single_phase_mat_from_material(material, iphase);
    invbulkmodulifluid_[iphase] = singlephasemat.inv_bulkmodulus();

    // get density
    density_[iphase] = singlephasemat.density();
  }

  for (int ivolfrac = 0; ivolfrac < numvolfrac_; ivolfrac++)
  {
    if (totalnumdofpernode_ - numfluidphases_ == numvolfrac_ * 2)
    {
      // get the single phase material
      const Mat::FluidPoroSingleVolFrac& singlevolfracmat =
          PoroPressureBased::ElementUtils::get_single_vol_frac_mat_from_material(
              material, ivolfrac + numfluidphases_);

      // get constant values
      volfracdensity_[ivolfrac] = singlevolfracmat.density();
    }
    // is volfrac for blood lung
    else if (totalnumdofpernode_ - numfluidphases_ == numvolfrac_)
    {
      // get the single phase material
      const Mat::FluidPoroVolFracPressureBloodLung& singlevolfracbloodlungmat = PoroPressureBased::
          ElementUtils::get_single_vol_frac_pressure_blood_lung_mat_from_material(
              material, ivolfrac + numfluidphases_);

      // get constant values
      volfracdensity_[ivolfrac] = singlevolfracbloodlungmat.density();
      initialvolfrac_ = singlevolfracbloodlungmat.initial_volfrac();
      volfrac_deformationbased_scaling_parameter_deformation_ =
          singlevolfracbloodlungmat.scaling_parameter_deformation();
      volfrac_deformationbased_scaling_parameter_pressure_ =
          singlevolfracbloodlungmat.scaling_parameter_pressure();
    }
    else
      FOUR_C_THROW("You have an invalid number of volume fractions in the phasemanager!");
  }

  issetup_ = true;
  return;
}

/*----------------------------------------------------------------------*
 | evaluate pressures, saturations and derivatives at GP     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerCore::evaluate_gp_state(
    double J, const VariableManagerMinAccess& varmanager, const int matnum)
{
  check_is_setup();

  if (isevaluated_ == true) FOUR_C_THROW("state has already been set!");

  // get material
  FOUR_C_ASSERT(ele_->material(matnum) != nullptr, "Material of element is null pointer!");
  const Core::Mat::Material& material = *(ele_->material(matnum));

  // access state vector
  const std::vector<double>& phinp = *varmanager.phinp();

  if (totalnumdofpernode_ != (int)phinp.size())
    FOUR_C_THROW("Length of phinp vector is not equal to the number of dofs");

  // fluid primary variables at phinp[0 ... numfluidphases - 1]
  const std::vector<double> fluid_phinp(phinp.data(), phinp.data() + numfluidphases_);

  std::vector<double> volfrac(numvolfrac_, 0.0);
  std::vector<double> volfracpressure(numvolfrac_, 0.0);

  if (numvolfrac_)
  {
    if (totalnumdofpernode_ == numfluidphases_ + numvolfrac_ * 2)
    {
      // volume fractions at phinp[numfluidphases ... numfluidphases - 1 + numvolfrac]
      volfrac.assign(phinp.data() + numfluidphases_, phinp.data() + numfluidphases_ + numvolfrac_);
      // volume fraction pressures at phinp[numfluidphases + numvolfrac ... numfluidphases - 1 +
      // 2*numvolfrac]
      volfracpressure.assign(phinp.data() + numfluidphases_ + numvolfrac_,
          phinp.data() + numfluidphases_ + numvolfrac_ + numvolfrac_);
    }
    else if (totalnumdofpernode_ == numfluidphases_ + numvolfrac_)
    {
      // volume fraction pressures at phinp[numfluidphases ... numfluidphases - 1 + numvolfrac]
      volfracpressure.assign(
          phinp.data() + numfluidphases_, phinp.data() + numfluidphases_ + numvolfrac_);

      // not a primary variable anymore
      if (fluid_phinp[0] / volfracpressure[0] > 1.0)
      {
        // sofar only one volfrac possible
        volfrac[0] =
            (initialvolfrac_ * pow(J, volfrac_deformationbased_scaling_parameter_deformation_) *
                pow(fluid_phinp[0] / volfracpressure[0],
                    volfrac_deformationbased_scaling_parameter_pressure_));
      }
      else
      {
        volfrac[0] =
            (initialvolfrac_ * pow(J, volfrac_deformationbased_scaling_parameter_deformation_));
      }
    }
    else
      FOUR_C_THROW("You have an invalid number of volume fractions in the phasemanager!");
  }

  if (numfluidphases_ != (int)fluid_phinp.size())
    FOUR_C_THROW("Length of fluid-phinp vector is not equal to the number of phases");

  // cast the material to multiphase material
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  // resize all vectors
  genpressure_.resize(numfluidphases_, 0.0);
  volfrac_.resize(numvolfrac_, 0.0);
  volfracpressure_.resize(numvolfrac_, 0.0);
  pressure_.resize(numfluidphases_, 0.0);
  saturation_.resize(numfluidphases_, 0.0);

  // evaluate the volume fractions
  volfrac_ = volfrac;
  if (numvolfrac_ != (int)volfrac.size())
    FOUR_C_THROW("Length of volfrac vector is not equal to the number of volume fractions");
  sumaddvolfrac_ = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac_; ivolfrac++) sumaddvolfrac_ += volfrac_[ivolfrac];
  volfracpressure_ = volfracpressure;
  if (numvolfrac_ != (int)volfracpressure.size())
    FOUR_C_THROW(
        "Length of volfrac pressure vector is not equal to the number of volume fractions");
  // evaluate the pressures
  multiphasemat.evaluate_gen_pressure(genpressure_, fluid_phinp);

  //! transform generalized pressures to true pressure values
  multiphasemat.transform_gen_pres_to_true_pres(genpressure_, pressure_);

  // explicit evaluation of saturation
  if (numfluidphases_ > 0) multiphasemat.evaluate_saturation(saturation_, fluid_phinp, pressure_);

  // solid pressure = sum (S_i*p_i)
  solidpressure_ =
      std::inner_product(saturation_.begin(), saturation_.end(), pressure_.begin(), 0.0);

  // done

  isevaluated_ = true;

  return;
}

/*----------------------------------------------------------------------*
 | zero all values at GP                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerCore::clear_gp_state()
{
  // this is just a safety call. All quantities should be recomputed
  // in the next EvaluateGPState call anyway, but you never know ...

  // zero everything
  std::fill(genpressure_.begin(), genpressure_.end(), 0.0);
  std::fill(volfrac_.begin(), volfrac_.end(), 0.0);
  std::fill(volfracpressure_.begin(), volfracpressure_.end(), 0.0);
  std::fill(pressure_.begin(), pressure_.end(), 0.0);
  std::fill(saturation_.begin(), saturation_.end(), 0.0);
  solidpressure_ = 0.0;
  sumaddvolfrac_ = 0.0;
  std::vector<double> zero(pressure_.size(), 0.0);

  // states are reset
  isevaluated_ = false;

  return;
}

/*----------------------------------------------------------------------*
 * get solid pressure                                        vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::solid_pressure() const
{
  check_is_evaluated();

  return solidpressure_;
}

/*----------------------------------------------------------------------*
 * recalculate solid pressure if volfracs are present  kremheller 09/17 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerCore::recalculate_solid_pressure(
    const double porosity)
{
  check_is_evaluated();

  // p_s = (porosity)/(porosity+sumaddvolfrac) * sum_i^numfluidphases (S_i*p_i)
  //     + 1.0 / (porosity+sumaddvolfrac) sum_i=1^numvolfrac (volfrac_i*pressure_i)
  // with porosity = porosity' - sumaddvolfrac
  // porosity' := porosity if no volume fractions were present
  solidpressure_ *= porosity / (sumaddvolfrac_ + porosity);
  for (int ivolfrac = 0; ivolfrac < numvolfrac_; ivolfrac++)
    solidpressure_ += volfrac_[ivolfrac] / (porosity + sumaddvolfrac_) * volfracpressure_[ivolfrac];
}

/*----------------------------------------------------------------------*
 *    get saturation of phase                               vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::saturation(int phasenum) const
{
  check_is_evaluated();

  return saturation_.at(phasenum);
}

/*----------------------------------------------------------------------*
 *  get pressure of phase                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::pressure(int phasenum) const
{
  check_is_evaluated();

  return pressure_.at(phasenum);
}

/*----------------------------------------------------------------------*
 *    get saturation                                        vuong 08/16 |
 *----------------------------------------------------------------------*/
const std::vector<double>& Discret::Elements::PoroFluidManager::PhaseManagerCore::saturation() const
{
  check_is_evaluated();

  return saturation_;
}

/*----------------------------------------------------------------------*
 *    get volfracs                                     kremheller 08/17 |
 *----------------------------------------------------------------------*/
const std::vector<double>& Discret::Elements::PoroFluidManager::PhaseManagerCore::vol_frac() const
{
  check_is_evaluated();

  return volfrac_;
}

/*----------------------------------------------------------------------*
 *    get volfrac of volfrac 'volfracnum'              kremheller 08/17 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::vol_frac(int volfracnum) const
{
  check_is_evaluated();

  return volfrac_.at(volfracnum);
}

/*----------------------------------------------------------------------*
 *  get pressures for volfracs                         kremheller 02/18 |
 *----------------------------------------------------------------------*/
const std::vector<double>&
Discret::Elements::PoroFluidManager::PhaseManagerCore::vol_frac_pressure() const
{
  check_is_evaluated();

  return volfracpressure_;
}

/*---------------------------------------------------------------------------*
 * get pressures for volume fractions                       kremheller 02/18 |
 *---------------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::vol_frac_pressure(
    int volfracnum) const
{
  check_is_evaluated();

  return volfracpressure_.at(volfracnum);
}

/*----------------------------------------------------------------------*
 * get the sum of the additional volume fractions      kremheller 09/17 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::sum_add_vol_frac() const
{
  check_is_evaluated();

  return sumaddvolfrac_;
}

/*----------------------------------------------------------------------*
 *  get pressure                                            vuong 08/16 |
 *----------------------------------------------------------------------*/
const std::vector<double>& Discret::Elements::PoroFluidManager::PhaseManagerCore::pressure() const
{
  check_is_evaluated();

  return pressure_;
}

/*----------------------------------------------------------------------*
 *   get bulk modulus of phase 'phasenum'                    vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::inv_bulkmodulus(int phasenum) const
{
  check_is_evaluated();

  return invbulkmodulifluid_.at(phasenum);
}

/*----------------------------------------------------------------------*
 *  check if fluid phase 'phasenum' is incompressible  kremheller 06/17 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::PoroFluidManager::PhaseManagerCore::incompressible_fluid_phase(
    int phasenum) const
{
  check_is_evaluated();

  return invbulkmodulifluid_.at(phasenum) < 1.0e-14;
}

/*----------------------------------------------------------------------*
 *   get inverse bulk modulus of solid phase                 vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::inv_bulkmodulus_solid() const
{
  check_is_evaluated();

  return invbulkmodulussolid_;
}

/*----------------------------------------------------------------------*
 *  check if solid is incompressible                   kremheller 06/17 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::PoroFluidManager::PhaseManagerCore::incompressible_solid() const
{
  check_is_evaluated();

  return invbulkmodulussolid_ < 1.0e-14;
}

/*----------------------------------------------------------------------*
 *   get density of phase 'phasenum'                    vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::density(int phasenum) const
{
  check_is_setup();

  return density_.at(phasenum);
}

/*---------------------------------------------------------------------------*
 * get densities for volume fractions                       kremheller 08/17 |
 *---------------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::vol_frac_density(int volfracnum) const
{
  check_is_setup();

  return volfracdensity_.at(volfracnum);
}

/*---------------------------------------------------------------------------*
 * get densities of solid phase                                 wirthl 12/18 |
 *---------------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerCore::solid_density() const
{
  check_is_evaluated();

  return soliddensity_;
}

double Discret::Elements::PoroFluidManager::PhaseManagerCore::initial_volfrac() const
{
  check_is_evaluated();

  return initialvolfrac_;
}

double Discret::Elements::PoroFluidManager::PhaseManagerCore::
    volfrac_blood_lung_parameter_deformation_dependence() const
{
  check_is_evaluated();

  return volfrac_deformationbased_scaling_parameter_deformation_;
}

double Discret::Elements::PoroFluidManager::PhaseManagerCore::
    volfrac_blood_lung_parameter_pressure_dependence() const
{
  check_is_evaluated();

  return volfrac_deformationbased_scaling_parameter_pressure_;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::PoroFluidManager::PhaseManagerDeriv::PhaseManagerDeriv(
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager)
    : PhaseManagerDecorator(phasemanager),
      pressurederiv_(nullptr),
      saturationderiv_(nullptr),
      saturationderivderiv_(nullptr),
      solidpressurederiv_(nullptr),
      solidpressurederivderiv_(nullptr)
{
  const int numfluidphases = phasemanager_->num_fluid_phases();
  // initialize matrixes and vectors
  pressurederiv_ =
      std::make_shared<Core::LinAlg::SerialDenseMatrix>(numfluidphases, numfluidphases);
  saturationderiv_ =
      std::make_shared<Core::LinAlg::SerialDenseMatrix>(numfluidphases, numfluidphases);
  saturationderivderiv_ = std::make_shared<std::vector<Core::LinAlg::SerialDenseMatrix>>(
      numfluidphases, Core::LinAlg::SerialDenseMatrix(numfluidphases, numfluidphases));
  solidpressurederiv_ = std::make_shared<Core::LinAlg::SerialDenseVector>(numfluidphases);
  solidpressurederivderiv_ =
      std::make_shared<Core::LinAlg::SerialDenseMatrix>(numfluidphases, numfluidphases);

  return;
}

/*----------------------------------------------------------------------*
 | evaluate pressures, saturations and derivatives at GP     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerDeriv::evaluate_gp_state(
    double J, const VariableManagerMinAccess& varmanager, const int matnum)
{
  // evaluate wrapped phase manager
  phasemanager_->evaluate_gp_state(J, varmanager, matnum);

  // get number of phases
  const int numfluidphases = phasemanager_->num_fluid_phases();
  // if we do not have any fluid phases we do not need any derivatives
  if (numfluidphases == 0) return;

  // get material
  FOUR_C_ASSERT(phasemanager_->element()->material(matnum) != nullptr,
      "Material of element is null pointer!");
  const Core::Mat::Material& material = *(phasemanager_->element()->material(matnum));

  // get pressure
  const std::vector<double>& pressure = phasemanager_->pressure();
  // get saturation
  const std::vector<double>& saturation = phasemanager_->saturation();

  // access state vector
  const std::vector<double>& phinp = *varmanager.phinp();

  if (phasemanager_->total_num_dof() != (int)phinp.size())
    FOUR_C_THROW("Length of phinp vector is not equal to the number of dofs");

  // get fluid primary variable
  const std::vector<double> fluid_phinp(phinp.data(), phinp.data() + numfluidphases);

  if (numfluidphases != (int)fluid_phinp.size())
    FOUR_C_THROW("Length of phinp vector is not equal to the number of phases");

  // cast
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  // calculate the derivative of the pressure (actually first its inverse)
  multiphasemat.evaluate_deriv_of_dof_wrt_pressure(*pressurederiv_, fluid_phinp);

  // now invert the derivatives of the dofs w.r.t. pressure to get the derivatives
  // of the pressure w.r.t. the dofs
  {
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverse;
    inverse.setMatrix(Teuchos::rcpFromRef(pressurederiv_->base()));
    int err = inverse.invert();
    if (err != 0)
      FOUR_C_THROW("Inversion of matrix for pressure derivative failed with error code {}.", err);
  }

  // calculate derivatives of saturation w.r.t. pressure
  Core::LinAlg::SerialDenseMatrix deriv(numfluidphases, numfluidphases);
  multiphasemat.evaluate_deriv_of_saturation_wrt_pressure(deriv, pressure);

  // chain rule: the derivative of saturation w.r.t. dof =
  // (derivative of saturation w.r.t. pressure) * (derivative of pressure w.r.t. dof)
  Core::LinAlg::multiply(*saturationderiv_, deriv, *pressurederiv_);

  // calculate 2nd derivatives of saturation w.r.t. pressure
  // TODO: this should work for pressure und diffpressure DOFs, however not for
  //       saturation DOFs
  std::vector<Core::LinAlg::SerialDenseMatrix> dummyderiv(
      numfluidphases, Core::LinAlg::SerialDenseMatrix(numfluidphases, numfluidphases));
  multiphasemat.evaluate_second_deriv_of_saturation_wrt_pressure(dummyderiv, pressure);
  for (int i = 0; i < numfluidphases; i++)
  {
    Core::LinAlg::multiply_tn(deriv, *pressurederiv_, (dummyderiv)[i]);
    Core::LinAlg::multiply((*saturationderivderiv_)[i], deriv, *pressurederiv_);
  }

  // compute derivative of solid pressure w.r.t. dofs with product rule
  solidpressurederiv_->putScalar(0.0);
  for (int iphase = 0; iphase < numfluidphases; iphase++)
    for (int jphase = 0; jphase < numfluidphases; jphase++)
      (*solidpressurederiv_)(iphase) += (*pressurederiv_)(jphase, iphase) * saturation[jphase] +
                                        (*saturationderiv_)(jphase, iphase) * pressure[jphase];

  // compute second derivative of solid pressure w.r.t. dofs with product rule
  // TODO also include second derivs of pressure and saturation
  solidpressurederivderiv_->putScalar(0.0);
  for (int iphase = 0; iphase < numfluidphases; iphase++)
    for (int jphase = 0; jphase < numfluidphases; jphase++)
      for (int kphase = 0; kphase < numfluidphases; kphase++)
        (*solidpressurederivderiv_)(jphase, kphase) +=
            (*pressurederiv_)(iphase, kphase) * (*saturationderiv_)(iphase, jphase) +
            (*saturationderiv_)(iphase, kphase) * (*pressurederiv_)(iphase, jphase);

  return;
}

/*----------------------------------------------------------------------*
 | zero all values at GP                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerDeriv::clear_gp_state()
{
  // this is just a safety call. All quantities should be recomputed
  // in the next EvaluateGPState call anyway, but you never know ...

  const int numfluidphases = phasemanager_->num_fluid_phases();
  phasemanager_->clear_gp_state();

  // zero everything
  pressurederiv_->putScalar(0.0);
  saturationderiv_->putScalar(0.0);
  for (int iphase = 0; iphase < numfluidphases; iphase++)
    (*saturationderivderiv_)[iphase].putScalar(0.0);
  solidpressurederiv_->putScalar(0.0);
  solidpressurederivderiv_->putScalar(0.0);


  return;
}

/*----------------------------------------------------------------------*
 *  get derivative of saturation of phase                   vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDeriv::saturation_deriv(
    int phasenum, int doftoderive) const
{
  phasemanager_->check_is_evaluated();

  return (*saturationderiv_)(phasenum, doftoderive);
}

/*----------------------------------------------------------------------*
 *  get 2nd derivative of saturation of phase          kremheller 05/17 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDeriv::saturation_deriv_deriv(
    int phasenum, int firstdoftoderive, int seconddoftoderive) const
{
  phasemanager_->check_is_evaluated();

  return (*saturationderivderiv_)[phasenum](firstdoftoderive, seconddoftoderive);
}

/*----------------------------------------------------------------------*
 *  get derivative of pressure of phase                      vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDeriv::pressure_deriv(
    int phasenum, int doftoderive) const
{
  phasemanager_->check_is_evaluated();

  return (*pressurederiv_)(phasenum, doftoderive);
}

/*----------------------------------------------------------------------*
 * get derivative of solid pressure                          vuong 08/16 |
 *-----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDeriv::solid_pressure_deriv(
    int doftoderive) const
{
  phasemanager_->check_is_evaluated();

  return (*solidpressurederiv_)(doftoderive);
}

/*---------------------------------------------------------------------------*
 * get second derivative of solid pressure                       vuong 08/16 |
 *---------------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDeriv::solid_pressure_deriv_deriv(
    int doftoderive, int doftoderive2) const
{
  phasemanager_->check_is_evaluated();

  return (*solidpressurederivderiv_)(doftoderive, doftoderive2);
}


/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::PhaseManagerDerivAndPorosity(
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager)
    : PhaseManagerDeriv(phasemanager),
      porosity_(0.0),
      j_(0.0),
      dporosity_dj_(0.0),
      dporosity_dp_(0.0),
      porosityderiv_(nullptr)
{
  const int totalnumdof = phasemanager_->total_num_dof();
  // initialize matrixes and vectors
  porosityderiv_ = std::make_shared<Core::LinAlg::SerialDenseVector>(totalnumdof);

  return;
}

/*----------------------------------------------------------------------*
 | evaluate pressures, saturations and derivatives at GP     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::evaluate_gp_state(
    double J, const VariableManagerMinAccess& varmanager, const int matnum)
{
  // evaluate base class
  PhaseManagerDeriv::evaluate_gp_state(J, varmanager, matnum);

  // access second material in structure element
  FOUR_C_ASSERT(
      phasemanager_->element()->num_material() > 1, "no second material defined for element ");
  FOUR_C_ASSERT(
      phasemanager_->element()->material(1)->material_type() == Core::Materials::m_structporo,
      "invalid structure material for poroelasticity");

  // cast second material to poro material
  std::shared_ptr<Mat::StructPoro> structmat =
      std::static_pointer_cast<Mat::StructPoro>(phasemanager_->element()->material(1));

  // empty parameter list
  Teuchos::ParameterList params;

  j_ = J;

  // use structure material to evaluate porosity
  structmat->compute_porosity(params, phasemanager_->solid_pressure(), J, -1, porosity_,
      &dporosity_dp_, &dporosity_dj_, nullptr, nullptr, nullptr, false);

  // Note:
  // for phase law density dependent, incompressible:
  // porosity = 1 - sumaddvolfrac - (1 - porosity_0 - sumaddvolfrac_0)/J
  // with porosity_0 = porosity_0' - sumaddvolfrac_0   (porosity_0' = element initial porosity)
  // porosity = 1 - sumaddvolfrac - (1 - porosity_0')/J

  // subtract additional volume fractions
  // porosity = porosity' - sumaddvolfracs
  porosity_ -= phasemanager_->sum_add_vol_frac();

  // calculate the derivative of the porosity w.r.t. fluid phases
  for (int iphase = 0; iphase < phasemanager_->num_fluid_phases(); iphase++)
    (*porosityderiv_)(iphase) = dporosity_dp_ * solid_pressure_deriv(iphase);

  // additional derivatives w.r.t. volume fractions = -1.0
  for (int ivolfrac = 0; ivolfrac < phasemanager_->num_vol_frac(); ivolfrac++)
    (*porosityderiv_)(ivolfrac + phasemanager_->num_fluid_phases()) = -1.0;

  // no additional derivatives w.r.t volume fraction pressures

  // recalculate the solid pressure in case of volume fractions
  if (phasemanager_->num_vol_frac() > 0) phasemanager_->recalculate_solid_pressure(porosity_);

  return;
}

/*----------------------------------------------------------------------*
 | zero all values at GP                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::clear_gp_state()
{
  // this is just a safety call. All quantities should be recomputed
  // in the next EvaluateGPState call anyway, but you never know ...

  PhaseManagerDeriv::clear_gp_state();

  // zero everything
  porosity_ = 0.0;
  j_ = 0.0;
  dporosity_dj_ = 0.0;
  dporosity_dp_ = 0.0;
  porosityderiv_->putScalar(0.0);

  return;
}

/*----------------------------------------------------------------------*
 *  get porosity                                            vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::porosity() const
{
  check_is_evaluated();

  return porosity_;
}

/*----------------------------------------------------------------------*
 *  get Jacobian of deformation gradient               kremheller 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::jacobian_def_grad() const
{
  check_is_evaluated();

  return j_;
}

/*----------------------------------------------------------------------*
 *  get derivative of porosity wrt jacobian of defgrad kremheller 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::
    porosity_deriv_wrt_jacobian_def_grad() const
{
  check_is_evaluated();

  return dporosity_dj_;
}

/*----------------------------------------------------------------------*
 *  get derivative of saturation of phase                   vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::porosity_deriv(
    int doftoderive) const
{
  phasemanager_->check_is_evaluated();

  return (*porosityderiv_)(doftoderive);
}

/*----------------------------------------------------------------------*
 * check if porosity depends on pressure               kremheller 07/17 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::porosity_depends_on_fluid()
    const
{
  phasemanager_->check_is_evaluated();

  // this check is needed for speeding up calculations of element stiffness matrices
  return (fabs(dporosity_dp_) > 1.0e-10 ||
          phasemanager_->total_num_dof() - phasemanager_->num_fluid_phases() > 0);
}

/*----------------------------------------------------------------------*
 * check if porosity depends on JacobianDefGradient    kremheller 07/17 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::PoroFluidManager::PhaseManagerDerivAndPorosity::porosity_depends_on_struct()
    const
{
  phasemanager_->check_is_evaluated();

  // this check is needed for speeding up calculations of element stiffness matrices
  return (fabs(dporosity_dj_) > 1.0e-10);
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::PoroFluidManager::PhaseManagerReaction::PhaseManagerReaction(
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager)
    : PhaseManagerDecorator(phasemanager), numscal_(0)
{
  return;
}

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerReaction::setup(
    const Core::Elements::Element* ele, const int matnum)
{
  // setup the wrapped class
  phasemanager_->setup(ele, matnum);

  // get material
  const Core::Mat::Material& material = *(phasemanager_->element()->material(matnum));

  // get total number of dofs
  const int totalnumdof = phasemanager_->total_num_dof();

  isreactive_.resize(totalnumdof, false);

  // cast the material to multiphase material
  const Mat::FluidPoroMultiPhaseReactions& multiphasemat =
      dynamic_cast<const Mat::FluidPoroMultiPhaseReactions&>(material);

  // get number of reactions
  const int numreactions = multiphasemat.num_reac();

  // cast third material to scatra material
  // TODO: does this always work? probably yes if called with porofluidmultiphase element, but not
  // when
  //       access from outside is requested with matnum =! 0, however, in that case we will not need
  //       the PhaseManagerReaction
  std::shared_ptr<Mat::MatList> scatramat =
      std::static_pointer_cast<Mat::MatList>(phasemanager_->element()->material(2));

  // get number of scalars
  numscal_ = scatramat->num_mat();
  scalartophasemap_.resize(numscal_);

  // fill scalar to phase id vector
  if (scatramat->material_type() == Core::Materials::m_matlist or
      scatramat->material_type() == Core::Materials::m_matlist_reactions)
  {
    for (int k = 0; k < numscal_; ++k)
    {
      int matid = scatramat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = scatramat->material_by_id(matid);
      if (singlemat->material_type() == Core::Materials::m_scatra_in_fluid_porofluid_pressure_based)
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(singlemat);
        scalartophasemap_[k].phaseID = poromat->phase_id();
        scalartophasemap_[k].species_type = Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid;
      }
      else if (singlemat->material_type() ==
               Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based)
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(singlemat);
        scalartophasemap_[k].phaseID = poromat->phase_id();
        scalartophasemap_[k].species_type =
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac;
      }
      else if (singlemat->material_type() ==
               Core::Materials::m_scatra_in_solid_porofluid_pressure_based)
      {
        // dummy value because species in solid do not have a phaseID
        scalartophasemap_[k].phaseID = -1000;
        scalartophasemap_[k].species_type = Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid;
      }
      else if (singlemat->material_type() ==
               Core::Materials::m_scatra_as_temperature_porofluid_pressure_based)
      {
        // dummy value because temperature does not have a phaseID
        scalartophasemap_[k].phaseID = -1000;
        scalartophasemap_[k].species_type =
            Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature;
      }
      else
        FOUR_C_THROW("only MAT_scatra_multiporo_(fluid,volfrac,solid,temperature) valid here");
    }
  }
  else if (scatramat->material_type() ==
           Core::Materials::m_scatra_in_fluid_porofluid_pressure_based)
  {
    const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& poromat =
        std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(scatramat);
    scalartophasemap_[0].phaseID = poromat->phase_id();
    scalartophasemap_[0].species_type = Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid;
  }
  else if (scatramat->material_type() ==
           Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based)
  {
    const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
        std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(scatramat);
    scalartophasemap_[0].phaseID = poromat->phase_id();
    scalartophasemap_[0].species_type = Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac;
  }
  else if (scatramat->material_type() ==
           Core::Materials::m_scatra_in_solid_porofluid_pressure_based)
  {
    // dummy value because species in solid do not have a phaseID
    scalartophasemap_[0].phaseID = -1000;
    scalartophasemap_[0].species_type = Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid;
  }
  else if (scatramat->material_type() ==
           Core::Materials::m_scatra_as_temperature_porofluid_pressure_based)
  {
    // dummy value because temperature does not have a phaseID
    scalartophasemap_[0].phaseID = -1000;
    scalartophasemap_[0].species_type = Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature;
  }
  else
    FOUR_C_THROW(
        "only MAT_matlist_reactions, MAT_matlist or "
        "MAT_scatra_multiporo_(fluid,volfrac,solid,temperature) valid here");

  for (int ireac = 0; ireac < numreactions; ireac++)
  {
    // get the single phase material
    Mat::FluidPoroSingleReaction& singlephasemat =
        PoroPressureBased::ElementUtils::get_single_reaction_mat_from_multi_reactions_material(
            multiphasemat, ireac);

    for (int iphase = 0; iphase < totalnumdof; iphase++)
    {
      isreactive_[iphase] = isreactive_[iphase] or singlephasemat.is_reactive(iphase);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerReaction::evaluate_gp_state(
    double J, const VariableManagerMinAccess& varmanager, const int matnum)
{
  phasemanager_->evaluate_gp_state(J, varmanager, matnum);

  // get material
  FOUR_C_ASSERT(phasemanager_->element()->material(matnum) != nullptr,
      "Material of element is null pointer!");
  const Core::Mat::Material& material = *(phasemanager_->element()->material(matnum));

  if (material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW(
        "Invalid material! Only MAT_FluidPoroMultiPhaseReactions material valid for reaction "
        "evaluation!");

  // cast the material to multiphase material
  const Mat::FluidPoroMultiPhaseReactions& multiphasemat =
      dynamic_cast<const Mat::FluidPoroMultiPhaseReactions&>(material);

  // get number of reactions
  const int numreactions = multiphasemat.num_reac();
  // get number of phases
  const int numfluidphases = phasemanager_->num_fluid_phases();
  // get number of volume fractions
  const int numvolfrac = phasemanager_->num_vol_frac();
  // get total number of dofs
  const int totalnumdof = phasemanager_->total_num_dof();

  // get the volume fraction vector
  const std::vector<double> volfrac = phasemanager_->vol_frac();
  // get the volume fraction pressure vector
  const std::vector<double> volfracpressure = phasemanager_->vol_frac_pressure();

  reacterms_.clear();
  reactermsderivs_.clear();
  reactermsderivspressure_.clear();
  reactermsderivssaturation_.clear();
  reactermsderivsporosity_.clear();
  reactermsderivsscalar_.clear();
  reacterms_.resize(totalnumdof, 0.0);
  reactermsderivs_.resize(totalnumdof, std::vector<double>(totalnumdof, 0.0));
  reactermsderivspressure_.resize(totalnumdof, std::vector<double>(numfluidphases, 0.0));
  reactermsderivssaturation_.resize(totalnumdof, std::vector<double>(numfluidphases, 0.0));
  reactermsderivsporosity_.resize(totalnumdof, 0.0);
  reactermsderivsscalar_.resize(totalnumdof, std::vector<double>(numscal_, 0.0));
  reactermsderivsvolfrac_.resize(totalnumdof, std::vector<double>(numvolfrac, 0.0));
  reactermsderivsvolfracpressure_.resize(totalnumdof, std::vector<double>(numvolfrac, 0.0));

  for (int ireac = 0; ireac < numreactions; ireac++)
  {
    // get the single phase material
    Mat::FluidPoroSingleReaction& singlephasemat =
        PoroPressureBased::ElementUtils::get_single_reaction_mat_from_multi_reactions_material(
            multiphasemat, ireac);

    // evaluate the reaction
    singlephasemat.evaluate_reaction(reacterms_, reactermsderivspressure_,
        reactermsderivssaturation_, reactermsderivsporosity_, reactermsderivsvolfrac_,
        reactermsderivsvolfracpressure_, reactermsderivsscalar_, phasemanager_->pressure(),
        phasemanager_->saturation(), phasemanager_->porosity(), volfrac, volfracpressure,
        *varmanager.scalarnp());
  }

  for (int jdof = 0; jdof < totalnumdof; jdof++)
  {
    std::vector<double>& myphasederiv = reactermsderivs_[jdof];
    const std::vector<double>& myderivspressure = reactermsderivspressure_[jdof];
    const std::vector<double>& myderivssaturation = reactermsderivssaturation_[jdof];
    const std::vector<double>& myderivsvolfrac = reactermsderivsvolfrac_[jdof];
    const std::vector<double>& myderivsvolfracpress = reactermsderivsvolfracpressure_[jdof];
    const double myderivsporosity = reactermsderivsporosity_[jdof];

    // reaction derivs w.r.t. to fluid phases
    for (int doftoderive = 0; doftoderive < numfluidphases; doftoderive++)
    {
      for (int idof = 0; idof < numfluidphases; idof++)
        myphasederiv[doftoderive] +=
            myderivspressure[idof] * phasemanager_->pressure_deriv(idof, doftoderive) +
            myderivssaturation[idof] * phasemanager_->saturation_deriv(idof, doftoderive);
      if (phasemanager_->porosity_depends_on_fluid())
      {
        myphasederiv[doftoderive] += myderivsporosity * phasemanager_->porosity_deriv(doftoderive);
      }
    }
    // reaction derivs w.r.t. to volume fraction phases
    for (int doftoderive = numfluidphases; doftoderive < totalnumdof - numvolfrac; doftoderive++)
    {
      // derivatives w.r.t. volume fractions directly appearing
      //                and porosity (since it depends on volfrac)
      myphasederiv[doftoderive] += myderivsvolfrac[doftoderive - numfluidphases] +
                                   myderivsporosity * phasemanager_->porosity_deriv(doftoderive);
    }
    // reaction derivs w.r.t. to volume fraction pressures
    for (int doftoderive = numfluidphases + numvolfrac; doftoderive < totalnumdof; doftoderive++)
    {
      myphasederiv[doftoderive] += myderivsvolfracpress[doftoderive - numfluidphases - numvolfrac];
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | zero all values at GP                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Discret::Elements::PoroFluidManager::PhaseManagerReaction::clear_gp_state()
{
  // this is just a safety call. All quantities should be recomputed
  // in the next EvaluateGPState call anyway, but you never know ...
  phasemanager_->clear_gp_state();

  reacterms_.clear();
  reactermsderivs_.clear();
  reactermsderivspressure_.clear();
  reactermsderivssaturation_.clear();
  reactermsderivsporosity_.clear();
  reactermsderivsscalar_.clear();
  reactermsderivsvolfrac_.clear();
  reactermsderivsvolfracpressure_.clear();

  return;
}

/*----------------------------------------------------------------------*
 | get the reaction term                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerReaction::reac_term(int phasenum) const
{
  phasemanager_->check_is_evaluated();
  return reacterms_[phasenum];
}

/*----------------------------------------------------------------------*
 | get the derivative of the reaction term                   vuong 08/16 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerReaction::reac_deriv(
    int phasenum, int doftoderive) const
{
  phasemanager_->check_is_evaluated();
  return reactermsderivs_[phasenum][doftoderive];
}

/*----------------------------------------------------------------------*
 | get total number of scalars                         kremheller 07/17 |
 *----------------------------------------------------------------------*/
int Discret::Elements::PoroFluidManager::PhaseManagerReaction::num_scal() const
{
  phasemanager_->check_is_setup();
  return numscal_;
}

/*------------------------------------------------------------------------*
 | get the derivative of the reaction term wrt. porosity kremheller 07/17 |
 *------------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerReaction::reac_deriv_porosity(
    int phasenum) const
{
  phasemanager_->check_is_evaluated();
  return reactermsderivsporosity_[phasenum];
}

/*----------------------------------------------------------------------*
 | get the derivative of the reaction term wrt. scalar kremheller 07/17 |
 *----------------------------------------------------------------------*/
double Discret::Elements::PoroFluidManager::PhaseManagerReaction::reac_deriv_scalar(
    int phasenum, int scaltoderive) const
{
  phasemanager_->check_is_evaluated();
  return reactermsderivsscalar_[phasenum][scaltoderive];
}

/*----------------------------------------------------------------------*
 | get the scalar to phase mapping                     kremheller 08/17 |
 *----------------------------------------------------------------------*/
Mat::ScaTraMatMultiPoro::ScalarToPhaseMap
Discret::Elements::PoroFluidManager::PhaseManagerReaction::scalar_to_phase(int iscal) const
{
  phasemanager_->check_is_evaluated();
  return scalartophasemap_[iscal];
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
template <int nsd>
Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::PhaseManagerDiffusion(
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager)
    : PhaseManagerDecorator(phasemanager)
{
}

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::setup(
    const Core::Elements::Element* ele, const int matnum)
{
  // setup the wrapped class
  phasemanager_->setup(ele, matnum);

  // get number of phases
  const int numfluidphases = phasemanager_->num_fluid_phases();
  const int numvolfrac = phasemanager_->num_vol_frac();
  const int totalnumdof = phasemanager_->total_num_dof();


  // resize vectors to numfluidphases and numvolfrac resp.
  permeabilitytensors_.resize(numfluidphases);
  constrelpermeability_.resize(numfluidphases, false);
  constdynviscosity_.resize(numfluidphases, false);
  permeabilitytensorsvolfracpress_.resize(numvolfrac);
  constdynviscosityvolfracpress_.resize(numvolfrac, false);

  const Core::Mat::Material& material = *(phasemanager_->element()->material(matnum));

  // check material type
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase material valid");

  // cast to multiphase material
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  // fluid phases
  for (int iphase = 0; iphase < numfluidphases; iphase++)
  {
    // get the single phase material
    const Mat::FluidPoroSinglePhase& singlephasemat =
        PoroPressureBased::ElementUtils::get_single_phase_mat_from_material(material, iphase);
    constrelpermeability_[iphase] = singlephasemat.has_constant_rel_permeability();
    constdynviscosity_[iphase] = singlephasemat.has_constant_viscosity();

    // TODO only isotropic, constant permeability for now
    permeabilitytensors_[iphase].clear();
    const double permeability = multiphasemat.permeability();
    for (int i = 0; i < nsd; i++) (permeabilitytensors_[iphase])(i, i) = permeability;
  }

  // volfrac pressure phases

  if (totalnumdof - numfluidphases == numvolfrac * 2)
  {
    for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
    {
      // get the volfrac pressure material
      const Mat::FluidPoroVolFracPressure& volfracpressmat =
          PoroPressureBased::ElementUtils::get_vol_frac_pressure_mat_from_material(
              material, ivolfrac + numvolfrac + numfluidphases);

      constdynviscosityvolfracpress_[ivolfrac] = volfracpressmat.has_constant_viscosity();
      // clear
      permeabilitytensorsvolfracpress_[ivolfrac].clear();

      // TODO only isotropic, constant permeability for now
      const double permeability = volfracpressmat.permeability();
      for (int i = 0; i < nsd; i++)
        (permeabilitytensorsvolfracpress_[ivolfrac])(i, i) = permeability;
    }
  }
  else if (totalnumdof - numfluidphases == numvolfrac)
  {
    // only one volfrac material n the case of blood lung possible
    // get the volfrac pressure material
    const Mat::FluidPoroVolFracPressureBloodLung& volfracpressuremat_bloodlung =
        PoroPressureBased::ElementUtils::get_single_vol_frac_pressure_blood_lung_mat_from_material(
            material, numfluidphases);

    constdynviscosityvolfracpress_[0] = volfracpressuremat_bloodlung.has_constant_viscosity();
    // clear
    permeabilitytensorsvolfracpress_[0].clear();
    // TODO only isotropic, constant permeability for now
    const double permeability = volfracpressuremat_bloodlung.permeability();
    for (int i = 0; i < nsd; i++) (permeabilitytensorsvolfracpress_[0])(i, i) = permeability;
  }
  else
  {
    FOUR_C_THROW("You have an invalid volfracclosingrelation in the PhasemanagerDiffusion!");
  }
}

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::evaluate_gp_state(
    double J, const VariableManagerMinAccess& varmanager, const int matnum)
{
  // evaluate wrapped manager
  phasemanager_->evaluate_gp_state(J, varmanager, matnum);

  // get material
  FOUR_C_ASSERT(phasemanager_->element()->material(matnum) != nullptr,
      "Material of element is null pointer!");
  const Core::Mat::Material& material = *(phasemanager_->element()->material(matnum));

  // get number of phases
  const int numfluidphases = phasemanager_->num_fluid_phases();

  // resize vectors
  relpermeabilities_.resize(numfluidphases);
  derrelpermeabilities_.resize(numfluidphases);

  // check material type
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase material valid");

  // cast to multiphase material
  const Mat::FluidPoroMultiPhase& multiphasemat =
      static_cast<const Mat::FluidPoroMultiPhase&>(material);

  for (int iphase = 0; iphase < numfluidphases; iphase++)
  {
    // get the single phase material
    const Mat::FluidPoroSinglePhase& singlephasemat =
        PoroPressureBased::ElementUtils::get_single_phase_mat_from_material(multiphasemat, iphase);

    // evaluate relative permeabilities
    relpermeabilities_[iphase] = singlephasemat.rel_permeability(phasemanager_->saturation(iphase));

    // evaluate derivative of relative permeability w.r.t. saturation
    derrelpermeabilities_[iphase] =
        singlephasemat.evaluate_deriv_of_rel_permeability_wrt_saturation(
            phasemanager_->saturation(iphase));
  }

  return;
}

/*----------------------------------------------------------------------*
 | zero all values at GP                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::clear_gp_state()
{
  // this is just a safety call. All quantities should be recomputed
  // in the next EvaluateGPState call anyway, but you never know ...
  phasemanager_->clear_gp_state();

  relpermeabilities_.clear();
  derrelpermeabilities_.clear();

  return;
}

/*---------------------------------------------------------------------------*
 * get diffusion tensor                                         vuong 08/16 |
 *---------------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::permeability_tensor(
    int phasenum, Core::LinAlg::Matrix<nsd, nsd>& permeabilitytensor) const
{
  phasemanager_->check_is_evaluated();
  // make a hard copy for now
  permeabilitytensor = permeabilitytensors_[phasenum];
}

/*---------------------------------------------------------------------------*
 * check for constant rel permeability                      kremheller 02/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
bool Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::has_constant_rel_permeability(
    int phasenum) const
{
  check_is_setup();

  return constrelpermeability_[phasenum];
}
/*---------------------------------------------------------------------------*
 * get relative diffusivity of phase 'phasenum'             kremheller 02/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::rel_permeability(
    int phasenum) const
{
  phasemanager_->check_is_evaluated();

  return relpermeabilities_[phasenum];
}
/*---------------------------------------------------------------------------*
 * get relative diffusivity of phase 'phasenum'             kremheller 02/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::rel_permeability_deriv(
    int phasenum) const
{
  phasemanager_->check_is_evaluated();

  return derrelpermeabilities_[phasenum];
}
/*---------------------------------------------------------------------------*
 * check for constant dynamic viscosity                     kremheller 06/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
bool Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::has_constant_dyn_viscosity(
    int phasenum) const
{
  check_is_setup();

  return constdynviscosity_[phasenum];
}
/*---------------------------------------------------------------------------*
 * get dynamic viscosity of phase 'phasenum'                kremheller 02/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::dyn_viscosity(
    int phasenum, double abspressgrad, int matnum) const
{
  phasemanager_->check_is_evaluated();

  return dyn_viscosity(*element()->material(matnum), phasenum, abspressgrad);
}

/*----------------------------------------------------------------------*
 *   get dynamic viscosity of phase 'phasenum'         kremheller 02/17 |
 *----------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::dyn_viscosity(
    const Core::Mat::Material& material, int phasenum, double abspressgrad) const
{
  // get the single phase material
  const Mat::FluidPoroSinglePhase& singlephasemat =
      PoroPressureBased::ElementUtils::get_single_phase_mat_from_material(material, phasenum);

  return singlephasemat.viscosity(abspressgrad);
}

/*---------------------------------------------------------------------------*
 * get derivative of dynamic viscosity of phase 'phasenum'  kremheller 06/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::dyn_viscosity_deriv(
    int phasenum, double abspressgrad) const
{
  phasemanager_->check_is_evaluated();

  return dyn_viscosity_deriv(*element()->material(), phasenum, abspressgrad);
}

/*---------------------------------------------------------------------------*
 * get derivative of dynamic viscosity of phase 'phasenum'  kremheller 06/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::dyn_viscosity_deriv(
    const Core::Mat::Material& material, int phasenum, double abspressgrad) const
{
  // get the single phase material
  const Mat::FluidPoroSinglePhase& singlephasemat =
      PoroPressureBased::ElementUtils::get_single_phase_mat_from_material(material, phasenum);

  return singlephasemat.viscosity_deriv(abspressgrad);
}

/*------------------------------------------------------------------------------------*
 * check for constant dynamic viscosity of volume fraction pressures kremheller 02/18 |
 *-------------------------------------------------------------------------------------*/
template <int nsd>
bool Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::has_constant_dyn_viscosity_vol_frac_pressure(int volfracpressnum) const
{
  check_is_setup();

  return constdynviscosityvolfracpress_[volfracpressnum];
}

/*----------------------------------------------------------------------------------*
 * get dynamic viscosity of volume fraction pressure 'volfracnum'  kremheller 02/18 |
 *-----------------------------------------------------------------------------------*/
template <int nsd>
double
Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::dyn_viscosity_vol_frac_pressure(
    int volfracpressnum, double abspressgrad, int matnum) const
{
  phasemanager_->check_is_evaluated();

  return dyn_viscosity_vol_frac_pressure(
      *element()->material(matnum), volfracpressnum, abspressgrad);
}

/*-------------------------------------------------------------------------------------------*
 *   get dynamic viscosity of volume fraction pressure 'volfracnum'         kremheller 02/18 |
 *----------------------------------------------------------------------------------- --------*/
template <int nsd>
double
Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<nsd>::dyn_viscosity_vol_frac_pressure(
    const Core::Mat::Material& material, int volfracpressnum, double abspressgrad) const
{
  // get the single phase material
  const Mat::FluidPoroVolFracPressure& volfracpressmat =
      PoroPressureBased::ElementUtils::get_vol_frac_pressure_mat_from_material(material,
          volfracpressnum + phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac());

  return volfracpressmat.viscosity(abspressgrad);
}

/*----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::dyn_viscosity_vol_frac_pressure_blood_lung(int volfracpressnum, double abspressgrad,
    int matnum) const
{
  phasemanager_->check_is_evaluated();

  return dyn_viscosity_vol_frac_pressure_blood_lung(
      *element()->material(matnum), volfracpressnum, abspressgrad);
}

/*-------------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------- --------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::dyn_viscosity_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
    int volfracpressnum, double abspressgrad) const
{
  // get the single phase material
  const Mat::FluidPoroVolFracPressureBloodLung& volfracpressmat =
      PoroPressureBased::ElementUtils::get_single_vol_frac_pressure_blood_lung_mat_from_material(
          material, volfracpressnum + phasemanager_->num_fluid_phases());

  return volfracpressmat.viscosity(abspressgrad);
}

/*------------------------------------------------------------------------------------------------*
 * get derivative of dynamic viscosity of volume fraction pressure 'volfracnum'  kremheller 02/18 |
 *-------------------------------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::dyn_viscosity_deriv_vol_frac_pressure(int volfracpressnum, double abspressgrad) const
{
  phasemanager_->check_is_evaluated();

  return dyn_viscosity_deriv_vol_frac_pressure(
      *element()->material(), volfracpressnum, abspressgrad);
}

/*------------------------------------------------------------------------------------------------*
 * get derivative of dynamic viscosity of volume fraction pressure 'volfracnum'  kremheller 02/18 |
 *-------------------------------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::dyn_viscosity_deriv_vol_frac_pressure(const Core::Mat::Material& material,
    int volfracpressnum, double abspressgrad) const
{
  // get the single phase material
  const Mat::FluidPoroVolFracPressure& volfracpressmat =
      PoroPressureBased::ElementUtils::get_vol_frac_pressure_mat_from_material(material,
          volfracpressnum + phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac());

  return volfracpressmat.viscosity_deriv(abspressgrad);
}

/*------------------------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::dyn_viscosity_deriv_vol_frac_pressure_blood_lung(int volfracpressnum,
    double abspressgrad) const
{
  phasemanager_->check_is_evaluated();

  return dyn_viscosity_deriv_vol_frac_pressure_blood_lung(
      *element()->material(), volfracpressnum, abspressgrad);
}

/*------------------------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::dyn_viscosity_deriv_vol_frac_pressure_blood_lung(const Core::Mat::Material& material,
    int volfracpressnum, double abspressgrad) const
{
  // get the single phase material
  const Mat::FluidPoroVolFracPressureBloodLung& volfracpressmat =
      PoroPressureBased::ElementUtils::get_single_vol_frac_pressure_blood_lung_mat_from_material(
          material,
          volfracpressnum + phasemanager_->num_fluid_phases() + phasemanager_->num_vol_frac());

  return volfracpressmat.viscosity_deriv(abspressgrad);
}

/*---------------------------------------------------------------------------*
 * get permeability tensor for volume fraction pressures    kremheller 02/18 |
 *---------------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<
    nsd>::permeability_tensor_vol_frac_pressure(int volfracpressnum,
    Core::LinAlg::Matrix<nsd, nsd>& permeabilitytensorvolfracpressure) const
{
  phasemanager_->check_is_evaluated();
  // make a hard copy for now
  permeabilitytensorvolfracpressure = permeabilitytensorsvolfracpress_[volfracpressnum];
}

/*----------------------------------------------------------------------*
 | constructor                                         kremheller 08/17 |
 *----------------------------------------------------------------------*/
template <int nsd>
Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::PhaseManagerVolFrac(
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phasemanager)
    : PhaseManagerDecorator(phasemanager)
{
  return;
}

/*----------------------------------------------------------------------*
 | constructor                                         kremheller 08/17 |
 *----------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::setup(
    const Core::Elements::Element* ele, const int matnum)
{
  // setup the wrapped class
  phasemanager_->setup(ele, matnum);

  const int totalnumdof = phasemanager_->total_num_dof();
  const int numfluidphases = phasemanager_->num_fluid_phases();
  const int numvolfrac = phasemanager_->num_vol_frac();

  if (numfluidphases >= totalnumdof)
    FOUR_C_THROW("We should not be here, total numdof is {}, numfluidphases is {}",
        phasemanager_->total_num_dof(), phasemanager_->num_fluid_phases());

  difftensorsvolfrac_.resize(numvolfrac);
  omega_half_.resize(numvolfrac);

  hasaddscalardpendentflux_.resize(numvolfrac, false);

  // get material
  const Core::Mat::Material& material = *(ele->material(matnum));

  // check the material
  if (material.material_type() != Core::Materials::m_fluidporo_multiphase and
      material.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("only poro multiphase and poro multiphase reactions material valid");

  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
  {
    // get the single phase material
    const Mat::FluidPoroSingleVolFrac& singlevolfracmat =
        PoroPressureBased::ElementUtils::get_single_vol_frac_mat_from_material(
            material, ivolfrac + numfluidphases);

    // clear
    difftensorsvolfrac_[ivolfrac].clear();

    // TODO only isotropic, constant diffusivity for now
    const double diffusivity = singlevolfracmat.diffusivity();
    for (int i = 0; i < nsd; i++) (difftensorsvolfrac_[ivolfrac])(i, i) = diffusivity;

    // for faster check
    if (singlevolfracmat.has_add_scalar_dependent_flux())
    {
      hasaddscalardpendentflux_[ivolfrac] = true;

      // omega half
      omega_half_[ivolfrac].resize(singlevolfracmat.num_scal());
      omega_half_[ivolfrac] = singlevolfracmat.omega_half();
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | constructor                                         kremheller 08/17 |
 *----------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::evaluate_gp_state(
    double J, const VariableManagerMinAccess& varmanager, const int matnum)
{
  // evaluate wrapped manager
  phasemanager_->evaluate_gp_state(J, varmanager, matnum);

  // safety check
  if (phasemanager_->inv_bulkmodulus_solid() > 1.0e-14)
    FOUR_C_THROW("So far volume fractions are only possible for an incompressible solid");

  const int numfluidphases = phasemanager_->num_fluid_phases();
  const int numvolfrac = phasemanager_->num_vol_frac();

  scalardiffs_.resize(numvolfrac);

  // get material
  FOUR_C_ASSERT(phasemanager_->element()->material(matnum) != nullptr,
      "Material of element is null pointer!");
  const Core::Mat::Material& material = *(phasemanager_->element()->material(matnum));

  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
  {
    // get the single phase material
    const Mat::FluidPoroSingleVolFrac& singlevolfracmat =
        PoroPressureBased::ElementUtils::get_single_vol_frac_mat_from_material(
            material, ivolfrac + numfluidphases);

    if (this->has_add_scalar_dependent_flux(ivolfrac))
    {
      if (phasemanager_->num_scal() != singlevolfracmat.num_scal())
        FOUR_C_THROW("Wrong number of scalars for additional scalar dependent flux");

      // has to be inside loop, otherwise phasemanager might not have scalars
      const std::vector<double> scalars = *varmanager.scalarnp();

      // constant scalar diffs
      scalardiffs_[ivolfrac].resize(singlevolfracmat.num_scal());
      scalardiffs_[ivolfrac] = singlevolfracmat.scalar_diffs();

      // receptor-kinetic law:
      // Anderson, A. R. A. & Chaplain, M. A. J.:
      // Continuous and discrete mathematical models of tumor-induced angiogenesis
      for (int iscal = 0; iscal < phasemanager_->num_scal(); iscal++)
        (scalardiffs_[ivolfrac])[iscal] *=
            ((omega_half_[ivolfrac])[iscal]) / ((omega_half_[ivolfrac])[iscal] + scalars[iscal]);
    }
  }

  return;
}

/*---------------------------------------------------------------------------*
 * get diffusion tensor for volume fractions                kremheller 08/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::diff_tensor_vol_frac(
    int volfracnum, Core::LinAlg::Matrix<nsd, nsd>& difftensorvolfrac) const
{
  phasemanager_->check_is_evaluated();
  // make a hard copy for now
  difftensorvolfrac = difftensorsvolfrac_[volfracnum];
}

/*---------------------------------------------------------------------------*
 * get densities for volume fractions                       kremheller 08/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
bool Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::has_add_scalar_dependent_flux(
    int volfracnum) const
{
  phasemanager_->check_is_evaluated();

  return hasaddscalardpendentflux_[volfracnum];
}

/*---------------------------------------------------------------------------*
 * check if additional scalar flux dependency active        kremheller 08/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
bool Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::has_add_scalar_dependent_flux(
    int volfracnum, int iscal) const
{
  phasemanager_->check_is_evaluated();

  return fabs((scalardiffs_[volfracnum])[iscal]) > 1.0e-12;
}

/*---------------------------------------------------------------------------*
 * check if receptor-kinetic law active                     kremheller 01/18 |
 *---------------------------------------------------------------------------*/
template <int nsd>
bool Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::has_receptor_kinetic_law(
    int volfracnum, int iscal) const
{
  phasemanager_->check_is_evaluated();

  return ((omega_half_[volfracnum])[iscal] < 1.0e12);
}

/*---------------------------------------------------------------------------*
 * get scalar diffs for volume fractions                    kremheller 08/17 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::scalar_diff(
    int volfracnum, int iscal) const
{
  phasemanager_->check_is_evaluated();

  return (scalardiffs_[volfracnum])[iscal];
}

/*---------------------------------------------------------------------------*
 * get scalar omega half for volume fractions               kremheller 01/18 |
 *---------------------------------------------------------------------------*/
template <int nsd>
double Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::omega_half(
    int volfracnum, int iscal) const
{
  phasemanager_->check_is_evaluated();

  return (omega_half_[volfracnum])[iscal];
}

/*----------------------------------------------------------------------*
 | zero all values at GP                               kremheller 08/17 |
 *----------------------------------------------------------------------*/
template <int nsd>
void Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<nsd>::clear_gp_state()
{
  // this is just a safety call. All quantities should be recomputed
  // in the next EvaluateGPState call anyway, but you never know ...
  phasemanager_->clear_gp_state();

  scalardiffs_.clear();

  return;
}

///*----------------------------------------------------------------------*
// *----------------------------------------------------------------------*/
//// template classes

template class Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<1>;
template class Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<2>;
template class Discret::Elements::PoroFluidManager::PhaseManagerDiffusion<3>;
template class Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<1>;
template class Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<2>;
template class Discret::Elements::PoroFluidManager::PhaseManagerVolFrac<3>;

FOUR_C_NAMESPACE_CLOSE
