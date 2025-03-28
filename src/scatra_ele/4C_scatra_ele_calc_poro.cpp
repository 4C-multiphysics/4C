// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_poro.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcPoro<distype>*
Discret::Elements::ScaTraEleCalcPoro<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcPoro<distype>>(
            new ScaTraEleCalcPoro<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcPoro<distype>::ScaTraEleCalcPoro(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      xyze0_(Core::LinAlg::Initialization::zero),
      eporosity_(Core::LinAlg::Initialization::zero),
      isnodalporosity_(false)
{
  // initialization of diffusion manager (override initialization in base class)
  my::diffmanager_ = std::make_shared<ScaTraEleDiffManagerPoro>(my::numscal_);

  return;
}

/*----------------------------------------------------------------------*
 | evaluate action                                          vuong 07/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ScaTraEleCalcPoro<distype>::evaluate_action(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const ScaTra::Action& action, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::calc_total_and_mean_scalars:
    {
      // get flag for inverting
      bool inverting = params.get<bool>("inverting");

      // need current scalar vector
      // -> extract local values from the global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, la[0].lm_);

      extract_element_and_node_values_poro(ele, params, discretization, la);

      // calculate scalars and domain integral
      calculate_scalars(ele, elevec1_epetra, inverting, false);

      break;
    }
    default:
      return my::evaluate_action(ele, params, discretization, action, la, elemat1_epetra,
          elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | read element coordinates                                 vuong 10/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::read_element_coordinates(
    const Core::Elements::Element* ele)
{
  // call base class
  my::read_element_coordinates(ele);

  // copy initial node position
  xyze0_ = my::xyze_;

  return;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     ehrl 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::extract_element_and_node_values(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  extract_element_and_node_values_poro(ele, params, discretization, la);

  my::extract_element_and_node_values(ele, params, discretization, la);

  return;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     ehrl 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::extract_element_and_node_values_poro(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // get number of dofset associated with velocity related dofs
  const int ndsvel = my::scatrapara_->nds_vel();

  // get velocity values at nodes
  const std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discretization.get_state(ndsvel, "convective velocity field");

  // safety check
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // extract pressure if applicable
  if (static_cast<unsigned>(numveldofpernode) > nsd_)
  {
    // construct location vector for pressure dofs
    std::vector<int> lmpre(nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
      lmpre[inode] = la[ndsvel].lm_[inode * numveldofpernode + nsd_];

    // extract local values of pressure field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*convel, my::eprenp_, lmpre);
  }

  // this is a hack. Check if the structure (assumed to be the dofset 1) has more DOFs than
  // dimension. If so, we assume that this is the porosity
  if (discretization.num_dof(1, ele->nodes()[0]) == nsd_ + 1)
  {
    isnodalporosity_ = true;

    // get number of dof-set associated with velocity related dofs
    const int ndsdisp = my::scatrapara_->nds_disp();

    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        discretization.get_state(ndsdisp, "dispnp");

    if (disp != nullptr)
    {
      std::vector<double> mydisp = Core::FE::extract_values(*disp, la[ndsdisp].lm_);

      for (unsigned inode = 0; inode < nen_; ++inode)  // number of nodes
        eporosity_(inode, 0) = mydisp[nsd_ + (inode * (nsd_ + 1))];
    }
    else
      FOUR_C_THROW("Cannot get state vector displacement");
  }
  else
    isnodalporosity_ = false;

  return;
}

/*----------------------------------------------------------------------*
 |  get the material constants  (protected)                  vuong 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    std::vector<double>& densn,          //!< density at t_(n)
    std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,         //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    const int iquad                      //!< id of current gauss point
)
{
  // calculate gauss point porosity from fluid and solid and (potentially) scatra solution
  compute_porosity(ele);

  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  // get diffusivity / diffusivities
  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<const Mat::MatList>& actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() < my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      my::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else
    my::materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);

  return;
}  // ScaTraEleCalcPoro::get_material_params

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::mat_scatra(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  if (iquad == -1)
    FOUR_C_THROW("no gauss point given for evaluation of scatra material. Check your input file.");

  // read the porosity from the diffusion manager
  const double porosity = diff_manager()->get_porosity(k);

  const std::shared_ptr<const Mat::ScatraMat>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMat>(material);

  //  if(k<NO_CONVECTION_NR)
  {
    // set diffusivity (scaled with porosity)
    set_diffusivity(actmat, k, porosity);

    // set densities (scaled with porosity)
    set_densities(porosity, densn, densnp, densam);
  }
  //  else
  //  {
  //    // set diffusivity (scaled with porosity)
  //    set_diffusivity(actmat,k,1.0);
  //
  //    // set densities (scaled with porosity)
  //    set_densities(1.0,densn,densnp,densam);
  //  }

  return;
}  // ScaTraEleCalcPoro<distype>::MatScaTra

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
inline void Discret::Elements::ScaTraEleCalcPoro<distype>::set_diffusivity(
    const std::shared_ptr<const Mat::ScatraMat>& material, const int k, const double scale)
{
  my::diffmanager_->set_isotropic_diff(material->diffusivity() * scale, k);

  return;
}

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
inline void Discret::Elements::ScaTraEleCalcPoro<distype>::set_densities(
    double porosity, double& densn, double& densnp, double& densam)
{
  // all densities are set to the porosity
  densn = porosity;
  densnp = porosity;
  densam = porosity;

  return;
}

/*----------------------------------------------------------------------*
 |  get the material constants  (protected)                  vuong 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::compute_porosity(
    const Core::Elements::Element* ele  //!< the element we are dealing with
)
{
  double porosity = 0.0;

  if (isnodalporosity_)
  {
    porosity = eporosity_.dot(my::funct_);
  }
  else
  {
    // gauss point displacements
    Core::LinAlg::Matrix<nsd_, 1> dispint(Core::LinAlg::Initialization::uninitialized);
    dispint.multiply(my::edispnp_, my::funct_);

    //------------------------get determinant of Jacobian dX / ds
    // transposed jacobian "dX/ds"
    Core::LinAlg::Matrix<nsd_, nsd_> xjm0;
    xjm0.multiply_nt(my::deriv_, xyze0_);

    // inverse of transposed jacobian "ds/dX"
    const double det0 = xjm0.determinant();

    my::xjm_.multiply_nt(my::deriv_, my::xyze_);
    const double det = my::xjm_.determinant();

    // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds)
    // )^-1
    const double J = det / det0;

    // fluid pressure at gauss point
    const double pres = compute_pore_pressure();

    // empty parameter list
    Teuchos::ParameterList params;

    if (ele->num_material() < 2) FOUR_C_THROW("no secondary material available");

    // here we rely that the structure material has been added as second material
    std::shared_ptr<Mat::StructPoro> structmat =
        std::dynamic_pointer_cast<Mat::StructPoro>(ele->material(1));
    if (structmat == nullptr) FOUR_C_THROW("cast to Mat::StructPoro failed!");

    // just evaluate the first scalar (used only in case of reactive porosity)
    std::shared_ptr<std::vector<double>> scalars = std::make_shared<std::vector<double>>(0);
    for (int k = 0; k < my::numscal_; ++k)
    {
      const double phinp = my::ephinp_[k].dot(my::funct_);
      scalars->push_back(phinp);
    }
    params.set<std::shared_ptr<std::vector<double>>>("scalar", scalars);

    params.set<double>("delta time", my::scatraparatimint_->dt());

    // use structure material to evaluate porosity
    structmat->compute_porosity(
        params, pres, J, -1, porosity, nullptr, nullptr, nullptr, nullptr, nullptr, false);
  }

  // save porosity in diffusion manager for later access
  diff_manager()->set_porosity(porosity);

  return;
}

/*----------------------------------------------------------------------*
 |  get the material constants  (protected)                  vuong 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Discret::Elements::ScaTraEleCalcPoro<distype>::compute_pore_pressure()
{
  return my::eprenp_.dot(my::funct_);
}

/*----------------------------------------------------------------------*
|  calculate scalar(s) and domain integral                  vuong 07/15|
| (overwrites method in ScaTraEleCalc)                                 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoro<distype>::calculate_scalars(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseVector& scalars, bool inverting,
    bool calc_grad_phi)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // calculate gauss point porosity from fluid and solid and (potentially) scatra solution
    compute_porosity(ele);

    // calculate integrals of (inverted) scalar(s) and domain
    if (inverting)
    {
      for (unsigned i = 0; i < nen_; i++)
      {
        const double fac_funct_i = fac * my::funct_(i);
        for (int k = 0; k < my::numscal_; k++)
        {
          const double porosity = diff_manager()->get_porosity(k);
          if (std::abs(my::ephinp_[k](i, 0)) > 1e-14)
            scalars[k] += fac_funct_i / (my::ephinp_[k](i, 0) * porosity);
          else
            FOUR_C_THROW("Division by zero");
        }
        // for domain volume
        scalars[my::numscal_] += fac_funct_i;
      }
    }
    else
    {
      for (unsigned i = 0; i < nen_; i++)
      {
        const double fac_funct_i = fac * my::funct_(i);
        for (int k = 0; k < my::numscal_; k++)
        {
          const double porosity = diff_manager()->get_porosity(k);
          scalars[k] += fac_funct_i * my::ephinp_[k](i, 0) * porosity;
        }
        // for domain volume
        scalars[my::numscal_] += fac_funct_i;
      }
    }
  }  // loop over integration points

  return;
}  // ScaTraEleCalc::calculate_scalars

// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::nurbs9>;
// template class Discret::Elements::ScaTraEleCalcPoro<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
