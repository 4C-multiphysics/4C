// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_chemo.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_chemotaxis.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

//! note for chemotaxis in 4C:
//! assume the following situation: scalar A does follow the gradient of scalar B (i.e. B is the
//! attractant and scalar A the chemotractant) with chemotactic coefficient 3.0
//!
//! the corresponding equations are: \partial_t A = -(3.0*A \nabla B)  (negative since scalar is
//! chemotracted)
//!                              \partial_t B = 0
//!
//! this equation is in 4C achieved by the MAT_scatra_reaction material:
//! ----------------------------------------------------------MATERIALS
//! MAT 1 MAT_matlist_chemotaxis LOCAL No NUMMAT 2 MATIDS 101 102 NUMPAIR 1 PAIRIDS 111 END
//! MAT 101 MAT_scatra DIFFUSIVITY 0.0
//! MAT 102 MAT_scatra DIFFUSIVITY 0.0
//! MAT 111 MAT_scatra_chemotaxis NUMSCAL 2 PAIR -1 1 CHEMOCOEFF 3.0

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcChemo<distype, probdim>*
Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcChemo<distype, probdim>>(
            new ScaTraEleCalcChemo<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 *  constructor---------------------------                              |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::ScaTraEleCalcChemo(
    const int numdofpernode, const int numscal, const std::string& disname)
    : my::ScaTraEleCalc(numdofpernode, numscal, disname),
      numcondchemo_(-1),
      pair_(0),
      chemocoeff_(0)
{
}

/*--------------------------------------------------------------------------- *
 |  calculation of chemotactic element matrix                     thon 06/ 15 |
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::calc_mat_chemo(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac,
    const double timetaufac, const double densnp, const double scatrares,
    const Core::LinAlg::Matrix<nen_, 1>& sgconv, const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  std::shared_ptr<varmanager> varmanager = my::scatravarmanager_;

  for (int condnum = 0; condnum < numcondchemo_; condnum++)
  {
    const std::vector<int>& pair = pair_[condnum];    // get stoichometrie
    const double& chemocoeff = chemocoeff_[condnum];  // get chemotaxis coefficient

    if (pair[k] == -1)  // if chemotractant
    {
      // Standard Galerkin terms

      Core::LinAlg::Matrix<nen_, nen_> gradgradmatrix(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nen_, 1> bigterm(Core::LinAlg::Initialization::zero);

      const double chemofac = timefacfac * densnp;
      const int partner = get_partner(pair);  // Get attracting partner ID

      const Core::LinAlg::Matrix<nsd_, 1> gradattractant =
          varmanager->grad_phi(partner);  // Gradient of attracting partner

      bigterm.multiply_tn(my::derxy_, gradattractant);
      gradgradmatrix.multiply_tn(
          my::derxy_, my::derxy_);  // N1,x*N1,x+N1,y*N1,y+... ; N1,x*N2,x+N1,y*N2,y+...

      for (unsigned vi = 0; vi < nen_; vi++)
      {
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ui++)
        {
          const int fui = ui * my::numdofpernode_ + k;

          emat(fvi, fui) -= chemofac * chemocoeff * bigterm(vi) * my::funct_(ui);
        }
      }

      for (unsigned vi = 0; vi < nen_; vi++)
      {
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ui++)
        {
          const int fui = ui * my::numdofpernode_ + partner;
          emat(fvi, fui) -=
              chemofac * chemocoeff * my::scatravarmanager_->phinp(k) * gradgradmatrix(vi, ui);
        }
      }

      if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
      {
        FOUR_C_THROW("stabilization for chemotactic problems is not jet implemented!");
      }  // end stabilization

    }  // pair[i] == -1
  }

}  // end calc_mat_chemo


/*--------------------------------------------------------------------------- *
 |  calculation of chemotactic RHS                                thon 06/ 15 |
 *----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::calc_rhs_chemo(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac, const double rhstaufac,
    const double scatrares, const double densnp)
{
  std::shared_ptr<varmanager> varmanager = my::scatravarmanager_;

  for (int condnum = 0; condnum < numcondchemo_; condnum++)
  {
    const std::vector<int>& pair = pair_[condnum];    // get stoichometrie
    const double& chemocoeff = chemocoeff_[condnum];  // get reaction coefficient

    if (pair[k] == -1)  // if chemotractant
    {
      // Standard Galerkin terms

      const int partner = get_partner(pair);

      Core::LinAlg::Matrix<nen_, 1> gradfunctattr(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> attractant = varmanager->grad_phi(partner);

      gradfunctattr.multiply_tn(my::derxy_, attractant);

      const double decoyed = varmanager->phinp(k);

      for (unsigned vi = 0; vi < nen_; vi++)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        erhs[fvi] += rhsfac * chemocoeff * decoyed * gradfunctattr(vi);
      }


      if (my::scatrapara_->stab_type() != Inpar::ScaTra::stabtype_no_stabilization)
      {
        FOUR_C_THROW("stabilization for chemotactic problems is not jet implemented!");
      }  // end stabilization

    }  // pair[i] == -1
  }

}  // calc_rhs_chemo


/*----------------------------------------------------------------------*
 |  get the attractant id                                   thon 06/ 15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::get_partner(
    const std::vector<int> pair)
{
  int partner = 0;

  for (int numscalar = 0; numscalar < my::numscal_; numscalar++)
  {
    if (pair[numscalar] == 1)
    {
      partner = numscalar;
      break;
    }
  }

  return partner;
}


/*----------------------------------------------------------------------*
 |  get the material constants  (private)                   thon 06/ 15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    std::vector<double>& densn,          //!< density at t_(n)
    std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,         //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    const int iquad                      //!< id of current gauss point
)
{
  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  // We may have some chemotactic and some non-chemotactic discretisation.
  // But since the calculation classes are singleton, we have to reset all chemotaxis stuff each
  // time
  clear_chemotaxis_terms();

  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<const Mat::MatList>& actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      my::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else if (material->material_type() == Core::Materials::m_matlist_chemotaxis)
  {
    const std::shared_ptr<const Mat::MatListChemotaxis>& actmat =
        std::dynamic_pointer_cast<const Mat::MatListChemotaxis>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    get_chemotaxis_coefficients(
        actmat);  // read all chemotaxis input from material and copy it into local variables

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      my::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else
  {
    my::materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);
  }
  return;
}  // ScaTraEleCalc::get_material_params

/*-----------------------------------------------------------------------------------*
 |  Clear all chemotaxtis related class variable (i.e. set them to zero)  thon 06/15 |
 *-----------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::clear_chemotaxis_terms()
{
  numcondchemo_ = 0;

  // We always have to reinitialize these vectors since our elements are singleton
  pair_.resize(numcondchemo_);
  chemocoeff_.resize(numcondchemo_);
}

/*-----------------------------------------------------------------------------------------*
 |  get numcond, pairing list, chemotaxis coefficient from material            thon 06/ 15 |
 *-----------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::get_chemotaxis_coefficients(
    const std::shared_ptr<const Core::Mat::Material> material  //!< pointer to current material
)
{
  const std::shared_ptr<const Mat::MatListChemotaxis>& actmat =
      std::dynamic_pointer_cast<const Mat::MatListChemotaxis>(material);

  if (actmat == nullptr) FOUR_C_THROW("cast to MatListChemotaxis failed");

  // We always have to reinitialize these vectors since our elements are singleton
  numcondchemo_ = actmat->num_pair();
  pair_.resize(numcondchemo_);
  chemocoeff_.resize(numcondchemo_);

  for (int i = 0; i < numcondchemo_; i++)
  {
    const int pairid = actmat->pair_id(i);
    const std::shared_ptr<const Mat::ScatraChemotaxisMat>& chemomat =
        std::dynamic_pointer_cast<const Mat::ScatraChemotaxisMat>(actmat->material_by_id(pairid));

    pair_[i] = *(chemomat->pair());            // get pairing
    chemocoeff_[i] = chemomat->chemo_coeff();  // get chemotaxis coefficient
  }
}

/*-------------------------------------------------------------------------------*
 |  calculation of strong residual for stabilization                             |
 | (depending on respective stationary or time-integration scheme)   thon 06/ 15 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::calc_strong_residual(
    const int k,           //!< index of current scalar
    double& scatrares,     //!< residual of convection-diffusion-reaction eq
    const double densam,   //!< density at t_(n+am)
    const double densnp,   //!< density at t_(n+1)
    const double rea_phi,  //!< reactive contribution
    const double rhsint,   //!< rhs at integration point
    const double tau       //!< the stabilisation parameter
)
{
  // Note: order is important here
  // First the scatrares without chemotaxis..
  my::calc_strong_residual(k, scatrares, densam, densnp, rea_phi, rhsint, tau);

  // Second chemotaxis to strong residual
  std::shared_ptr<varmanager> varmanager = my::scatravarmanager_;

  double chemo_phi = 0;

  for (int condnum = 0; condnum < numcondchemo_; condnum++)
  {
    const std::vector<int>& pair = pair_[condnum];    // get stoichometrie
    const double& chemocoeff = chemocoeff_[condnum];  // get reaction coefficient

    if (pair[k] == -1)
    {
      const int partner = get_partner(pair);
      double laplattractant = 0;

      if (my::use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        Core::LinAlg::Matrix<nen_, 1> laplace(Core::LinAlg::Initialization::zero);
        my::get_laplacian_strong_form(laplace);
        laplattractant = laplace.dot(my::ephinp_[partner]);
      }

      Core::LinAlg::Matrix<1, 1> chemoderivattr(Core::LinAlg::Initialization::zero);
      chemoderivattr.multiply_tn(varmanager->grad_phi(partner), varmanager->grad_phi(k));

      chemo_phi += chemocoeff * (chemoderivattr(0, 0) + laplattractant * varmanager->phinp(k));
    }
  }

  chemo_phi *= my::scatraparatimint_->time_fac() / my::scatraparatimint_->dt();

  // Add to residual
  scatrares += chemo_phi;

  return;
}

// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcChemo<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
