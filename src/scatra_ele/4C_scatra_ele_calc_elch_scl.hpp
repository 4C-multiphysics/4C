// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ELCH_SCL_HPP
#define FOUR_C_SCATRA_ELE_CALC_ELCH_SCL_HPP
#include "4C_config.hpp"

#include "4C_scatra_ele_calc_elch_diffcond.hpp"
#include "4C_scatra_ele_parameter_elch_diffcond.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declarations
    class ScaTraEleDiffManagerElchScl;
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchScl;
    template <Core::FE::CellType distype>
    class ScaTraEleUtilsElchScl;

    // class implementation
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcElchScl : public ScaTraEleCalcElchDiffCond<distype, probdim>
    {
     protected:
      /// protected constructor, since we are a Singleton, but a derived class exists
      ScaTraEleCalcElchScl(const int numdofpernode, const int numscal, const std::string& disname);

      using my = ScaTraEleCalc<distype, probdim>;
      using myelch = ScaTraEleCalcElch<distype, probdim>;
      using mydiffcond = ScaTraEleCalcElchDiffCond<distype, probdim>;

     public:
      /// Singleton access method
      static ScaTraEleCalcElchScl<distype, probdim>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

     protected:
      /*========================================================================*/
      //! @name general framework
      /*========================================================================*/

      void calc_mat_and_rhs(Core::LinAlg::SerialDenseMatrix& emat,
          Core::LinAlg::SerialDenseVector& erhs, const int k, const double fac,
          const double timefacfac, const double rhsfac, const double taufac,
          const double timetaufac, const double rhstaufac,
          Core::LinAlg::Matrix<my::nen_, 1>& tauderpot, double& rhsint) override;

      //! CalcMat: Coulomb's equation (no source)
      //!
      //! \param emat  element matrix to be filled
      //! \param timefacfac  domain-integration factor times time integration factor
      //! \param invf  1/Faraday
      //! \param scalefac  scaling factor for consistent micro-macro-coupling
      //! \param gradpot  spatial gradient of electric potential
      //! \param epsilon  dielectric permittivity of bulk electrolyte

      void calc_mat_pot_coulomb(Core::LinAlg::SerialDenseMatrix& emat, double timefacfac,
          double invf, double scalefac, const Core::LinAlg::Matrix<my::nsd_, 1>& gradpot,
          double epsilon);

      //! CalcRhs: Coulomb's equation (no source)
      //!
      //! \param erhs  element rhs to be filled
      //! \param fac  domain-integration factor
      //! \param invf  1/Faraday
      //! \param cond_invperm  conductivity / permitivity to scale pot. equations for consistent
      //! micro-macro-coupling
      //! \param gradpot  gradient of electric potential
      //! \param epsilon dielectric permittitivity of bulk electrolyte
      void calc_rhs_pot_coulomb(Core::LinAlg::SerialDenseVector& erhs, double fac, double invf,
          double cond_invperm, const Core::LinAlg::Matrix<my::nsd_, 1>& gradpot, double epsilon);

      //! CalcRhs: Source contribution to electric potential (free charge)
      //!
      //! \param emat  element matrix to calculate
      //! \param k  index of current scalar
      //! \param fac  domain-integration factor
      //! \param invf  1/Faraday
      //! \param cond_invperm  conductivity / permitivity to scale pot. equations for consistent
      //! micro-macro-coupling
      //! \param z_k_F  transference number times Faraday constant
      void calc_mat_pot_src(Core::LinAlg::SerialDenseMatrix& emat, int k, double fac, double invf,
          double cond_invperm, double z_k_F);

      //! CalcRhs: Source contribution to electric potential (free charge)
      //!
      //! \param erhs  element rhs to calculate
      //! \param k  index of current scalar
      //! \param fac  domain-integration factor
      //! \param invf  1/Faraday
      //! \param cond_invperm  conductivity / permitivity to scale pot. equations for consistent
      //! micro-macro-coupling
      //! \param q_f  free charge density
      void calc_rhs_pot_src(Core::LinAlg::SerialDenseVector& erhs, int k, double fac, double invf,
          double cond_invperm, double q_f);

      //! CalcRhs: Calculate diffusion contribution to current
      //!
      //! \param erhs  element rhs to calculate
      //! \param rhsfac  time-integration factor for rhs times domain-integration factor
      //! \param invfval  transference number time Faraday const. inverted
      //! \param gradphi  ector of scalar gradients at t_(n+1)
      void calc_rhs_diff_cur(Core::LinAlg::SerialDenseVector& erhs, double rhsfac,
          const std::vector<double>& invfval,
          const std::vector<Core::LinAlg::Matrix<my::nsd_, 1>>& gradphi);

      //! CalcMat: Calculate diffusion contribution to current
      //!
      //! \param emat   element matrix to calculate
      //! \param timefacfac  domain-integration factor times time-integration factor
      //! \param invfval  transference number time Faraday const. inverted
      //! \param gradphi  vector of scalar gradients at t_(n+1)
      void calc_mat_diff_cur(Core::LinAlg::SerialDenseMatrix& emat, double timefacfac,
          const std::vector<double>& invfval,
          const std::vector<Core::LinAlg::Matrix<my::nsd_, 1>>& gradphi);

      void calc_mat_and_rhs_outside_scalar_loop(Core::LinAlg::SerialDenseMatrix& emat,
          Core::LinAlg::SerialDenseVector& erhs, const double fac, const double timefacfac,
          const double rhsfac) override;

      /*========================================================================*/
      //! @name material and related and related functions
      /*========================================================================*/

      void get_material_params(const Core::Elements::Element* ele, std::vector<double>& densn,
          std::vector<double>& densnp, std::vector<double>& densam, double& visc,
          const int iquad = -1) override;

      /*========================================================================*/
      //! @name methods for evaluation of individual terms
      /*========================================================================*/

      //! Calculate free charge based on concentration
      double calc_free_charge(const double concentration);

      //! Calculate derivative of free charge w.r.t. concentration
      double calc_free_charge_der_conc();

      /*========================================================================*/
      //! @name additional service routines
      /*========================================================================*/

      //! get diffusion manager for diffusion-conduction formulation
      std::shared_ptr<ScaTraEleDiffManagerElchScl> diff_manager()
      {
        return std::static_pointer_cast<ScaTraEleDiffManagerElchScl>(my::diffmanager_);
      }

      //! get internal variable manager for diffusion-conduction formulation
      std::shared_ptr<ScaTraEleInternalVariableManagerElchScl<my::nsd_, my::nen_>> var_manager()
      {
        return std::static_pointer_cast<
            ScaTraEleInternalVariableManagerElchScl<my::nsd_, my::nen_>>(my::scatravarmanager_);
      }

      //! get utility class supporting element evaluation for diffusion-conduction formulation
      ScaTraEleUtilsElchScl<distype>* utils()
      {
        return static_cast<ScaTraEleUtilsElchScl<distype>*>(myelch::utils_);
      }

      //! flag for used element formulation (material based)
      Inpar::ElCh::DiffCondMat diffcondmat_;

      //! parameter class for diffusion-conduction formulation
      const ScaTraEleParameterElchDiffCond* diffcondparams_;
    };

    /// ScaTraEleDiffManagerElchSclimplementation
    /*!
      This class keeps all Diffusion-Conduction-specific transport parameter needed for the
      evaluation of an element.
    */
    class ScaTraEleDiffManagerElchScl : public ScaTraEleDiffManagerElchDiffCond
    {
     public:
      using dmdiffcond = ScaTraEleDiffManagerElchDiffCond;

      ScaTraEleDiffManagerElchScl(int numscal)
          : ScaTraEleDiffManagerElchDiffCond(numscal),
            c_max_(-1.0),
            c_bulk_(-1.0),
            c_lim_(-1.0),
            chi_(-1.0),
            epsilon_(-1.0)
      {
      }

      /*========================================================================*/
      //! @name access methods for transport parameter
      /*========================================================================*/

      //! Set dieelectric susceptibility of medium
      void set_susceptibility(const double chi) { chi_ = chi; }

      //! Set dieelectric permittivity of the medium
      void set_permittivity(const double epsilon) { epsilon_ = epsilon; }

      //! Get dieelectric permittivity of medium
      double get_permittivity() { return epsilon_; }

      //! Set Bulk concentration of cations
      void set_bulk_conc(const double cbulk) { c_bulk_ = cbulk; }

      //! Get Bulk concentration of cations
      double get_bulk_conc() { return c_bulk_; }

      void set_trans_num(const double transnum, const int k) override
      {
        if (transnum != 1.0)
          FOUR_C_THROW("Only transference number of 1.0 is allowed in the SCL-model.");
        dmdiffcond::set_trans_num(transnum, k);
      }

      void set_deriv_trans_num(const double derivtransnum, const int k, const int iscal) override
      {
        if (derivtransnum != 0.0)
        {
          FOUR_C_THROW(
              "Only constant transference number (1.0) is allowed. Derivative w.r.t. concentration "
              "has to be zero.");
        }
        dmdiffcond::set_deriv_trans_num(derivtransnum, k, iscal);
      }

      void set_phase_poro(const double eps, const int phase) override
      {
        // Only porosity of 1.0 can be assigned to the SCL-material
        if (eps != 1.0) FOUR_C_THROW("Only constant porosity (1.0) is allowed for SCL-material.");
        dmdiffcond::set_phase_poro(eps, phase);
      }

      void set_phase_tort(const double tort, const int phase) override
      {
        // only tortuosity of 1.0 can be assigned to the SCL-material
        if (tort != 1.0)
          FOUR_C_THROW("Only constant tortuosity (1.0) is allowed for SCL-material.");
        dmdiffcond::set_phase_tort(tort, phase);
      }

      /*========================================================================*/
      //! @name output
      /*========================================================================*/

      void output_transport_params(const int numscal) override
      {
        // call base class routine
        dmdiffcond::output_transport_params(numscal);

        // additional outputs
        std::cout << "maximum concentration:    " << c_max_ << std::endl;

        for (int k = 0; k < numscal; ++k)
          std::cout << "limiting concentration:    " << c_lim_ << std::endl;

        std::cout << "dieelectric susceptibility   " << chi_ << std::endl;
        std::cout << "dieelectric permittivity   " << epsilon_ << std::endl;
      }

     private:
      /*========================================================================*/
      //! @name transport parameter
      /*========================================================================*/

      //! maximum concentration of species
      double c_max_;

      //! bulk concentration (= anion concentration)
      double c_bulk_;

      //! limit concentration for extrapolation strategy of diffusion coefficient
      double c_lim_;

      //! dieelectric susceptibility of electrolyte material
      double chi_;

      //! dieelectric permittivity of electrolyte material
      double epsilon_;

      /*========================================================================*/
      //! @name geometrical parameters of the porous medium
      /*========================================================================*/
    };

    /// ScaTraEleInternalVariableManagerElchDiffCond implementation
    /*!
      This class keeps all internal variables needed for the diffusion-conduction formulation.
    */
    template <int nsd, int nen>
    class ScaTraEleInternalVariableManagerElchScl
        : public ScaTraEleInternalVariableManagerElchDiffCond<nsd, nen>
    {
     public:
      using vm = ScaTraEleInternalVariableManager<nsd, nen>;

      ScaTraEleInternalVariableManagerElchScl(int numscal,
          const Discret::Elements::ScaTraEleParameterElch* elchparams,
          const Discret::Elements::ScaTraEleParameterElchDiffCond* diffcondparams)
          : ScaTraEleInternalVariableManagerElchDiffCond<nsd, nen>(
                numscal, elchparams, diffcondparams),
            curint_(Core::LinAlg::Initialization::zero)
      {
      }

     private:
      /*========================================================================*/
      //! @name internal variables evaluated at the Gauss point
      /*========================================================================*/

      //! current density at Gauss point
      Core::LinAlg::Matrix<nsd, 1> curint_;
    };
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
