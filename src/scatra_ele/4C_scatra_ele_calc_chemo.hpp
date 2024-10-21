#ifndef FOUR_C_SCATRA_ELE_CALC_CHEMO_HPP
#define FOUR_C_SCATRA_ELE_CALC_CHEMO_HPP

#include "4C_config.hpp"

#include "4C_mat_scatra_chemotaxis.hpp"
#include "4C_scatra_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace ELEMENTS
  {
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcChemo : public virtual ScaTraEleCalc<distype, probdim>
    {
     protected:
      //! (private) protected constructor, since we are a Singleton.
      ScaTraEleCalcChemo(const int numdofpernode, const int numscal, const std::string& disname);

     private:
      typedef ScaTraEleCalc<distype, probdim> my;
      using my::nen_;
      using my::nsd_;
      using varmanager = ScaTraEleInternalVariableManager<nsd_, nen_>;

     public:
      //! Singleton access method
      static ScaTraEleCalcChemo<distype, probdim>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

     protected:
      //! calculation of chemotactic element matrix
      void calc_mat_chemo(Core::LinAlg::SerialDenseMatrix& emat, const int k,
          const double timefacfac, const double timetaufac, const double densnp,
          const double scatrares, const Core::LinAlg::Matrix<nen_, 1>& sgconv,
          const Core::LinAlg::Matrix<nen_, 1>& diff) override;

      //! calculation of chemotactic RHS
      void calc_rhs_chemo(Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac,
          const double rhstaufac, const double scatrares, const double densnp) override;

      //! get the material parameters
      void get_material_params(
          const Core::Elements::Element* ele,  //!< the element we are dealing with
          std::vector<double>& densn,          //!< density at t_(n)
          std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
          std::vector<double>& densam,         //!< density at t_(n+alpha_M)
          double& visc,                        //!< fluid viscosity
          const int iquad                      //!< id of current gauss point
          ) override;

      //! Clear all chemotaxtis related class variable (i.e. set them to zero)
      void clear_chemotaxis_terms();

      //! Get and save all chemotaxtis related class variable
      virtual void get_chemotaxis_coefficients(
          const Teuchos::RCP<const Core::Mat::Material> material  //!< pointer to current material
      );

      //! Get ID of attractant (i.e. scalar which gradient the current scalar shall follow)
      virtual int get_partner(const std::vector<int> pair);

      //! calculation of strong residual for stabilization
      void calc_strong_residual(const int k,  //!< index of current scalar
          double& scatrares,                  //!< residual of convection-diffusion-reaction eq
          const double densam,                //!< density at t_(n+am)
          const double densnp,                //!< density at t_(n+1)
          const double rea_phi,               //!< reactive contribution
          const double rhsint,                //!< rhs at integration point
          const double tau                    //!< the stabilisation parameter
          ) override;


      int numcondchemo_;                    //!< number of chemotactic conditions
      std::vector<std::vector<int>> pair_;  //!< vector containing the pairings
      std::vector<double> chemocoeff_;  //!< constants by which the chemotactic terms are multiplied

    };  // end Chemotaxis

  }  // namespace ELEMENTS

}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
