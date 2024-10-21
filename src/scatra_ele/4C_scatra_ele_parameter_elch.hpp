#ifndef FOUR_C_SCATRA_ELE_PARAMETER_ELCH_HPP
#define FOUR_C_SCATRA_ELE_PARAMETER_ELCH_HPP

#include "4C_config.hpp"

#include "4C_inpar_elch.hpp"
#include "4C_scatra_ele_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace ELEMENTS
  {
    // class implementation
    class ScaTraEleParameterElch : public ScaTraEleParameterBase
    {
     public:
      //! singleton access method
      static ScaTraEleParameterElch* instance(
          const std::string& disname  //!< name of discretization
      );

      //! return flag for coupling of lithium-ion flux density and electric current density at
      //! Dirichlet and Neumann boundaries
      bool boundary_flux_coupling() const { return boundaryfluxcoupling_; };

      //! set parameters
      void set_parameters(Teuchos::ParameterList& parameters  //!< parameter list
          ) override;

      //! return type of closing equation for electric potential
      Inpar::ElCh::EquPot equ_pot() const { return equpot_; };

      //! return Faraday constant
      double faraday() const { return faraday_; };

      //! return the (universal) gas constant
      double gas_constant() const { return gas_constant_; };

      //! return dielectric constant
      double epsilon() const { return epsilon_; };

      //! return constant F/RT
      double frt() const { return frt_; };

      //! return the homogeneous temperature in the scatra field (can be time dependent)
      double temperature() const { return temperature_; }

     private:
      //! private constructor for singletons
      ScaTraEleParameterElch(const std::string& disname  //!< name of discretization
      );

      //! flag for coupling of lithium-ion flux density and electric current density at Dirichlet
      //! and Neumann boundaries
      bool boundaryfluxcoupling_;

      //! equation used for closing of the elch-system
      enum Inpar::ElCh::EquPot equpot_;

      //! Faraday constant
      double faraday_;

      //! (universal) gas constant
      double gas_constant_;

      //! dielectric constant
      const double epsilon_;

      //! pre-calculation of regularly used constant F/RT
      //! (a division is much more expensive than a multiplication)
      double frt_;

      //! homogeneous temperature within the scalar transport field (can be time dependent)
      double temperature_;
    };
  }  // namespace ELEMENTS
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
