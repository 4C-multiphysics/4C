/*----------------------------------------------------------------------*/
/*! \file

\brief routines to initialize homogeneous isotropic turbulence simulations


\level 2

*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_FLUID_TURBULENCE_HIT_INITIAL_FIELD_HPP
#define FOUR_C_FLUID_TURBULENCE_HIT_INITIAL_FIELD_HPP

#include "4C_config.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace FLD
{
  // forward declarations
  class FluidImplicitTimeInt;

  // inital condition for homogeneous isotropic turbulence
  // based on the Comte-Bellot - Corrsion experiment
  class HomIsoTurbInitialField
  {
   public:
    //! constructor
    HomIsoTurbInitialField(
        FluidImplicitTimeInt& timeint, const Inpar::FLUID::InitialField initfield);

    //! destructor
    virtual ~HomIsoTurbInitialField() = default;

    //! calculate initial field
    virtual void calculate_initial_field();

   protected:
    //! sort criterium for double values up to a tolerance of 10-9
    class LineSortCriterion
    {
     public:
      bool operator()(const double& p1, const double& p2) const { return (p1 < p2 - 1E-9); }

     protected:
     private:
    };

    //! non-dimensionalize and store experimental data
    void prepare_exparimental_data();

    //! estimate energy form given energy spectrum (experimental data)
    double interpolate_energy_from_spectrum(double k);

    //! estimate energy form given energy spectrum (function for E)
    double calculate_energy_from_spectrum(double k);

    //! fluid discretization
    Teuchos::RCP<Core::FE::Discretization> discret_;

    //! state vectors to be initialized
    Teuchos::RCP<Core::LinAlg::Vector> velnp_;
    Teuchos::RCP<Core::LinAlg::Vector> veln_;
    Teuchos::RCP<Core::LinAlg::Vector> velnm_;
    //! type of energy spectrum for initialization
    Inpar::FLUID::InitialField type_;

    //! number of resolved mode
    int nummodes_;

    //! vector of coordinates in one spatial direction (same for the other two directions)
    Teuchos::RCP<std::vector<double>> coordinates_;

    //! vector containing wave numbers of experiment
    std::vector<double> k_exp_;
    //! vector containing corresponding energy
    std::vector<double> E_exp_;
  };

  // inital condition for homogeneous isotropic turbulence
  // based on the Comte-Bellot - Corrsion experiment
  class HomIsoTurbInitialFieldHDG : public HomIsoTurbInitialField
  {
   public:
    //! constructor
    HomIsoTurbInitialFieldHDG(
        FluidImplicitTimeInt& timeint, const Inpar::FLUID::InitialField initfield);


    //! calculate initial field
    void calculate_initial_field() override;

   protected:
    Teuchos::RCP<Core::LinAlg::Vector> intveln_;
    Teuchos::RCP<Core::LinAlg::Vector> intvelnm_;
    Teuchos::RCP<Core::LinAlg::Vector> intvelnp_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
