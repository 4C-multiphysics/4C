#ifndef FOUR_C_FLUID_TIMINT_RED_OST_HPP
#define FOUR_C_FLUID_TIMINT_RED_OST_HPP


#include "4C_config.hpp"

#include "4C_fluid_timint_ost.hpp"
#include "4C_fluid_timint_red.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  class TimIntRedModelsOst : public TimIntOneStepTheta, public TimIntRedModels
  {
   public:
    /// Standard Constructor
    TimIntRedModelsOst(const Teuchos::RCP<Core::FE::Discretization>& actdis,
        const Teuchos::RCP<Core::LinAlg::Solver>& solver,
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<Core::IO::DiscretizationWriter>& output, bool alefluid = false);


    /*!
    \brief initialization

    */
    void init() override;

    /*!
    \brief read restart data

    */
    void read_restart(int step) override;


   protected:
   private:
  };  // class TimIntRedModelsOst

}  // namespace FLD


FOUR_C_NAMESPACE_CLOSE

#endif
