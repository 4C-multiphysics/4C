#ifndef FOUR_C_SCATRA_RESULTTEST_HDG_HPP
#define FOUR_C_SCATRA_RESULTTEST_HDG_HPP

#include "4C_config.hpp"

#include "4C_linalg_serialdensevector.hpp"
#include "4C_scatra_resulttest.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  // forward declaration
  class TimIntHDG;

  // class implementation
  class HDGResultTest : public ScaTraResultTest
  {
   public:
    //! constructor
    HDGResultTest(Teuchos::RCP<ScaTraTimIntImpl> timint);

   private:
    //! get nodal result to be tested
    double result_node(const std::string quantity,  //! name of quantity to be tested
        Core::Nodes::Node* node                     //! node carrying the result to be tested
    ) const override;

    //! time integrator
    Teuchos::RCP<const TimIntHDG> scatratiminthdg_;

    Teuchos::RCP<Core::LinAlg::SerialDenseVector> errors_;

  };  // class HDGResultTest : public ScaTraResultTest
}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
