/*----------------------------------------------------------------------*/
/*! \file

\brief Monolithic coupling of 3D structural dynamics and 0D cardiovascular flow models

\level 2

*----------------------------------------------------------------------*/

#ifndef FOUR_C_CARDIOVASCULAR0D_RESULTTEST_HPP
#define FOUR_C_CARDIOVASCULAR0D_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace UTILS
{
  class Cardiovascular0D;
  class Cardiovascular0DManager;
}  // namespace UTILS

namespace Core::IO
{
  class DiscretizationWriter;
}

/*!
  \brief Structure specific result test class
*/
class Cardiovascular0DResultTest : public Core::UTILS::ResultTest
{
 public:
  Cardiovascular0DResultTest(
      UTILS::Cardiovascular0DManager& cardvasc0dman, Teuchos::RCP<Core::FE::Discretization> discr);

  void test_special(
      const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;



 private:
  Teuchos::RCP<Core::FE::Discretization> actdisc_;  ///< standard discretization

  const Teuchos::RCP<Core::LinAlg::Vector<double>> cardvasc0d_dof_;

  const bool havecardio_4elementwindkessel_;
  const bool havecardio_arterialproxdist_;
  const bool havecardio_syspulcirculation_;
  const bool havecardiorespir_syspulperiphcirculation_;
};

FOUR_C_NAMESPACE_CLOSE

#endif
