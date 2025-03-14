// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_RESULTTEST_HPP
#define FOUR_C_SSI_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace SSI
{
  // forward declarations
  class SSIBase;
  class SsiMono;

  /*!
    \brief result testing functionality for scalar-structure interaction problems

    This class provides result testing functionality for quantities associated with
    scalar-structure interaction as an overall problem type. Quantities associated
    with either the scalar or the structural field are not tested by this class, but
    by field-specific result testing classes. Feel free to extend this class if necessary.

    \sa ResultTest

  */
  class SSIResultTest : public Core::Utils::ResultTest
  {
   public:
    /*!
     * @brief constructor
     *
     * @param[in] ssi_base  time integrator for scalar-structure interaction
     */
    explicit SSIResultTest(const std::shared_ptr<const SSI::SSIBase> ssi_base);

    void test_special(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   private:
    /*!
     * @brief get special result to be tested
     *
     * @param[in] quantity  name of quantity to be tested
     * @return special result
     */
    double result_special(const std::string& quantity) const;

    //! return time integrator for monolithic scalar-structure interaction
    const SSI::SsiMono& ssi_mono() const;

    //! time integrator for scalar-structure interaction
    const std::shared_ptr<const SSI::SSIBase> ssi_base_;
  };
}  // namespace SSI
FOUR_C_NAMESPACE_CLOSE

#endif
