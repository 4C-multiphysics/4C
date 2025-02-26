// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_RESULTTEST_HPP
#define FOUR_C_STRUCTURE_NEW_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_result_test.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class Solver;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}  // namespace Core::IO

namespace Solid
{
  namespace TimeInt
  {
    class BaseDataGlobalState;
  }  // namespace TimeInt
  namespace ModelEvaluator
  {
    class Data;
  }  // namespace ModelEvaluator

  /*! \brief Structure specific result test class */
  class ResultTest : public Core::Utils::ResultTest
  {
    /// possible status flag for the result test
    enum class Status : char
    {
      evaluated,
      unevaluated
    };


   public:
    /// possible value for test operation on geometry
    enum class TestOp : int
    {
      sum,
      max,
      min,
      unknown
    };

    //! Constructor for time integrators of general kind
    //! \author bborn \date 06/08 (originally)
    ResultTest();

    //! initialization of class variables
    virtual void init(
        const Solid::TimeInt::BaseDataGlobalState& gstate, const Solid::ModelEvaluator::Data& data);

    //! setup of class variables
    virtual void setup();

    //! \brief structure version of nodal value tests
    //!
    //! Possible position flags are "dispx", "dispy", "dispz",
    //!                             "velx", "vely", "velz",
    //!                             "accx", "accy", "accz"
    //!                             "stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_xz",
    //!                             "stress_yz"
    //!
    //! \note The type of stress that is used for testing has to be specified in IO->STRUCT_STRESS
    void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

    //! \brief structure version of nodal value tests on geometry
    //!
    //! Possible position flags are "dispx", "dispy", "dispz",
    //!                             "velx", "vely", "velz",
    //!                             "accx", "accy", "accz"
    //!                             "stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_xz",
    //!                             "stress_yz"
    //!
    //! \note The type of stress that is used for testing has to be specified in IO->STRUCT_STRESS
    void test_node_on_geometry(const Core::IO::InputParameterContainer& container, int& nerr,
        int& test_count, const std::vector<std::vector<std::vector<int>>>& nodeset) override;

    /*! \brief test special quantity not associated with a particular element or node
     *
     *  \param[in] res          input file line containing result test specification
     *  \param[out] nerr        updated number of failed result tests
     *  \param[out] test_count  updated number of result tests
     *  \param[out] uneval_test_count  updated number of unevaluated tests
     *
     */
    void test_special(const Core::IO::InputParameterContainer& container, int& nerr,
        int& test_count, int& uneval_test_count) override;

   protected:
    /// get the indicator state
    inline const bool& is_init() const { return isinit_; };

    /// get the indicator state
    inline const bool& is_setup() const { return issetup_; };

    /// Check if init() and setup() have been called
    inline void check_init_setup() const
    {
      FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
    }

    /// Check if init() has been called
    inline void check_init() const { FOUR_C_ASSERT(is_init(), "Call init() first!"); }

   private:
    /** \brief Get the result of the special structural quantity
     *
     *  The %special_status flag is used to identify circumstances where an
     *  evaluation of the status test is not possible, because the quantity
     *  is not accessible. One example is the number of nonlinear iterations
     *  for a step which is not a part of the actual simulation. Think of a
     *  restart scenario.
     *
     *  \param[in]  quantity        name of the special quantity
     *  \param[out] special_status  status of the specual result test
     *  \return  The value for the subsequent comparison.
     *
     *  \author hiermeier \date 11/17 */
    std::optional<double> get_special_result(
        const std::string& quantity, Status& special_status) const;

    /** \brief Get the last number of linear iterations
     *
     * If the number of iterations for the desired step is accessible, it will
     *  be returned and the special_status flag is set to evaluated. Note that
     *  the step number is part of the quantity name and will be automatically
     *  extracted. The used format is
     *
     *                       lin_iter_step_<INT>
     *
     *  The integer <INT> must be at the very last position separated by an
     *  underscore.
     *
     *  \param[in]  quantity        name of the special quantity
     *  \param[out] special_status  status of the special result test
     *  \return  The number of linear iterations, if possible. Otherwise -1.
     *
     */
    std::optional<int> get_last_lin_iteration_number(
        const std::string& quantity, Status& special_status) const;

    /** \brief Get the number of nonlinear iterations
     *
     *  If the number of iterations for the desired step is accessible, it will
     *  be returned and the special_status flag is set to evaluated. Note that
     *  the step number is part of the quantity name and will be automatically
     *  extracted. The used format is
     *
     *                       num_iter_step_<INT>
     *
     *  The integer <INT> must be at the very last position separated by an
     *  underscore.
     *
     *  \param[in]  quantity        name of the special quantity
     *  \param[out] special_status  status of the special result test
     *  \return  The number of nonlinear iterations, if possible. Otherwise -1.
     *
     *  \author hiermeier \date 11/17 */
    std::optional<int> get_nln_iteration_number(
        const std::string& quantity, Status& special_status) const;

    std::optional<int> get_nodes_per_proc_number(
        const std::string& quantity, Status& special_status) const;


    /** \brief Get the value for a specific energy (internal, kinetic, total, etc.)
     *
     *  If the energy is accessible, it will be returned and special_status flag is set to
     *  evaluated. If not, error is thrown in Solid::ModelEvaluator::Data
     *
     *  \param[in]  quantity        name of the energy
     *  \param[out] special_status  status of the special result test
     *  \return     The requested energy
     *
     *  \author kremheller \date 11/19 */
    std::optional<double> get_energy(const std::string& quantity, Status& special_status) const;

    /**! \brief extract nodal value on specific position
     *  \param[in]  node        node index
     *  \param[in]  position  the quantity to extract
     *  \param[out] result  the nodal value
     *  \return     the flag indicates whether the node is in the current proc or not */
    int get_nodal_result(double& result, const int node, const std::string& position) const;

   protected:
    //! flag which indicates if the init() routine has already been called
    bool isinit_;

    //! flag which indicates if the setup() routine has already been called
    bool issetup_;

   private:
    //! our discretisation
    std::shared_ptr<const Core::FE::Discretization> strudisc_;
    // our solution
    //! global displacement DOFs
    std::shared_ptr<const Core::LinAlg::Vector<double>> disn_;
    //! global velocity DOFs
    std::shared_ptr<const Core::LinAlg::Vector<double>> veln_;
    //! global acceleration DOFs
    std::shared_ptr<const Core::LinAlg::Vector<double>> accn_;
    //! global reaction DOFs
    std::shared_ptr<const Core::LinAlg::Vector<double>> reactn_;
    /* NOTE: these have to be present explicitly
     * as they are not part of the problem instance like in fluid3
     */

    //! pointer to the global state object of the structural time integration
    std::shared_ptr<const Solid::TimeInt::BaseDataGlobalState> gstate_;
    //! pointer to the data container of the structural time integration
    std::shared_ptr<const Solid::ModelEvaluator::Data> data_;
  };  // class ResultTest

  /*----------------------------------------------------------------------------*/
  /** \brief Get the integer at the very last position of a name string
   *
   *  \pre The integer must be separated by an underscore from the prefix, e.g.
   *              any-name-even_with_own_underscores_3
   *       The method will return 3 in this case.
   *
   *  \param[in] name  string name to extract from
   *  \return Extracted integer at the very last position of the name.
   *
   *  \author hiermeier \date 11/17 */
  int get_integer_number_at_last_position_of_name(const std::string& quantity);

}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
