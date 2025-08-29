// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_SPRINGDASHPOT_HPP
#define FOUR_C_CONSTRAINT_SPRINGDASHPOT_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_pairedvector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>


FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Adapter
{
  class CouplingNonLinMortar;
}

namespace Constraints
{
  class SpringDashpot
  {
   public:
    //! Type of spring
    enum class RobinSpringDashpotType
    {
      xyz,            ///<
      refsurfnormal,  ///<
      cursurfnormal   ///<
    };

    /*!
    \brief constructor
     */
    SpringDashpot(
        std::shared_ptr<Core::FE::Discretization> dis, const Core::Conditions::Condition& cond);

    //! add contribution of spring dashpot BC to residual vector
    // old version, NOT consistently integrated over element surface!!
    void evaluate_force(Core::LinAlg::Vector<double>& fint,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        const Core::LinAlg::Vector<double>& vel, const Teuchos::ParameterList& p);

    //! add contribution of spring dashpot BC to stiffness matrix
    // old version, NOT consistently integrated over element surface!!
    // ToDo: remove redundant code in evaluate_force and evaluate_force_stiff
    // -> however should migrate to new EvaluateRobin... mhv 08/2016
    void evaluate_force_stiff(Core::LinAlg::SparseMatrix& stiff, Core::LinAlg::Vector<double>& fint,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        const Core::LinAlg::Vector<double>& vel, Teuchos::ParameterList p);

    // NEW version, consistently integrated over element surface!!
    void evaluate_robin(std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        const Core::LinAlg::Vector<double>& velo, Teuchos::ParameterList p);

    //! reset after Newton step
    void reset_newton();

    //! reset after prestressing with MULF
    void reset_prestress(const Core::LinAlg::Vector<double>& dis);

    //! set reset after prestressing with MULF
    void set_restart(Core::LinAlg::Vector<double>& vec);

    //! set reset after prestressing with MULF
    void set_restart_old(Core::LinAlg::MultiVector<double>& vec);

    //! output of gap, normal, and nodal stiffness
    void output_gap_normal(Core::LinAlg::Vector<double>& gap,
        Core::LinAlg::MultiVector<double>& normals,
        Core::LinAlg::MultiVector<double>& stress) const;

    //! select spring stiffness for tensile or compressive spring
    double select_stiffness(double gap)
    {
      if (gap > 0)
        return stiff_tens_;  // gap positive: tensile spring
      else
        return stiff_comp_;  // gap negative: compressive spring
    }

    //! output of spring offset
    void output_prestr_offset(Core::LinAlg::Vector<double>& springprestroffset) const;

    //! output of spring offset
    void output_prestr_offset_old(Core::LinAlg::MultiVector<double>& springprestroffset) const;

    //! return type of spring
    RobinSpringDashpotType get_spring_type() { return springtype_; }

    //! update condition for new time step
    void update();

    /*!
     * \brief Reset the current state variables to the ones of the previous timestep
     *
     * This method is used in conjunction with
     * Solid::MODELEVALUATOR::ModelEvaluatorManager::reset_step_state() and is used to prepare the
     * output of the previous timestep after calling update(). This is used for example to output
     * the last successful timestep.
     */
    void reset_step_state();

   private:
    //! set type of spring during initialization
    void set_spring_type();

    //! set up MORTAR interface for direction cursurfnormal
    void initialize_cur_surf_normal();

    //! calculate nodal area - old!
    void get_area(const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom);

    //! get current normal
    void get_cur_normals(
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& disp, Teuchos::ParameterList p);

    //! initialize prestr offset
    void initialize_prestr_offset();

    std::shared_ptr<Core::FE::Discretization> actdisc_;  ///< standard discretization
    const Core::Conditions::Condition* spring_;          ///< spring dashpot condition

    /// Mortar interface in case of curnormal springs
    std::shared_ptr<Adapter::CouplingNonLinMortar> mortar_;

    //! @name Spring properties
    //@{

    //! Spring stiffness when spring is in tension
    const double stiff_tens_;

    //! Spring stiffness when spring is in compression
    const double stiff_comp_;

    //! Spring offset
    const double offset_;

    //! Dashpot viscosity
    const double viscosity_;

    //! Coupling id of reference DSURFACE
    const int coupling_;

    //@}

    //! @name Condition properties
    //@{

    //! Condition nodes
    const std::vector<int>* nodes_;

    //! Condition real area
    std::map<int, double> area_;

    //@}

    //! @name Spring dashpot evaluation
    //@{

    //! Nodal gap in reference configuration
    std::map<int, double> gap0_;

    //! Nodal gap in current configuration
    std::map<int, double> gap_;

    //! Nodal gap velocity in current configuration
    std::map<int, double> gapdt_;

    //! Nodal gap in current configuration (last time step)
    std::map<int, double> gapn_;

    //! Linearization of nodal gap
    std::map<int, std::map<int, double>> dgap_;

    //! Nodal normal
    std::map<int, std::vector<double>> normals_;

    //! Linearization of nodal normal
    std::map<int, std::vector<Core::Gen::Pairedvector<int, double>>> dnormals_;

    //! Nodal force applied by spring dashpot BC for output
    std::map<int, std::vector<double>> springstress_;

    //! Prestressing offset
    std::map<int, std::vector<double>> offset_prestr_;

    //@}

    /*! \brief New prestressing offset
     *
     *  This is a pointer to the accumulated whole displacement vector of all last load steps
     *  has dimension of full problem
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> offset_prestr_new_;

   private:
    //! Type of spring
    RobinSpringDashpotType springtype_;

  };  // class
}  // namespace Constraints

FOUR_C_NAMESPACE_CLOSE

#endif
