// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_DIRICHLETNEUMANN_HPP
#define FOUR_C_FSI_DIRICHLETNEUMANN_HPP

#include "4C_config.hpp"

#include "4C_fsi_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FSI
{
  /**
   *  \brief Dirichlet-Neumann coupled interface system
   *
   * This is an abstract interface for the algorithm class of Dirichlet-Neumann
   * partitioned FSI problems. The fields themselves are solved using appropriate
   * field algorithms (that are used for standalone solvers as well.)
   * The FSI interface problem is solved using NOX.
   *
   * The FSIOp() method does one FSI cycle, that is one
   * solve of all participating fields. But for sake of clarity
   * this cycle is expressed via the two operator methods FluidOp() and
   * StructOp().
   *
   * FluidOp() takes a kinematic interface variable, applies it in some
   * shape or form to the fluid field, solves the fluid field on the
   * newly deformed fluid mesh and returns the interface forces. This needs
   * to be implemented in a derived class.
   *
   * StructOp() takes interface forces, applies them to the structural
   * field, solves the field and returns the kinematic interface variable.
   * This needs to be implemented in a derived class
   *
   * This coupling process builds on the available field solvers. However,
   * the independent parallel distribution of the fields complicates the
   * exchange of coupling information. Therefore three instances of the
   * Coupling class are used that couple those fields. On top of these
   * there are helper methods StructToAle(), struct_to_fluid(),
   * fluid_to_struct() and AleToFluid() to easily exchange distributed
   * interface vectors between fields.
   */
  class DirichletNeumann : public Partitioned
  {
   public:
    /** \brief Constructor
     *
     * @param[in] comm Communicator
     */
    explicit DirichletNeumann(MPI_Comm comm);

    /// Creates the appropriate DirichletNeumann algortihm
    //    std::shared_ptr<DirichletNeumann> FSI::DirichletNeumann::Factory(_PROBLEM_TYPE type);
    //    \FIXME

    /// setup this object
    void setup() override;

    /// Access function for the #kinematiccoupling_ flag
    bool get_kinematic_coupling() const { return kinematiccoupling_; }

    /** \brief Set function for the kinematiccoupling variable
     *  \param[in] variable value to write into #kinematiccoupling_
     */
    void set_kinematic_coupling(const bool variable) { kinematiccoupling_ = variable; }

   protected:
    /** \brief composed FSI operator
     *
     * The FSI operator performs the coupling iteration between the solid and the fluid field.
     *
     * \param[in] x Interface coupling variable (kinematic quantity or force), depending on the
     * value of #kinematiccoupling_.
     *
     * \param[in, out] F residual vector
     * \param[in] fillFlag Type of evaluation in computeF() (cf. NOX documentation for details)
     */
    void fsi_op(const Core::LinAlg::Vector<double> &x, Core::LinAlg::Vector<double> &F,
        const FillType fillFlag) override;

    /** \brief interface fluid operator
     * \param[in] icoup kinematic interface variable
     * \param[in] fillFlag Type of evaluation in computeF() (cf. NOX documentation for details)
     *
     * \returns interface force
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> fluid_op(
        std::shared_ptr<Core::LinAlg::Vector<double>> icoup, const FillType fillFlag) override = 0;

    /** \brief interface structural operator
     * \param[in] iforce interface force
     * \param[in] fillFlag Type of evaluation in computeF() (cf. NOX documentation for details)
     *
     * \returns kinematic interface variable
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> struct_op(
        std::shared_ptr<Core::LinAlg::Vector<double>> iforce, const FillType fillFlag) override = 0;

    std::shared_ptr<Core::LinAlg::Vector<double>> initial_guess() override = 0;

   private:
    /**
     * \brief  Flag to switch between kinematic and force coupling.
     *
     * - When performing kinematic coupling, we perform a fix-point scheme for the kinematic
     * variable obtained from the structure. Hence, we solve the fluid problem first.
     * - When performing force coupling, we perform a fix-point scheme for the interface force
     * obtained from the fluid field. Hence, we solve the structure problem first.
     */
    bool kinematiccoupling_;
  };

}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
