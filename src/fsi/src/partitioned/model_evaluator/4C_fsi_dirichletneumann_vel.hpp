// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_DIRICHLETNEUMANN_VEL_HPP
#define FOUR_C_FSI_DIRICHLETNEUMANN_VEL_HPP

#include "4C_config.hpp"

#include "4C_fsi_dirichletneumann.hpp"

FOUR_C_NAMESPACE_OPEN

// Forward declarations

namespace Adapter
{
  class FBIConstraintenforcer;
}

namespace BeamInteraction
{
  class BeamToFluidMeshtyingVtkOutputWriter;
}

namespace Core::Binstrategy
{
  class BinningStrategy;
}

namespace FSI
{
  /**
   * \brief Dirichlet-Neumann interface velocity based algorithm
   *
   */
  class DirichletNeumannVel : public DirichletNeumann
  {
    friend class DirichletNeumannFactory;

   protected:
    /**
     *  \brief constructor
     *
     * You will have to use the FSI::DirichletNeumannFactory to create an instance of this class
     */
    explicit DirichletNeumannVel(MPI_Comm comm);

   public:
    /*! \brief Outer level FSI time loop
     *
     * We overload this interface here in order to carry out operations that only have to be done
     * once at the start of the simulation, but need information which are not available during
     * setup in the case of a restart.
     *
     *  \param[in] interface Our interface to NOX
     */
    void timeloop(const std::shared_ptr<NOX::Nln::Interface::RequiredBase> interface) override;

    /** \brief Here we decide which type of coupling we are going to use
     *
     * Here we check the input for the coupling variable
     *
     */
    void setup() override;

    /** \brief Here the base class writes output for each field and in addition we write coupling
     * related output
     *
     */
    void output() override;

    /// Set the binning object for the presort strategy in the FBI constraint enforcer
    void set_binning(std::shared_ptr<Core::Binstrategy::BinningStrategy> binning);

   protected:
    /** \brief interface fluid operator
     *
     * In here, the nonlinear solve for the fluid field is prepared and called and the resulting
     * interface force is returned
     *
     * \param[in] ivel The interface velocity
     * \param[in] fillFlag Type of evaluation in computeF() (cf. NOX documentation for details)
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> fluid_op(
        std::shared_ptr<Core::LinAlg::Vector<double>> ivel, const FillType fillFlag) override;

    /** \brief interface structural operator
     *
     * In here, the nonlinear solve for the structure field is prepared and called and the resulting
     * interface velocity is returned
     *
     * \param[in] iforce The interface force
     * \param[in] fillFlag Type of evaluation in computeF() (cf. NOX documentation for details)
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> struct_op(
        std::shared_ptr<Core::LinAlg::Vector<double>> iforce, const FillType fillFlag) override;

    /// Computes initial guess for the next iteration
    std::shared_ptr<Core::LinAlg::Vector<double>> initial_guess() override;

    /**
     * \brief In here all coupling related quantities are given to the fluid solver
     *
     * \param[in] iv In our case, the input parameter is not used!
     * \returns ivel The fluid velocity on the (whole) fluid domain
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> struct_to_fluid(
        std::shared_ptr<Core::LinAlg::Vector<double>> iv) override;

    /**
     * \brief In here all coupling related quantities are assembled for the structure solver
     *
     * \param[in] iv In our case, the input parameter is not used!
     * \returns iforce The fsi force acting on the structure
     */

    std::shared_ptr<Core::LinAlg::Vector<double>> fluid_to_struct(
        std::shared_ptr<Core::LinAlg::Vector<double>> iv) override;


   private:
    /**
     * \brief Object that allows to capsule the different constraint enforcement strategies and
     * effectively separating it from the actual algorithm
     */
    std::shared_ptr<Adapter::FBIConstraintenforcer> constraint_manager_;

    std::shared_ptr<BeamInteraction::BeamToFluidMeshtyingVtkOutputWriter>
        visualization_output_writer_;
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
