// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TSI_PARTITIONED_HPP
#define FOUR_C_TSI_PARTITIONED_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 12/09 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_tsi_algorithm.hpp"
#include "4C_tsi_input.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                           dano 12/09 |
 *----------------------------------------------------------------------*/
namespace TSI
{
  //! TSI algorithm base
  //!
  //!
  //!  Base class of TSI algorithms. Derives from structure_base_algorithm and
  //!  Thermo::BaseAlgorithm with temperature field.
  //!  There can (and will) be different subclasses that implement different
  //!  coupling schemes.
  //!
  //!  \warning The order of calling the two BaseAlgorithm-constructors (that
  //!  is the order in which we list the base classes) is important here! In the
  //!  constructors control file entries are written. And these entries define
  //!  the order in which the filters handle the Discretizations, which in turn
  //!  defines the dof number ordering of the Discretizations... Don't get
  //!  confused. Just always list structure, thermo. In that order.
  class Partitioned final : public Algorithm
  {
   public:
    explicit Partitioned(MPI_Comm comm);

    //! non-linear solve, i.e. (multiple) corrector
    [[nodiscard]] TSI::StepStatus solve() override;

    //! initialise internal variables needed as guess for the partitioned TSI algorithm
    void setup_system() override;

    //! time loop for TSI algorithm with one-way coupling
    void time_loop_sequ_stagg();

    //! time loop for TSI algorithm with one-way coupling
    void time_loop_one_way();

    //! time loop for TSI algorithm with iteration between fields (full coupling)
    void time_loop_full();

    //! read restart data
    void read_restart(int step  //!< step number where the calculation is continued
        ) override;


   protected:
    //! @name Time loop building blocks

    //! start a new time step
    void prepare_time_step() override;

    //! calculate stresses, strains, energies
    void prepare_output(bool force_prepare) override;

    //! take current results for converged and save for next time step
    void update() override;

    //! solve temperature equations for current time step
    void do_thermo_step();

    //! solve displacement equations for current time step
    void do_structure_step();

    // two-way coupling (iterative staggered)

    //! outer iteration loop
    void outer_iteration_loop();

    //@}

    //! convergence check for iterative staggered TSI solver
    bool convergence_check(int itnum,  //!< index of current iteration
        int itmax,                     //!< maximal number of iterations
        double ittol                   //!< iteration tolerance
    );

    ConvNorm normtypeinc_;  //!< convergence check for residual temperatures

    //! maximum iteration steps
    int itmax_;
    //! convergence tolerance
    double ittol_;

    //! temperature increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> tempincnp_;
    //! displacement increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> dispincnp_;

   private:
    //! displacements at time step (t_n) or (t_n+1)
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp_;
    //! velocities at time step (t_n) or (t_n+1)
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel_;

    //! temperature at time step (t_n) or (t_n+1)
    std::shared_ptr<Core::LinAlg::Vector<double>> temp_;

    //! @name Aitken relaxation

    //! difference of last two solutions
    // del = r^{i+1}_{n+1} = d^{i+1}_{n+1} - d^i_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> del_;
    //! difference of difference of last two pair of solutions
    // delhist = ( r^{i+1}_{n+1} - r^i_{n+1} )
    std::shared_ptr<Core::LinAlg::Vector<double>> delhist_;
    //! Aitken factor
    double mu_;

    //@}

    //! coupling algorithm
    SolutionSchemeOverFields coupling_;
    //! we couple based on displacements
    bool displacementcoupling_;
    //! quasi-static solution of the mechanical equation
    bool quasistatic_;

  };  // Partitioned
}  // namespace TSI


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
