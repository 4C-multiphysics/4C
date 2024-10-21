#ifndef FOUR_C_POROELAST_SCATRA_PART_2WC_HPP
#define FOUR_C_POROELAST_SCATRA_PART_2WC_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_poroelast_scatra_part.hpp"

FOUR_C_NAMESPACE_OPEN

namespace PoroElastScaTra
{
  class PoroScatraPart2WC : public PoroScatraPart
  {
   public:
    //! explicit constructor
    explicit PoroScatraPart2WC(const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams);

    //! full time loop
    void timeloop() override;

    //! read and set fields needed for restart
    void read_restart(int restart) override;

    //! prepare time step for single fields
    void prepare_time_step(bool printheader = true) override;

    //! perform iteration loop between fields
    void solve() override;

    //! prepare output
    void prepare_output() override;

    //! update time step
    void update() override;

    //! write output print to screen
    void output() override;

   protected:
    //! perform iteration step of structure field
    void do_poro_step() override;

    //! perform iteration step of scatra field
    void do_scatra_step() override;

    //! convergence check of outer loop
    bool convergence_check(int itnum);

    //! scalar increment of the outer loop
    Teuchos::RCP<Core::LinAlg::Vector<double>> scaincnp_;
    //! structure increment of the outer loop
    Teuchos::RCP<Core::LinAlg::Vector<double>> structincnp_;
    //! fluid increment of the outer loop
    Teuchos::RCP<Core::LinAlg::Vector<double>> fluidincnp_;

    //! maximum iteration steps
    int itmax_;
    //! convergence tolerance
    double ittol_;
  };

}  // namespace PoroElastScaTra


FOUR_C_NAMESPACE_CLOSE

#endif
