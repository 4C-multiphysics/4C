// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_PARTITIONED_2WC_HPP
#define FOUR_C_SSI_PARTITIONED_2WC_HPP

#include "4C_config.hpp"

#include "4C_ssi_input.hpp"
#include "4C_ssi_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN

namespace SSI
{
  //! base class to deal with partitioned 2WC SSI. Mainly it is the same as for thermo-structure
  //! interaction (TSI)
  class SSIPart2WC : public SSIPart
  {
   public:
    //! constructor
    SSIPart2WC(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);


    /*! \brief Initialize this object

    Hand in all objects/parameters/etc. from outside.
    Construct and manipulate internal objects.

    \note Try to only perform actions in init(), which are still valid
          after parallel redistribution of discretizations.
          If you have to perform an action depending on the parallel
          distribution, make sure you adapt the affected objects after
          parallel redistribution.
          Example: cloning a discretization from another discretization is
          OK in init(...). However, after redistribution of the source
          discretization do not forget to also redistribute the cloned
          discretization.
          All objects relying on the parallel distribution are supposed to
          the constructed in \ref setup().

    \warning none
    \return int

    */
    void init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;

    //! full time loop
    void timeloop() override;

    //! perform iteration loop between fields
    virtual void outer_loop();

    //! prepare time loop
    virtual void prepare_time_loop();

    //! prepare time step for single fields
    void prepare_time_step(bool printheader = true) override;

    //! update time step and print to screen
    virtual void update_and_output();

   protected:
    //! solve field 1
    virtual void operator1() { do_struct_step(); }

    //! solve field 2
    virtual void operator2() { do_scatra_step(); }

    //! pre operator called before first field operator
    virtual void pre_operator1() {}

    //! pre operator called before second field operator
    virtual void pre_operator2() {}

    //! post operator called after first field operator
    virtual void post_operator1() {}

    //! post operator called after second field operator
    virtual void post_operator2() {}

    //! perform iteration step of structure field and set the new disp and vel states in the scatra
    //! field
    void do_struct_step() override;

    //! perform iteration step of scatra field and set the new phi state in the structure field
    void do_scatra_step() override;

    //! update the current states in every iteration
    //! states are set to the last solutions obtained
    virtual void iter_update_states();

    //! convergence check of outer loop
    virtual bool convergence_check(int itnum);

    /// velocity calculation given the displacements
    std::shared_ptr<Core::LinAlg::Vector<double>> calc_velocity(
        const Core::LinAlg::Vector<double>& dispnp);

    //! scalar increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> scaincnp_;
    //! displacement increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> dispincnp_;

    //! convergence tolerance
    double ittol_ = -1.0;
    //! maximum iteration steps
    int itmax_ = -1;
  };

  //! class to deal with displacement relaxated partitioned 2WC SSI. Relaxation parameter is
  //! constant
  class SSIPart2WCSolidToScatraRelax : public SSIPart2WC
  {
   public:
    //! constructor
    SSIPart2WCSolidToScatraRelax(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup this object

     Initializes members and performs problem specific setup.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void

    */
    void init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

   protected:
    //! perform iteration loop between fields with relaxed displacements
    void outer_loop() override;

    //! //! calculate relaxation parameter
    virtual void calc_omega(double& omega, const int itnum);

    //! relaxation parameter
    double omega_;
  };

  //! class to deal with displacement relaxated 2WC partitioned SSI. Relaxation parameter is
  //! calculated via Aitken
  class SSIPart2WCSolidToScatraRelaxAitken : public SSIPart2WCSolidToScatraRelax
  {
   public:
    //! constructor
    SSIPart2WCSolidToScatraRelaxAitken(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup all class internal objects and members

    setup() is not supposed to have any input arguments !

    Must only be called after init().

    Construct all objects depending on the parallel distribution and
    relying on valid maps like, e.g. the state vectors, system matrices, etc.

    Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;

   protected:
    //! Calculate relaxation parameter via Aitken
    void calc_omega(double& omega, const int itnum) override;

    //! old displacement increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> dispincnpold_;
  };

  //! class to deal with scalar relaxated 2WC partitioned SSI. Relaxation parameter is constant
  class SSIPart2WCScatraToSolidRelax : public SSIPart2WC
  {
   public:
    //! constructor
    SSIPart2WCScatraToSolidRelax(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup this object

     Initializes members and performs problem specific setup.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void

    */
    void init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

   protected:
    //! perform iteration loop between fields with relaxed scalar
    void outer_loop() override;

    //! calculate relaxation parameter
    virtual void calc_omega(double& omega, const int itnum);

    //! relaxation parameter
    double omega_;
  };

  //! class to deal with scalar relaxated 2WC partitioned SSI. Relaxation parameter is calculated
  //! via Aitken
  class SSIPart2WCScatraToSolidRelaxAitken : public SSIPart2WCScatraToSolidRelax
  {
   public:
    //! constructor
    SSIPart2WCScatraToSolidRelaxAitken(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup all class internal objects and members

    setup() is not supposed to have any input arguments !

    Must only be called after init().

    Construct all objects depending on the parallel distribution and
    relying on valid maps like, e.g. the state vectors, system matrices, etc.

    Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;

   protected:
    //! Calculate relaxation parameter via Aitken
    void calc_omega(double& omega, const int itnum) override;

    //! old scatra increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> scaincnpold_;
  };
}  // namespace SSI

FOUR_C_NAMESPACE_CLOSE

#endif
