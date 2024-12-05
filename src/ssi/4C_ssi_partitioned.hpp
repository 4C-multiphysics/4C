// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSI_PARTITIONED_HPP
#define FOUR_C_SSI_PARTITIONED_HPP

#include "4C_config.hpp"

#include "4C_ssi_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace SSI
{
  class SSIPart : public SSIBase
  {
   public:
    /// setup SSI algorithm
    SSIPart(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /*!
    \brief Initialize this object

     Initializes members and performs problem specific setup.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void
    \date 08/16
    \author rauch
    */
    void init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

    void read_restart(int restart) override;

    /*!
    \brief Setup vectors and pointers

     Initializes members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void
    \date 08/16
    \author rauch
    */
    void setup() override;

    /// time loop of coupled problem
    void timeloop() override = 0;

   protected:
    /// prepare time step of single fields
    virtual void prepare_time_step(bool printheader = true) = 0;

    /// do one time step (one way coupled) or one inner iteration loop step (two way coupled),
    /// depending on coupling algorithm
    virtual void do_struct_step() = 0;

    /// do one time step (one way coupled) or one inner iteration loop step (two way coupled),
    /// depending on coupling algorithm
    virtual void do_scatra_step() = 0;

   private:
    //! set up structural model evaluator for scalar-structure interaction
    void setup_model_evaluator() override;
  };

}  // namespace SSI

FOUR_C_NAMESPACE_CLOSE

#endif
