/*----------------------------------------------------------------------*/
/*! \file
 \brief one way coupled partitioned scalar structure interaction

 \level 2


 *------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_SSI_PARTITIONED_1WC_HPP
#define FOUR_C_SSI_PARTITIONED_1WC_HPP

#include "4C_config.hpp"

#include "4C_ssi_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN

namespace SSI
{
  class SSIPart1WC : public SSIPart
  {
   public:
    //! constructor
    SSIPart1WC(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup this object

     Initializes members and performs problem specific setup.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void
    \date 08/16
    \author rauch
    */
    void Init(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

    //! actual time loop (implemented by derived class)
    void Timeloop() override = 0;

   protected:
    //! prepare time step for single fields
    void PrepareTimeStep(bool printheader = true) override = 0;

    //! perform one time step of structure field
    void DoStructStep() override;

    //! perform one time step of scatra field
    void DoScatraStep() override;

    // Flag for reading scatra result from restart files
    bool isscatrafromfile_;
  };

  class SSIPart1WCSolidToScatra : public SSIPart1WC
  {
   public:
    //! constructor
    SSIPart1WCSolidToScatra(
        const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup this object

     Initializes members and performs problem specific setup.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void
    \date 08/16
    \author rauch
    */
    void Init(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

    //! actual time loop
    void Timeloop() override;

    //! prepare time step for single fields
    void PrepareTimeStep(bool printheader = true) override;
  };

  class SSIPart1WCScatraToSolid : public SSIPart1WC
  {
   public:
    //! constructor
    SSIPart1WCScatraToSolid(
        const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams);


    /*!
    \brief Setup this object

     Initializes members and performs problem specific setup.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, vectors may have wrong maps.

    \warning none
    \return void
    \date 08/16
    \author rauch
    */
    void Init(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
        const std::string& struct_disname, const std::string& scatra_disname, bool isAle) override;

    //! actual time loop
    void Timeloop() override;

    //! prepare time step for single fields
    void PrepareTimeStep(bool printheader = true) override;

    //! return, if time loop has finished
    bool Finished() const;
  };

}  // namespace SSI

FOUR_C_NAMESPACE_CLOSE

#endif
