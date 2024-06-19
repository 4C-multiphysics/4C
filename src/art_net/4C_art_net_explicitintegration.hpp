
/*! \file
\brief Associated with control routine for artery solvers,

     including instationary solvers based on

     o two-step Taylor-Galerkin

\level 2




*----------------------------------------------------------------------*/

#ifndef FOUR_C_ART_NET_EXPLICITINTEGRATION_HPP
#define FOUR_C_ART_NET_EXPLICITINTEGRATION_HPP

#include "4C_config.hpp"

#include "4C_art_net_art_junction.hpp"
#include "4C_art_net_art_write_gnuplot.hpp"
#include "4C_art_net_timint.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_function.hpp"

#include <Epetra_MpiComm.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

#include <cstdlib>
#include <ctime>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

namespace Arteries
{
  /*!
  \brief time integration for arterial network problems

  */
  class ArtNetExplicitTimeInt : public TimInt
  {
    // friend class ArtNetResultTest;

   public:
    /*!
    \brief Standard Constructor

    */
    ArtNetExplicitTimeInt(Teuchos::RCP<Core::FE::Discretization> dis, const int linsolvernumber,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& artparams,
        Core::IO::DiscretizationWriter& output);



    /*!
    \brief Initialization

    */
    void Init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& arteryparams, const std::string& scatra_disname) override;

    // create field test
    Teuchos::RCP<Core::UTILS::ResultTest> CreateFieldTest() override;


    /*!
    \brief solve linearised artery and bifurcation

    */
    void Solve(Teuchos::RCP<Teuchos::ParameterList> CouplingTo3DParams) override;

    void SolveScatra() override;

    /*!
      \brief build linear system matrix and rhs


      \param vel new guess at velocity, cross-sectional area, and pressure
    */
    void evaluate(Teuchos::RCP<const Epetra_Vector> vel){};

    /*!
    \brief Update the solution after convergence of the linear
           iteration. Current solution becomes old solution of next
           timestep.
    */
    void TimeUpdate() override;

    /*!
    \brief Initialize the saving state vectors
    */
    void InitSaveState() override;

    /*!
    \brief Save the current vectors into the saving state vectors
    */
    void SaveState() override;

    /*!
    \brief Load the currently saved state vectors into the currently used vectors
    */
    void LoadState() override;

    /*!
    \brief update configuration and output to file/screen

    */
    void Output(bool CoupledTo3D, Teuchos::RCP<Teuchos::ParameterList> CouplingParams) override;

    /*!
    \brief Test results

    */
    void TestResults() override;

    /*!
    \brief calculate values that could be used for postprocessing
           such as pressure and flowrate.
    */
    void calc_postprocessing_values();


    void calc_scatra_from_scatra_fw(
        Teuchos::RCP<Epetra_Vector> scatra, Teuchos::RCP<Epetra_Vector> scatra_fb);

    /*!
    \brief read restart data

    */
    void read_restart(int step, bool CoupledTo3D = false) override;

    //! @name access methods for composite algorithms

    //  Teuchos::RCP<Epetra_Vector> Residual() { return residual_; } //This variable might be needed
    //  in future!
    Teuchos::RCP<Epetra_Vector> Qnp() { return qnp_; }
    Teuchos::RCP<Epetra_Vector> QAnp() { return qanp_; }
    Teuchos::RCP<Epetra_Vector> Areanp() { return areanp_; }
    // Teuchos::RCP<Epetra_Vector> Presnp() { return presnp_; }
    Teuchos::RCP<Epetra_Vector> Qn() { return qn_; }
    Teuchos::RCP<Epetra_Vector> QAn() { return qan_; }
    Teuchos::RCP<Epetra_Vector> Arean() { return arean_; }
    // Teuchos::RCP<Epetra_Vector> Presn()  { return presn_; }

    /// provide access to the Dirichlet map
    Teuchos::RCP<const Core::LinAlg::MapExtractor> DirichMaps() { return dbcmaps_; }

    /// Extract the Dirichlet toggle vector based on Dirichlet BC maps
    ///
    /// This method provides backward compatability only. Formerly, the Dirichlet conditions
    /// were handled with the Dirichlet toggle vector. Now, they are stored and applied
    /// with maps, ie #dbcmaps_. Eventually, this method will be removed.
    const Teuchos::RCP<const Epetra_Vector> Dirichlet();

    /// Extract the Inverse Dirichlet toggle vector based on Dirichlet BC maps
    ///
    /// This method provides backward compatability only. Formerly, the Dirichlet conditions
    /// were handled with the Dirichlet toggle vector. Now, they are stored and applied
    /// with maps, ie #dbcmaps_. Eventually, this method will be removed.
    const Teuchos::RCP<const Epetra_Vector> InvDirichlet();

    Teuchos::RCP<Core::LinAlg::SparseMatrix> MassMatrix()
    {
      return Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(massmat_);
    }

    //@}


   protected:
    /// (standard) mass matrix
    Teuchos::RCP<Core::LinAlg::SparseOperator> massmat_;

    /// maps for scatra Dirichlet and free DOF sets
    Teuchos::RCP<Epetra_Vector> nodeIds_;
    Teuchos::RCP<Epetra_Vector> scatra_bcval_;
    Teuchos::RCP<Epetra_Vector> scatra_dbctog_;

    //! @name Volumetric Flow rate and Cross-Sectional area at time n+1, n and n-1
    Teuchos::RCP<Epetra_Vector> qanp_;
    Teuchos::RCP<Epetra_Vector> qan_;
    Teuchos::RCP<Epetra_Vector> qanm_;
    //@}

    //! @name Volumetric Flow rate and Cross-Sectional area at time n before solving Fluid 3D
    Teuchos::RCP<Epetra_Vector> qan_3D_;
    //@}

    //! @name Volumetric Flow rate at time n+1, n and n-1
    Teuchos::RCP<Epetra_Vector> qnp_;
    Teuchos::RCP<Epetra_Vector> qn_;
    Teuchos::RCP<Epetra_Vector> qnm_;
    //@}

    //! @name Pressure at time n
    Teuchos::RCP<Epetra_Vector> pn_;
    //@}

    //! @name Area at time n
    Teuchos::RCP<Epetra_Vector> an_;
    //@}

    //! @name Forward and backwar characteristic wave speeds at time n+1, n and n-1
    Teuchos::RCP<Epetra_Vector> Wfo_;
    Teuchos::RCP<Epetra_Vector> Wbo_;
    Teuchos::RCP<Epetra_Vector> Wfnp_;
    Teuchos::RCP<Epetra_Vector> Wfn_;
    Teuchos::RCP<Epetra_Vector> Wfnm_;
    Teuchos::RCP<Epetra_Vector> Wbnp_;
    Teuchos::RCP<Epetra_Vector> Wbn_;
    Teuchos::RCP<Epetra_Vector> Wbnm_;
    //@}

    //! @name scalar transport vectors at time n+1, n and n-1
    Teuchos::RCP<Epetra_Vector> scatraO2nm_;
    Teuchos::RCP<Epetra_Vector> scatraO2n_;
    Teuchos::RCP<Epetra_Vector> scatraO2np_;
    Teuchos::RCP<Epetra_Vector> scatraO2wfn_;
    Teuchos::RCP<Epetra_Vector> scatraO2wfnp_;
    Teuchos::RCP<Epetra_Vector> scatraO2wbn_;
    Teuchos::RCP<Epetra_Vector> scatraO2wbnp_;

    Teuchos::RCP<Epetra_Vector> scatraCO2n_;
    Teuchos::RCP<Epetra_Vector> scatraCO2np_;
    Teuchos::RCP<Epetra_Vector> scatraCO2wfn_;
    Teuchos::RCP<Epetra_Vector> scatraCO2wfnp_;
    Teuchos::RCP<Epetra_Vector> scatraCO2wbn_;
    Teuchos::RCP<Epetra_Vector> scatraCO2wbnp_;

    Teuchos::RCP<Epetra_Vector> export_scatra_;
    //@}

    //! @name saving state vectors
    Teuchos::RCP<Epetra_Vector> saved_qanp_;
    Teuchos::RCP<Epetra_Vector> saved_qan_;
    Teuchos::RCP<Epetra_Vector> saved_qanm_;

    Teuchos::RCP<Epetra_Vector> saved_Wfnp_;
    Teuchos::RCP<Epetra_Vector> saved_Wfn_;
    Teuchos::RCP<Epetra_Vector> saved_Wfnm_;

    Teuchos::RCP<Epetra_Vector> saved_Wbnp_;
    Teuchos::RCP<Epetra_Vector> saved_Wbn_;
    Teuchos::RCP<Epetra_Vector> saved_Wbnm_;

    Teuchos::RCP<Epetra_Vector> saved_scatraO2np_;
    Teuchos::RCP<Epetra_Vector> saved_scatraO2n_;
    Teuchos::RCP<Epetra_Vector> saved_scatraO2nm_;
    //@}

    //! @name cross-sectional area at time n+1, n and n-1
    Teuchos::RCP<Epetra_Vector> arean_;
    Teuchos::RCP<Epetra_Vector> areanp_;
    Teuchos::RCP<Epetra_Vector> areanm_;
    //@}

    //! @name Dirichlet boundary condition vectors
    Teuchos::RCP<Epetra_Vector> bcval_;
    Teuchos::RCP<Epetra_Vector> dbctog_;
    //@}

    //! @name Junction boundary condition
    Teuchos::RCP<UTILS::ArtJunctionWrapper> artjun_;
    //@}

    //! @name 1D artery values at the junctions
    Teuchos::RCP<std::map<const int, Teuchos::RCP<Arteries::UTILS::JunctionNodeParams>>>
        junc_nodal_vals_;
    //@}

    //! @name A condition to export 1D arteries as a gnuplot format
    Teuchos::RCP<UTILS::ArtWriteGnuplotWrapper> artgnu_;
    //@}

    //@}

  };  // class ArtNetExplicitTimeInt

}  // namespace Arteries


FOUR_C_NAMESPACE_CLOSE

#endif
