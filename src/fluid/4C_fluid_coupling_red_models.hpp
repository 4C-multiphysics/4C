/*-----------------------------------------------------------*/
/*! \file

\brief Method to deal with coupling between 3D fluid and 0/1D vascular
problem


\level 3
*/
/*-----------------------------------------------------------*/

// #ifdef D_COUPLED_ARTNET
#ifndef FOUR_C_FLUID_COUPLING_RED_MODELS_HPP
#define FOUR_C_FLUID_COUPLING_RED_MODELS_HPP


#include "4C_config.hpp"

#include "4C_art_net_dyn_drt.hpp"
#include "4C_io.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_red_airways_dyn_drt.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  namespace UTILS
  {
    //--------------------------------------------------------------------
    // Wrapper class (to be called from outside) for coupling 3D/red D  bc
    //--------------------------------------------------------------------

    /*!
    \brief coupling boundary condition wrapper
    this class is meant to do some organisation stuff

    */

    class FluidCouplingBc;

    class FluidCouplingWrapperBase
    {
      friend class FluidImplicitTimeInt;

     public:
      /*!
      \brief Standard Constructor
      */
      FluidCouplingWrapperBase(Teuchos::RCP<DRT::Discretization> dis_3D,
          Teuchos::RCP<DRT::Discretization> dis_redD, IO::DiscretizationWriter& output,
          double dt_3d, double dt_redD);

      /*!
      \brief Destructor
      */
      virtual ~FluidCouplingWrapperBase() = default;

      /*!
      \brief Wrapper for FluidCouplingWrapper::FlowRateCalculation
      */
      void FlowRateCalculation(double time, double dta);

      /*!
      \brief Wrapper for FluidCouplingWrapper::PressureCalculation
      */
      void PressureCalculation(double time, double dta);

      /*!
      \brief Wrapper for FluidCouplingWrapper::OutflowBoundary
      */
      void apply_boundary_conditions(double time, double dta, double theta);

      /*!
      \brief Wrapper for FluidCouplingWrapper::UpdateResidual
      */
      void UpdateResidual(Teuchos::RCP<Epetra_Vector> residual);



      void EvaluateDirichlet(
          Teuchos::RCP<Epetra_Vector> velnp, const Epetra_Map& condmap, double time);

      /*!
      \brief Wrapper for FluidCouplingWrapper::WriteRestart
      */
      void WriteRestart(IO::DiscretizationWriter& output);

      /*!
      \brief Wrapper for FluidCouplingWrapper::read_restart
      */
      void read_restart(IO::DiscretizationReader& reader);


      virtual void Integrate(bool flag, Teuchos::RCP<Teuchos::ParameterList>&) = 0;

      virtual void SaveState() = 0;

      virtual void LoadState() = 0;

      /*!
      \brief compute TimeUpdate
      */
      virtual void TimeUpdate() = 0;

     private:
      /*!
      \brief all single coupling conditions
      */
      std::map<const int, Teuchos::RCP<FluidCouplingBc>> coup_map3_d_;

      //! map of coupling variables returned by the reduced-D model at time step n+1
      Teuchos::RCP<std::map<std::string, double>> map_red_dnp_;

      //! map of coupling variables returned by the reduced-D model at time step n
      Teuchos::RCP<std::map<std::string, double>> map_red_dn_;

      //! map of coupling variables returned by the 3-D model at time step n+1
      Teuchos::RCP<std::map<std::string, double>> map3_dnp_;

      //! map of coupling variables returned by the 3-D model at time step n
      Teuchos::RCP<std::map<std::string, double>> map3_dn_;

      //! 3D fluid discretization
      Teuchos::RCP<DRT::Discretization> discret3_d_;

      //! Reduced-D artery network discretization
      Teuchos::RCP<DRT::Discretization> discret_red_d_;

      //! Reduced-D artery network time integration
      //  Teuchos::RCP<ART::ArtNetExplicitTimeInt>              ArtExpTime_integ_;


      //! the output writer
      IO::DiscretizationWriter& output_;

      //! the fluid 3D time step size
      double dt_f3_;

      //! the reduced model time step size
      double dt_rm_;

    };  // class FluidCouplingWrapper

    template <class red_D_time_int>

    class FluidCouplingWrapper : public FluidCouplingWrapperBase
    {
     public:
      FluidCouplingWrapper(Teuchos::RCP<DRT::Discretization> dis_3D,
          Teuchos::RCP<DRT::Discretization> dis_redD, Teuchos::RCP<red_D_time_int> time_intg,
          IO::DiscretizationWriter& output, double dt_3d, double dt_rm)
          : FluidCouplingWrapperBase(dis_3D, dis_redD, output, dt_3d, dt_rm),
            reduced_d_time_integ_(time_intg)
      {
      }

      void Integrate(bool flag, Teuchos::RCP<Teuchos::ParameterList>& params) override
      {
        reduced_d_time_integ_->Integrate(true, params);
      }

      void SaveState() override { reduced_d_time_integ_->SaveState(); }

      void LoadState() override { reduced_d_time_integ_->LoadState(); }

      void TimeUpdate() override { reduced_d_time_integ_->TimeUpdate(); }

     private:
      //! Reduced-D artery network time integration
      Teuchos::RCP<red_D_time_int> reduced_d_time_integ_;
    };

    //--------------------------------------------------------------------
    // Actual coupling bc calculation
    //--------------------------------------------------------------------
    /*!
    \brief coupling boundary condition for vascular outflow boundaries

    */

    class FluidCouplingBc
    {
      friend class FluidCouplingWrapperBase;

     public:
      /*!
      \brief Standard Constructor
      */
      FluidCouplingBc(Teuchos::RCP<DRT::Discretization> dis_3D,
          Teuchos::RCP<DRT::Discretization> dis_reD, IO::DiscretizationWriter& output, double dt_3d,
          double dt_rm, int condid, int numcond, int numcond2);

      /*!
      \brief Empty Constructor
      */
      FluidCouplingBc();

      /*!
      \brief Destructor
      */
      virtual ~FluidCouplingBc() = default;

     protected:
      /*!
      \brief write flowrates_ and flowratespos_ to result files
      */
      void WriteRestart(IO::DiscretizationWriter& output, int condnum);

      /*!
      \brief read flowrates_ and flowratespos_
      */
      void read_restart(IO::DiscretizationReader& reader, int condnum);


      /*!
        \brief compute and store flow rate of all previous
        time steps belonging to one cycle
      */
      double FlowRateCalculation(double time, double dta, int condid);

      /*!
        \brief compute and store pressure of all previous
        time steps belonging to one cycle
      */
      double PressureCalculation(double time, double dta, int condid);


      /*!
        \brief compute convolution integral and apply pressure
        to elements
      */
      void OutflowBoundary(double pressure, double time, double dta, double theta, int condid);

      /*!
        \brief compute apply inflow as a Dirichlet BC
      */
      void InflowBoundary(double flowrate, double time, double dta, double theta, int condid);

      void UpdateResidual(Teuchos::RCP<Epetra_Vector> residual);



      void EvaluateDirichlet(
          Teuchos::RCP<Epetra_Vector> velnp, const Epetra_Map& condmap, double time);

      /*!
      \brief compute TimeUpdate
      */
      void TimeUpdate() {}

      void Integrate(bool flag, Teuchos::RCP<Teuchos::ParameterList>& params) {}

      void SaveState(){};

      void LoadState(){};

     private:
      /*!
      \brief calculate area at outflow boundary
      */
      double area(double& density, double& viscosity, int condid);



     protected:
      // coupled neumann BC
      Teuchos::RCP<Epetra_Vector> couplingbc_;

     private:
      //! ID of present condition
      int condid_;

      //! 3D fluid time step size
      double dt_f3_;

      //! reduced-D time step size
      double dt_rm_;

      //! coupling error at the boundary
      double max_error_;

      //! number of maximum allowable iterations at the boundary
      double max_itr_;

      //! velocity
      double velocity_;

      //! the processor ID from the communicator
      int myrank_;

      //! 3D fluid discretization
      Teuchos::RCP<DRT::Discretization> discret_3_d_;

      //! fluid discretization
      Teuchos::RCP<DRT::Discretization> discret_red_d_;

      //! the output writer
      IO::DiscretizationWriter& output_;

      //! flow rate
      double flowrate_;

      //! pressure
      double pressure_;

      //! corrector variable for dirichlet velocity s.t. applied flowrate is correct
      double alfa_;

    };  // class FluidCouplingBc

  }  // namespace UTILS
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
