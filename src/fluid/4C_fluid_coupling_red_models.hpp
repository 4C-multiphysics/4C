// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_COUPLING_RED_MODELS_HPP
#define FOUR_C_FLUID_COUPLING_RED_MODELS_HPP


#include "4C_config.hpp"

#include "4C_art_net_dyn_drt.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  namespace Utils
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
      FluidCouplingWrapperBase(std::shared_ptr<Core::FE::Discretization> dis_3D,
          std::shared_ptr<Core::FE::Discretization> dis_redD,
          Core::IO::DiscretizationWriter& output, double dt_3d, double dt_redD);

      /*!
      \brief Destructor
      */
      virtual ~FluidCouplingWrapperBase() = default;

      /*!
      \brief Wrapper for FluidCouplingWrapper::flow_rate_calculation
      */
      void flow_rate_calculation(double time, double dta);

      /*!
      \brief Wrapper for FluidCouplingWrapper::pressure_calculation
      */
      void pressure_calculation(double time, double dta);

      /*!
      \brief Wrapper for FluidCouplingWrapper::outflow_boundary
      */
      void apply_boundary_conditions(double time, double dta, double theta);

      /*!
      \brief Wrapper for FluidCouplingWrapper::update_residual
      */
      void update_residual(Core::LinAlg::Vector<double>& residual);



      void evaluate_dirichlet(
          Core::LinAlg::Vector<double>& velnp, const Core::LinAlg::Map& condmap, double time);

      /*!
      \brief Wrapper for FluidCouplingWrapper::write_restart
      */
      void write_restart(Core::IO::DiscretizationWriter& output);

      /*!
      \brief Wrapper for FluidCouplingWrapper::read_restart
      */
      void read_restart(Core::IO::DiscretizationReader& reader);


      virtual void integrate(bool flag, std::shared_ptr<Teuchos::ParameterList>&) = 0;

      virtual void save_state() = 0;

      virtual void load_state() = 0;

      /*!
      \brief compute TimeUpdate
      */
      virtual void time_update() = 0;

     private:
      /*!
      \brief all single coupling conditions
      */
      std::map<const int, std::shared_ptr<FluidCouplingBc>> coup_map_3d_;

      //! map of coupling variables returned by the reduced-D model at time step n+1
      std::shared_ptr<std::map<std::string, double>> map_red_dnp_;

      //! map of coupling variables returned by the reduced-D model at time step n
      std::shared_ptr<std::map<std::string, double>> map_red_dn_;

      //! map of coupling variables returned by the 3-D model at time step n+1
      std::shared_ptr<std::map<std::string, double>> map3_dnp_;

      //! map of coupling variables returned by the 3-D model at time step n
      std::shared_ptr<std::map<std::string, double>> map3_dn_;

      //! 3D fluid discretization
      std::shared_ptr<Core::FE::Discretization> discret_3d_;

      //! Reduced-D artery network discretization
      std::shared_ptr<Core::FE::Discretization> discret_red_d_;

      //! Reduced-D artery network time integration
      //  std::shared_ptr<Arteries::ArtNetExplicitTimeInt>              ArtExpTime_integ_;


      //! the output writer
      Core::IO::DiscretizationWriter& output_;

      //! the fluid 3D time step size
      double dt_f3_;

      //! the reduced model time step size
      double dt_rm_;

    };  // class FluidCouplingWrapper

    template <class RedDTimeInt>

    class FluidCouplingWrapper : public FluidCouplingWrapperBase
    {
     public:
      FluidCouplingWrapper(std::shared_ptr<Core::FE::Discretization> dis_3D,
          std::shared_ptr<Core::FE::Discretization> dis_redD,
          std::shared_ptr<RedDTimeInt> time_intg, Core::IO::DiscretizationWriter& output,
          double dt_3d, double dt_rm)
          : FluidCouplingWrapperBase(dis_3D, dis_redD, output, dt_3d, dt_rm),
            reduced_d_time_integ_(time_intg)
      {
      }

      void integrate(bool flag, std::shared_ptr<Teuchos::ParameterList>& params) override
      {
        reduced_d_time_integ_->integrate(true, params);
      }

      void save_state() override { reduced_d_time_integ_->save_state(); }

      void load_state() override { reduced_d_time_integ_->load_state(); }

      void time_update() override { reduced_d_time_integ_->time_update(); }

     private:
      //! Reduced-D artery network time integration
      std::shared_ptr<RedDTimeInt> reduced_d_time_integ_;
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
      FluidCouplingBc(std::shared_ptr<Core::FE::Discretization> dis_3D,
          std::shared_ptr<Core::FE::Discretization> dis_reD, Core::IO::DiscretizationWriter& output,
          double dt_3d, double dt_rm, int condid, int numcond, int numcond2);

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
      void write_restart(Core::IO::DiscretizationWriter& output, int condnum) const;

      /*!
      \brief read flowrates_ and flowratespos_
      */
      void read_restart(Core::IO::DiscretizationReader& reader, int condnum);


      /*!
        \brief compute and store flow rate of all previous
        time steps belonging to one cycle
      */
      double flow_rate_calculation(double time, double dta, int condid);

      /*!
        \brief compute and store pressure of all previous
        time steps belonging to one cycle
      */
      double pressure_calculation(double time, double dta, int condid);


      /*!
        \brief compute convolution integral and apply pressure
        to elements
      */
      void outflow_boundary(double pressure, double time, double dta, double theta, int condid);

      /*!
        \brief compute apply inflow as a Dirichlet BC
      */
      void inflow_boundary(double flowrate, double time, double dta, double theta, int condid);

      void update_residual(Core::LinAlg::Vector<double>& residual);



      void evaluate_dirichlet(
          Core::LinAlg::Vector<double>& velnp, const Core::LinAlg::Map& condmap, double time);

      /*!
      \brief compute TimeUpdate
      */
      void time_update() {}

      void integrate(bool flag, Teuchos::ParameterList& params) {}

      void save_state() {};

      void load_state() {};

     private:
      /*!
      \brief calculate area at outflow boundary
      */
      double area(double& density, double& viscosity, int condid);



     protected:
      // coupled neumann BC
      std::shared_ptr<Core::LinAlg::Vector<double>> couplingbc_;

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
      std::shared_ptr<Core::FE::Discretization> discret_3d_;

      //! fluid discretization
      std::shared_ptr<Core::FE::Discretization> discret_red_d_;

      //! the output writer
      Core::IO::DiscretizationWriter& output_;

      //! flow rate
      double flowrate_;

      //! pressure
      double pressure_;

      //! corrector variable for dirichlet velocity s.t. applied flowrate is correct
      double alfa_;

    };  // class FluidCouplingBc

  }  // namespace Utils
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
