// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_VOLUMETRIC_SURFACEFLOW_CONDITION_HPP
#define FOUR_C_FLUID_VOLUMETRIC_SURFACEFLOW_CONDITION_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_io.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  namespace Utils
  {
    //--------------------------------------------------------------------
    // Wrapper class (to be called from outside) for volumetric surface
    // flow
    //--------------------------------------------------------------------

    /*!
    \brief Womersley boundary condition wrapper
    this class is meant to do some organisation stuff
    */
    class FluidVolumetricSurfaceFlowWrapper
    {
      friend class FluidImplicitTimeInt;


     public:
      /*!
      \brief Standard Constructor
      */
      FluidVolumetricSurfaceFlowWrapper(
          std::shared_ptr<Core::FE::Discretization> actdis, double dta);

      /*!
      \brief Destructor
      */
      virtual ~FluidVolumetricSurfaceFlowWrapper() = default;


      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::EvaluateVelocities
      */
      void evaluate_velocities(Core::LinAlg::Vector<double>& velocities, const double time);


      void insert_cond_vector(
          Core::LinAlg::Vector<double>& vec1, Core::LinAlg::Vector<double>& vec2);

      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::update_residual
      */
      void update_residual(std::shared_ptr<Core::LinAlg::Vector<double>> residual);


      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::Output
      */
      void output(Core::IO::DiscretizationWriter& output);

      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::read_restart
      */
      void read_restart(Core::IO::DiscretizationReader& reader);


     private:
      /*!
      \brief all single fluid volumetric surface flow conditions
      */
      std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>> fvsf_map_;

      //! fluid discretization
      std::shared_ptr<Core::FE::Discretization> discret_;

    };  // class FluidWomersleyWrapper

    class TotalTractionCorrector
    {
      friend class FluidImplicitTimeInt;


     public:
      /*!
      \brief Standard Constructor
      */

      TotalTractionCorrector(std::shared_ptr<Core::FE::Discretization> actdis, double dta);

      /*!
      \brief Destructor
      */
      virtual ~TotalTractionCorrector() = default;


      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::EvaluateVelocities
      */
      void evaluate_velocities(
          Core::LinAlg::Vector<double>& velocities, double time, double theta, double dta);

      /*!
      \brief export and set boundary values
      */
      void export_and_set_boundary_values(Core::LinAlg::Vector<double>& source,
          Core::LinAlg::Vector<double>& target, std::string name);

      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::update_residual
      */
      void update_residual(Core::LinAlg::Vector<double>& residual);


      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::Output
      */
      void output(Core::IO::DiscretizationWriter& output);

      /*!
      \brief Wrapper for FluidVolumetricSurfaceFlowBc::read_restart
      */
      void read_restart(Core::IO::DiscretizationReader& reader);


     private:
      /*!
      \brief all single fluid volumetric surface flow conditions
      */
      std::map<const int, std::shared_ptr<class FluidVolumetricSurfaceFlowBc>> fvsf_map_;

      //! fluid discretization
      std::shared_ptr<Core::FE::Discretization> discret_;

    };  // class TotalTractionCorrector



    //--------------------------------------------------------------------
    // Actual Womersley bc calculation stuff
    //--------------------------------------------------------------------
    /*!
    \brief Womersley boundary condition

    */
    class FluidVolumetricSurfaceFlowBc
    {
      friend class FluidVolumetricSurfaceFlowWrapper;
      friend class TotalTractionCorrector;
      //  friend class FluidSurfaceTotalTractionCorrectionWrapper;

     public:
      /*!
      \brief Standard Constructor
      */
      FluidVolumetricSurfaceFlowBc(std::shared_ptr<Core::FE::Discretization> actdis, double dta,
          std::string ds_condname, std::string dl_condname, int condid, int surf_numcond,
          int line_numcond);

      /*!
      \brief Empty Constructor
      */
      FluidVolumetricSurfaceFlowBc();

      /*!
      \brief Destructor
      */
      virtual ~FluidVolumetricSurfaceFlowBc() = default;

      /*!
      \brief calculates the center of mass
      */
      void center_of_mass_calculation(std::shared_ptr<std::vector<double>> coords,
          std::shared_ptr<std::vector<double>> normal, std::string ds_condname);


      /*!
      \brief calculates the local radii of all nodes
      */
      void eval_local_normalized_radii(std::string ds_condname, std::string dl_condname);

      /*!
      \brief get the node row map of the womersley condition
      */
      void build_condition_node_row_map(std::shared_ptr<Core::FE::Discretization> dis,
          std::string condname, int condid, int condnum,
          std::shared_ptr<Core::LinAlg::Map>& cond_noderowmap);

      /*!
      \brief get the dof row map of the womersley condition
      */
      void build_condition_dof_row_map(std::shared_ptr<Core::FE::Discretization> dis,
          const std::string condname, int condid, int condnum,
          std::shared_ptr<Core::LinAlg::Map>& cond_dofrowmap);

      /*!
      \brief Evaluate velocities
      */
      void evaluate_velocities(
          const double flowrate, const std::string ds_condname, const double time);


      /*!
      \brief Evaluate flowrate
      */
      double evaluate_flowrate(const std::string ds_condname, const double time);


      /*!
      \brief update_residual
      */
      void update_residual(Core::LinAlg::Vector<double>& residual);

      /*!
      \brief Evaluate velocities
      */
      void velocities(Core::FE::Discretization& disc, Core::LinAlg::Vector<double>& bcdof,
          Core::LinAlg::Map& cond_noderowmap, Core::LinAlg::Vector<double>& local_radii,
          Core::LinAlg::Vector<double>& border_radii, std::vector<double>& normal,
          Teuchos::ParameterList& params);

      /*!
      \brief Polynomail shaped velocity profile
      */
      double polynomail_velocity(double r, int order);

      /*!
      \brief Womersley shaped velocity profile
      */
      double womersley_velocity(double r, double R, double Bn,
          // complex<double> Bn,
          double phi, int n, double t);

      /*!
      \brief Corrects the flow rate
      */
      void correct_flow_rate(const Teuchos::ParameterList eleparams, const std::string ds_condname,
          const FLD::BoundaryAction action, const double time, const bool force_correction);

      /*!
      \brief Calculate the Flowrate on a boundary
      */
      double flow_rate_calculation(Teuchos::ParameterList eleparams, double time,
          std::string ds_condname, FLD::BoundaryAction action, int condid);

      double pressure_calculation(
          double time, std::string ds_condname, std::string action, int condid);
      /*!
      \brief Calculate the Flowrate on a boundary
      */
      void set_velocities(Core::LinAlg::Vector<double>& velocities);

      /*!
      \brief Reset condition velocities
      */
      void reset_velocities();

      /*!
      \brief evaluate the traction velocity component
      */
      void evaluate_traction_velocity_comp(Teuchos::ParameterList eleparams,
          std::string ds_condname, double flowrate, int condid, double time, double theta,
          double dta);

      /*!
      \brief export and set boundary values
      */
      void export_and_set_boundary_values(Core::LinAlg::Vector<double>& source,
          Core::LinAlg::Vector<double>& target, std::string name);

      /*!
      \brief reset traction velocity components
      */
      void reset_traction_velocity_comp();

      /*!
      \brief Calculate the Flowrate on a boundary
      */
      void dft(std::shared_ptr<std::vector<double>> f,
          std::shared_ptr<std::vector<std::complex<double>>>& F, int starting_pos);



     protected:
     private:
      /*!
      \brief calculate area at outflow boundary
      */
      double area(double& density, double& viscosity, std::string ds_condname, int condid);

      /*!
      \brief output
      */
      void output(Core::IO::DiscretizationWriter& output, std::string ds_condname, int condnum);

      /*!
      \brief Read restart
      */
      void read_restart(
          Core::IO::DiscretizationReader& reader, std::string ds_condname, int condnum);

      /*!
      \brief Bessel function of orders 0 and 1
      */
      std::complex<double> bessel_j01(std::complex<double> z, bool order);

      /*!
      \brief Interpolation function
      */
      void interpolate(
          std::vector<double>& V1, std::vector<double>& V2, int index1, int& index2, double period);

      /*!
      \brief Return prebiasing flag
       */

      std::string prebiasing_flag() { return prebiasing_flag_; }

     private:
      //! ID of present condition
      int condid_;

      //! Number of present surface condition
      int condnum_s_;

      //! Number of present line condition
      int condnum_l_;

      //! time period of present cyclic problem
      double period_;

      //! fluid viscosity
      double viscosity_;

      //! fluid density
      double density_;

      //! time step size
      double dta_;

      //! the processor ID from the communicator
      int myrank_;

      //! fluid discretization
      std::shared_ptr<Core::FE::Discretization> discret_;

      //! Flowrate array for Womersley conditions
      std::shared_ptr<std::vector<double>> flowrates_;

      //! Position at which the next element should be replaced
      //! initialised to zero as the first element will be replaced first
      int flowratespos_;

      //! center of mass coordinates
      std::shared_ptr<std::vector<double>> cmass_;

      //! average normal of the surface
      std::shared_ptr<std::vector<double>> normal_;

      //! direction normal of the velocity
      std::shared_ptr<std::vector<double>> vnormal_;

      //! a Node row map of the nodes that belong to the current condition
      std::shared_ptr<Core::LinAlg::Map> cond_surfnoderowmap_;

      //! a Node row map of the nodes that belong to border of the current condition
      std::shared_ptr<Core::LinAlg::Map> cond_linenoderowmap_;

      //! a Dof row map of the degrees of freedom that belong to the current condition
      std::shared_ptr<Core::LinAlg::Map> cond_dofrowmap_;

      //! A map of the local radii
      std::shared_ptr<Core::LinAlg::Vector<double>> local_radii_;

      //! A map of corresponding border radii
      std::shared_ptr<Core::LinAlg::Vector<double>> border_radii_;

      //! A map of only condition velocites
      std::shared_ptr<Core::LinAlg::Vector<double>> cond_velocities_;

      //! A dof col map of only condition velocites
      std::shared_ptr<Core::LinAlg::Vector<double>> drt_velocities_;

      //! A map of only condition velocites
      std::shared_ptr<Core::LinAlg::Vector<double>> cond_traction_vel_;

      //! initial area of the condition surface
      double area_;

      //! Number of modes
      int n_harmonics_;

      //! order of a polynomial velocity profile
      int order_;

      //! Type of the flow profile
      std::string flowprofile_type_;

      //! Prebiasing flag
      std::string prebiasing_flag_;

      //! is +1 if inflow, else -1
      double flow_dir_;

      //! flag to correct the flowprofile
      bool correct_flow_;


    };  // FluidVolumetricSurfaceFlowBc

  }  // namespace Utils
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
