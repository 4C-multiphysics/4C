// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_IMPEDANCECONDITION_HPP
#define FOUR_C_FLUID_IMPEDANCECONDITION_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>
#include <set>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
}  // namespace Core::IO
namespace Core::LinAlg
{
  class MultiMapExtractor;
  class SparseOperator;
}  // namespace Core::LinAlg


namespace FLD
{
  namespace Utils
  {
    class FluidImpedanceWrapper
    {
      // friend class FLD::FluidImplicitTimeInt;

     public:
      /*!
      \brief Standard Constructor
      */
      FluidImpedanceWrapper(const std::shared_ptr<Core::FE::Discretization> actdis);

      /*!
      \brief Destructor
      */
      virtual ~FluidImpedanceWrapper() = default;

      /*!
        \brief Wrapper for FluidImpedacnceBc::use_block_matrix
      */
      void use_block_matrix(std::shared_ptr<std::set<int>> condelements,
          const Core::LinAlg::MultiMapExtractor& domainmaps,
          const Core::LinAlg::MultiMapExtractor& rangemaps, bool splitmatrix);

      /*!
      \brief Calculate impedance tractions and add it to fluid residual and linearisation
      */
      void add_impedance_bc_to_residual_and_sysmat(const double dta, const double time,
          Core::LinAlg::Vector<double>& residual, Core::LinAlg::SparseOperator& sysmat);

      /*!
      \brief Wrap for time update of impedance conditions
      */
      void time_update_impedances(const double time);

      /*!
      \brief Wrapper for FluidImpedacnceBc::write_restart
      */
      void write_restart(Core::IO::DiscretizationWriter& output);

      /*!
      \brief Wrapper for FluidImpedacnceBc::read_restart
      */
      void read_restart(Core::IO::DiscretizationReader& reader);

      /*!
      \brief return vector of relative pressure errors of last cycle
      */
      std::vector<double> get_w_krelerrors();

     private:
      /*!
      \brief all single impedance conditions
      */
      std::map<const int, std::shared_ptr<class FluidImpedanceBc>> impmap_;

    };  // class FluidImpedanceWrapper



    //--------------------------------------------------------------------
    // Actual impedance bc calculation stuff
    //--------------------------------------------------------------------
    /*!
    \brief impedance boundary condition for vascular outflow boundaries

    */
    class FluidImpedanceBc
    {
      friend class FluidImpedanceWrapper;

     public:
      /*!
      \brief Standard Constructor
      */
      FluidImpedanceBc(const std::shared_ptr<Core::FE::Discretization> actdis, const int condid,
          Core::Conditions::Condition* impedancecond);

      /*!
      \brief Empty Constructor
      */
      FluidImpedanceBc();

      /*!
      \brief Destructor
      */
      virtual ~FluidImpedanceBc() = default;

     protected:
      /*!
      \brief Split linearization matrix to a BlockSparseMatrixBase
      */
      void use_block_matrix(std::shared_ptr<std::set<int>> condelements,
          const Core::LinAlg::MultiMapExtractor& domainmaps,
          const Core::LinAlg::MultiMapExtractor& rangemaps, bool splitmatrix);

      /*!
        \brief compute and store flow rate of all previous
        time steps belonging to one cycle
      */
      void flow_rate_calculation(const int condid);

      /*!
        \brief compute convolution integral and apply pressure
        to elements
      */
      void calculate_impedance_tractions_and_update_residual_and_sysmat(
          Core::LinAlg::Vector<double>& residual, Core::LinAlg::SparseOperator& sysmat,
          const double dta, const double time, const int condid);

      /*!
        \brief Update flowrate and pressure vector
      */
      void time_update_impedance(const double time, const int condid);

      /*!
      \brief write flowrates_ and flowratespos_ to result files
      */
      void write_restart(Core::IO::DiscretizationWriter& output, const int condnum);

      /*!
      \brief read flowrates_ and flowratespos_
      */
      void read_restart(Core::IO::DiscretizationReader& reader, const int condnum);

     private:
      /*!
      \brief calculate area at outflow boundary
      */
      double area(const int condid);

      /*!
      \brief return relative error of last cycle
      */
      double get_w_krelerror() { return w_krelerror_; }

     private:
      //! fluid discretization
      const std::shared_ptr<Core::FE::Discretization> discret_;

      //! the processor ID from the communicator
      const int myrank_;

      //! one-step theta time integration factor
      double theta_;

      //! condition type ( implemented so far: windkessel, ressistive )
      const std::string treetype_;

      //! time period of present cyclic problem
      const double period_;

      //! 'material' parameters required for artery tree
      const double r1_, r2_, c_;

      //! function number
      const Core::IO::Noneable<int> functnum_;

      //! traction vector for impedance bc
      std::shared_ptr<Core::LinAlg::Vector<double>> impedancetbc_;

      //! linearisation of traction vector
      std::shared_ptr<Core::LinAlg::SparseOperator> impedancetbcsysmat_;

      //! Pressure at time step n+1
      double p_np_;

      //! Pressure at time step n
      double p_n_;

      //! Flux at time step n+1
      double q_np_;

      //! Flux at time step n
      double q_n_;

      //! variable describing the relative error between pressure at (n+1)T and at (n)T
      double w_krelerror_;

      //! Pressure at beginning of the period
      double p_0_;
    };  // class FluidImpedanceBc

  }  // namespace Utils
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
