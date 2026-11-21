// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_CONSTRAINTMIXTURE_HISTORY_HPP
#define FOUR_C_MAT_CONSTRAINTMIXTURE_HISTORY_HPP

#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class ConstraintMixtureHistoryType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ConstraintMixtureHistoryType"; }

    static ConstraintMixtureHistoryType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ConstraintMixtureHistoryType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for history of constraint mixture material
  class ConstraintMixtureHistory : public Core::Communication::ParObject
  {
    friend class ConstraintMixture;

   public:
    /// construct empty history object
    ConstraintMixtureHistory() { ; }

    //! @name Packing and Unpacking
    int unique_par_object_id() const override
    {
      return ConstraintMixtureHistoryType::instance().unique_par_object_id();
    }
    void pack(Core::Communication::PackBuffer& data) const override;
    void unpack(Core::Communication::UnpackBuffer& buffer) override;
    //@}

    /// Setup
    void setup(int ngp, const double massprodbasal, bool expvar);

   private:
    /// @name Access to History
    //@{
    /// set time variables
    void set_time(double deptime, double dt)
    {
      depositiontime_ = deptime;
      dt_ = dt;
    };
    /// get time variables
    void get_time(double* deptime, double* dt)
    {
      *deptime = depositiontime_;
      *dt = dt_;
    };
    /// set stretches
    void set_stretches(int gp, Core::LinAlg::Matrix<4, 1> stretches);
    /// get stretches
    void get_stretches(int gp, Core::LinAlg::Matrix<4, 1>* stretches);
    /// set mass production rates
    void set_mass(int gp, Core::LinAlg::Matrix<4, 1> massprod);
    /// set mass production rate of single fiber
    void set_mass(int gp, double massprod, int idfiber);
    /// get mass production rates
    void get_mass(int gp, Core::LinAlg::Matrix<4, 1>* massprod);
    /// set vardegrad
    void set_var_degrad(int gp, int idfiber, double vardegrad);
    /// get vardegrad
    void get_var_degrad(int gp, int idfiber, double* vardegrad);
    /// return number of gausspoints
    int num_gp() const { return numgp_; }
    //@}

    /// deposition time of collagen fibers
    double depositiontime_;
    /// time step at this time
    double dt_;
    /// number of gausspoints
    int numgp_;
    /// variable degradation of collagen?
    bool expvar_;
    /// stretch of first collagen fiber family
    std::shared_ptr<std::vector<double>> collagenstretch1_;
    /// stretch of second collagen fiber family
    std::shared_ptr<std::vector<double>> collagenstretch2_;
    /// stretch of third collagen fiber family
    std::shared_ptr<std::vector<double>> collagenstretch3_;
    /// stretch of fourth collagen fiber family
    std::shared_ptr<std::vector<double>> collagenstretch4_;
    /// mass production rate of first fiber family
    std::shared_ptr<std::vector<double>> massprod1_;
    /// mass production rate of second fiber family
    std::shared_ptr<std::vector<double>> massprod2_;
    /// mass production rate of third fiber family
    std::shared_ptr<std::vector<double>> massprod3_;
    /// mass production rate of fourth fiber family
    std::shared_ptr<std::vector<double>> massprod4_;
    /// variable degradation of first fiber family
    std::shared_ptr<std::vector<double>> vardegrad1_;
    /// variable degradation of second fiber family
    std::shared_ptr<std::vector<double>> vardegrad2_;
    /// variable degradation of third fiber family
    std::shared_ptr<std::vector<double>> vardegrad3_;
    /// variable degradation of fourth fiber family
    std::shared_ptr<std::vector<double>> vardegrad4_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
