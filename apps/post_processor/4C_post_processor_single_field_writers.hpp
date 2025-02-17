// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POST_PROCESSOR_SINGLE_FIELD_WRITERS_HPP
#define FOUR_C_POST_PROCESSOR_SINGLE_FIELD_WRITERS_HPP

#include "4C_config.hpp"

#include "4C_post_filter_base.hpp"
#include "4C_post_writer_base.hpp"


FOUR_C_NAMESPACE_OPEN

class PostWriterBase;

/*!
 \brief Writer for structural problems
 */
class StructureFilter : public PostFilterBase
{
 public:
  StructureFilter(PostField* field, std::string name, std::string stresstype = "none",
      std::string straintype = "none")
      : PostFilterBase(field, name), stresstype_(stresstype), straintype_(straintype)
  {
  }

 protected:
  void write_all_results(PostField* field) override;

  void write_all_results_one_time_step(PostResult& result, bool firststep, bool laststep) override;

  /*!
  \brief postprocess gauss point stresses and write results

  */
  void post_stress(const std::string groupname, const std::string stresstype);
  void write_stress(const std::string groupname, PostResult& result, const ResultType stresskind);
  void write_eigen_stress(
      const std::string groupname, PostResult& result, const ResultType stresskind);

  std::string stresstype_;
  std::string straintype_;
};

/*!
 \brief Writer for mortar interface problems

 Each mortar interface is written as its own discretization. The MortarFilter will process only one
 of these interfaces, i.e. there will be as many MortarFilers as there are mortar interfaces.
 */
class MortarFilter : public PostFilterBase
{
 public:
  /*!
  \brief Constructor

  @param field Field to be processed
  @param name ??
  */
  MortarFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  /*!
  \brief Write all results of a field

  So far, we don't need special control over quantities to be filtered. We just filter every result
  field.

  @param[in] field Field to be processed

  \sa WriteDofResults(), WriteNodeResults(), WriteElementResults()
  */
  void write_all_results(PostField* field) final;
};

/*!
 \brief Writer for fluid problems
 */
class FluidFilter : public PostFilterBase
{
 public:
  FluidFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  void write_all_results(PostField* field) override;
};

/*!
 \brief Writer for xfluid problems
 */
class XFluidFilter : public PostFilterBase
{
 public:
  XFluidFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  void write_all_results(PostField* field) override;
};

/*!
 \brief Writer for ale problems
 */
class AleFilter : public PostFilterBase
{
 public:
  AleFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  void write_all_results(PostField* field) override;
};


/*!
 \brief Writer for interface fields in XFEM
 */
class InterfaceFilter : public PostFilterBase
{
 public:
  InterfaceFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  void write_all_results(PostField* field) override;
};


/*!
 \brief Writer for lubrication problems

*/
class LubricationFilter : public PostFilterBase
{
 public:
  /// constructor
  LubricationFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  void write_all_results(PostField* field) override;
};


/*!
 \brief Writer for (in)stationary heat conduction

*/
class ThermoFilter : public PostFilterBase
{
 public:
  /// constructor
  ThermoFilter(PostField* field, std::string name, std::string heatfluxtype = "none",
      std::string tempgradtype = "none")
      : PostFilterBase(field, name), heatfluxtype_(heatfluxtype), tempgradtype_(tempgradtype)
  {
  }

 protected:
  void write_all_results(PostField* field) override;

  /*!
  \brief postprocess gauss point heatfluxes and write results

  */
  void post_heatflux(const std::string groupname, const std::string heatfluxtype);
  void write_heatflux(const std::string groupname, PostResult& result, const ResultType kind);

  std::string heatfluxtype_;  ///< type of heat flux output
  std::string tempgradtype_;  ///< type of spatial temperature gradient output
};


/// Writer for undefined problem types
/*
  Just write all the vectors we have.
 */
class AnyFilter : public PostFilterBase
{
 public:
  AnyFilter(PostField* field, std::string name) : PostFilterBase(field, name) {}

 protected:
  void write_all_results(PostField* field) override;
};

FOUR_C_NAMESPACE_CLOSE

#endif
