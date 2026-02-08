// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_python_surrogate_contactconstitutivelaw.hpp"

#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_par_bundle.hpp"

#ifdef FOUR_C_WITH_PYBIND11

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <mutex>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLawParams::
    PythonSurrogateConstitutiveLawParams(const Core::IO::InputParameterContainer& container)
    : CONTACT::CONSTITUTIVELAW::Parameter(container),
      python_filename_(container.get<std::filesystem::path>("Python_Filename"))
{
  if (!std::filesystem::exists(python_filename_))
    FOUR_C_THROW("File {} does not exist.", python_filename_.string());
}

/*----------------------------------------------------------------------*/
/*  PIMPL implementation                                                */
/*----------------------------------------------------------------------*/
class __attribute__((
    visibility("hidden"))) CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::Impl
{
 public:
  pybind11::object evaluate;
  pybind11::object evaluate_derivative;

  Impl(const CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLawParams& params)
  {
    // Initialize interpreter once per process.
    static std::once_flag py_init_flag;
    static std::unique_ptr<pybind11::scoped_interpreter> py_guard;

    std::call_once(
        py_init_flag, []() { py_guard = std::make_unique<pybind11::scoped_interpreter>(); });

    pybind11::module sys = pybind11::module::import("sys");
    sys.attr("path").attr("insert")(0, params.get_python_filepath().parent_path().string());

    pybind11::module model =
        pybind11::module::import(params.get_python_filepath().stem().string().c_str());

    evaluate = model.attr("evaluate");
    evaluate_derivative = model.attr("evaluate_derivative");
  }
};


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::PythonSurrogateConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLawParams params)
    : pimpl_(nullptr), params_(std::move(params))
{
  pimpl_ = std::make_unique<Impl>(params_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::~PythonSurrogateConstitutiveLaw() =
    default;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::evaluate(
    const double gap, CONTACT::Node* cnode)
{
  FOUR_C_ASSERT(gap + params_.get_offset() <= 0.0,
      "The Evaluate function can only operate on active nodes. With a current gap = {} and an "
      "ititial offset = {}, this node is inactive though.",
      gap, params_.get_offset());

  const double pressure = pimpl_->evaluate(gap, params_.get_offset()).cast<double>();

  return pressure;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::evaluate_derivative(
    const double gap, CONTACT::Node* cnode)
{
  FOUR_C_ASSERT(gap + params_.get_offset() <= 0.0,
      "The Evaluate function can only operate on active nodes. With a current gap = {} and an "
      "ititial offset = {}, this node is inactive though.",
      gap, params_.get_offset());


  const double derivative = pimpl_->evaluate_derivative(gap, params_.get_offset()).cast<double>();

  return derivative;
}

FOUR_C_NAMESPACE_CLOSE

#endif
