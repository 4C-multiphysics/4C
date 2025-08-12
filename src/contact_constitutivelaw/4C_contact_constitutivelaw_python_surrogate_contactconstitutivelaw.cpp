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

namespace py = pybind11;

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
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::PythonSurrogateConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLawParams params)
    : params_(std::move(params))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::evaluate(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes. ");
  }

  py::scoped_interpreter guard{};

  py::module sys = py::module::import("sys");
  sys.attr("path").attr("insert")(0, params_.get_python_filepath().parent_path().string());
  py::module model = py::module::import(params_.get_python_filepath().stem().c_str());
  py::object evaluate = model.attr("evaluate");

  const double pressure = evaluate(gap, params_.get_offset()).cast<double>();

  return pressure;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLaw::evaluate_derivative(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes.");
  }

  py::scoped_interpreter guard{};

  py::module sys = py::module::import("sys");
  sys.attr("path").attr("insert")(0, params_.get_python_filepath().parent_path().string());
  py::module model = py::module::import(params_.get_python_filepath().stem().c_str());
  py::object evaluate_derivative = model.attr("evaluate_derivative");

  const double derivative = evaluate_derivative(gap, params_.get_offset()).cast<double>();

  return derivative;
}

FOUR_C_NAMESPACE_CLOSE

#endif
