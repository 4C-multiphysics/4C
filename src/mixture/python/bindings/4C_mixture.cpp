// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config_submodule_registry.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_full_constrained_mixture_fiber.hpp"
#include "4C_mixture_remodelfiber-internal.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

FOUR_C_NAMESPACE_OPEN
namespace
{
  struct FiberMaterial
  {
    double k1;
    double k2;
    bool compression;
  };

  struct LinearGrowthPoissonTurnover
  {
    double growth_constant;
    double decay_time;
  };

  FourC::Core::Mat::PAR::Parameter::Data create_material(const FiberMaterial& fiber_material)
  {
    auto container = FourC::Core::IO::InputParameterContainer();
    container.add("K1", fiber_material.k1);
    container.add("K2", fiber_material.k2);
    container.add("COMPRESSION", fiber_material.compression);
    return {.parameters = container};
  }

  class FullConstrainedMixtureFiberBinder
  {
   public:
    FullConstrainedMixtureFiberBinder(const FiberMaterial& fiber_material,
        const LinearGrowthPoissonTurnover& turnover_law, double lambda_pre,
        FourC::Mixture::HistoryAdaptionStrategy adaptive_strategy, bool growth_enabled)
        : mat_data_(create_material(fiber_material)),
          fiber_(
              std::make_shared<FourC::Mixture::RemodelFiberMaterialExponential<double>>(&mat_data_),
              {turnover_law.growth_constant, turnover_law.decay_time, true}, lambda_pre,
              adaptive_strategy, growth_enabled)
    {
    }
    void reinitialize_history(double lambda_f, double time)
    {
      fiber_.reinitialize_history(lambda_f, time);
    }
    void recompute_state(double lambda_f, double time, double delta_time)
    {
      fiber_.recompute_state(lambda_f, time, delta_time);
    }
    void update() { fiber_.update(); }
    double get_cauchy_stress() { return fiber_.computed_sigma_; }
    double get_growth_scalar() { return fiber_.computed_growth_scalar_; }

    double get_d_cauchy_stress_d_lambda()
    {
      return 2 * fiber_.computed_dsigma_dlambda_f_sq_ * fiber_.current_state_.lambda_f;
    }

    double get_d_growth_scalar_d_lambda()
    {
      return 2 * fiber_.computed_dgrowth_scalar_dlambda_f_sq_ * fiber_.current_state_.lambda_f;
    }

    double get_adaptive_tolerance()
    {
      if (fiber_.adaptive_history_strategy_ !=
              FourC::Mixture::HistoryAdaptionStrategy::model_equation &&
          fiber_.adaptive_history_strategy_ !=
              FourC::Mixture::HistoryAdaptionStrategy::higher_order_integration)
        throw std::runtime_error(
            "This property is only defined if the history adaption strategy is window");
      return fiber_.adaptive_tolerance_;
    }
    void set_adaptive_tolerance(double tolerance) { fiber_.adaptive_tolerance_ = tolerance; }

    double get_window_size()
    {
      if (fiber_.adaptive_history_strategy_ != FourC::Mixture::HistoryAdaptionStrategy::window)
        throw std::runtime_error(
            "This property is only defined if the history adaption strategy is window");

      return fiber_.window_size;
    }
    void set_window_size(int window_size) { fiber_.window_size = window_size; }

    std::size_t get_history_size()
    {
      return std::accumulate(fiber_.history_.begin(), fiber_.history_.end(), 0,
          [](std::size_t sum, const FourC::Mixture::DepositionHistoryInterval<double>& item)
          { return item.timesteps.size() + sum; });
    }

    std::vector<double> get_history_times()
    {
      std::size_t total_items = std::accumulate(fiber_.history_.begin(), fiber_.history_.end(), 0,
          [](std::size_t sum, const FourC::Mixture::DepositionHistoryInterval<double>& item)
          { return item.timesteps.size() + sum; });

      std::vector<double> times(total_items);
      auto it = times.begin();
      for (const auto& interval : fiber_.history_)
      {
        for (const auto item : interval.timesteps)
        {
          *it = item.deposition_time;
          ++it;
        }
      }
      return times;
    }

   private:
    FourC::Mixture::PAR::RemodelFiberMaterialExponential<double> mat_data_;
    FourC::Mixture::FullConstrainedMixtureFiber<double> fiber_;
  };

  class ImplicitRemodelFiberBinder
  {
   public:
    ImplicitRemodelFiberBinder(const FiberMaterial& fiber_material,
        const LinearGrowthPoissonTurnover& turnover_law, double lambda_pre)
        : mat_data_(create_material(fiber_material)),
          fiber_(
              std::make_shared<FourC::Mixture::RemodelFiberMaterialExponential<double>>(&mat_data_),
              {turnover_law.growth_constant, turnover_law.decay_time, true}, lambda_pre)
    {
    }
    void recompute_state(double lambda_f, double dt)
    {
      current_lambda_f_ = lambda_f;
      fiber_.set_state(lambda_f, 1.0);
      fiber_.integrate_local_evolution_equations_implicit(dt);
    }
    void update() { fiber_.update(); }
    double get_cauchy_stress() { return fiber_.evaluate_current_fiber_cauchy_stress(); }
    double get_growth_scalar() { return fiber_.evaluate_current_growth_scalar(); }
    double get_d_cauchy_stress_d_lambda()
    {
      return fiber_.evaluate_d_current_cauchy_stress_d_lambda_f_sq() * 2 * current_lambda_f_;
    }
    double get_d_growth_scalar_d_lambda()
    {
      return fiber_.evaluate_d_current_growth_scalar_d_lambda_f_sq() * 2 * current_lambda_f_;
    }
    double get_current_lambda_r() { return fiber_.evaluate_current_lambda_r(); }
    // double get_current_lambda_e() { return fiber_.evaluate_current_lambda_e(); }

   private:
    double current_lambda_f_ = 0.0;
    FourC::Mixture::PAR::RemodelFiberMaterialExponential<double> mat_data_;
    FourC::Mixture::Implementation::RemodelFiberImplementation<2, double> fiber_;
  };

  class ExplicitRemodelFiberBinder
  {
   public:
    ExplicitRemodelFiberBinder(const FiberMaterial& fiber_material,
        const LinearGrowthPoissonTurnover& turnover_law, double lambda_pre)
        : mat_data_(create_material(fiber_material)),
          fiber_(
              std::make_shared<FourC::Mixture::RemodelFiberMaterialExponential<double>>(&mat_data_),
              {turnover_law.growth_constant, turnover_law.decay_time, true}, lambda_pre)
    {
    }
    void set_state(double lambda_f)
    {
      current_lambda_f_ = lambda_f;
      fiber_.set_state(lambda_f, 1.0);
    }
    void update(double dt)
    {
      fiber_.update();
      fiber_.integrate_local_evolution_equations_explicit(dt);
    }
    double get_cauchy_stress() { return fiber_.evaluate_current_fiber_cauchy_stress(); }
    double get_growth_scalar() { return fiber_.evaluate_current_growth_scalar(); }
    double get_kappa_dot()
    {
      return fiber_.evaluate_growth_evolution_equation_dt(current_lambda_f_,
                 fiber_.evaluate_current_lambda_r(), 1.0, fiber_.evaluate_current_growth_scalar()) /
             fiber_.evaluate_current_growth_scalar();
    }
    double get_d_cauchy_stress_d_lambda()
    {
      return fiber_.evaluate_d_current_cauchy_stress_d_lambda_f_sq() * 2 * current_lambda_f_;
    }
    double get_d_growth_scalar_d_lambda() { return 0; }
    double get_current_lambda_r() { return fiber_.evaluate_current_lambda_r(); }
    // double get_current_lambda_e() { return fiber_.evaluate_current_lambda_e(); }

   private:
    double current_lambda_f_ = 0.0;
    FourC::Mixture::PAR::RemodelFiberMaterialExponential<double> mat_data_;
    FourC::Mixture::Implementation::RemodelFiberImplementation<2, double> fiber_;
  };

  void init_submodule_mixture(py::module_& submodule)
  {
    py::class_<FiberMaterial>(submodule, "ExponentialFiberMaterial")
        .def(py::init<double, double, bool>(), py::arg("k1"), py::arg("k2"),
            py::arg("compression") = true);

    py::class_<LinearGrowthPoissonTurnover>(submodule, "LinearGrowthPoissonTurnover")
        .def(py::init<double, double>(), py::arg("growth_constant"), py::arg("decay_time"));

    py::enum_<FourC::Mixture::HistoryAdaptionStrategy>(submodule, "HistoryAdaptionStrategy")
        .value("none", FourC::Mixture::HistoryAdaptionStrategy::none)
        .value("window", FourC::Mixture::HistoryAdaptionStrategy::window)
        .value("model_equation", FourC::Mixture::HistoryAdaptionStrategy::model_equation)
        .value("higher_order", FourC::Mixture::HistoryAdaptionStrategy::higher_order_integration);

    py::class_<FullConstrainedMixtureFiberBinder>(submodule, "FullConstrainedMixtureFiber")
        .def(py::init<const FiberMaterial&, const LinearGrowthPoissonTurnover&, double,
                 FourC::Mixture::HistoryAdaptionStrategy&, bool>(),
            py::arg("fiber_material"), py::arg("turnover_law"), py::arg("lambda_pre"),
            py::arg("adaptive_strategy"), py::arg("growth_enabled") = true)
        .def("reinitialize_history", &FullConstrainedMixtureFiberBinder::reinitialize_history,
            py::arg("lambda_f"), py::arg("total_time"))
        .def("recompute_state", &FullConstrainedMixtureFiberBinder::recompute_state,
            py::arg("lambda_f"), py::arg("total_time"), py::arg("delta_time"))
        .def("update", &FullConstrainedMixtureFiberBinder::update)
        .def("get_history_times", &FullConstrainedMixtureFiberBinder::get_history_times)
        .def_property("history_size", &FullConstrainedMixtureFiberBinder::get_history_size, nullptr)
        .def_property(
            "cauchy_stress", &FullConstrainedMixtureFiberBinder::get_cauchy_stress, nullptr)
        .def_property(
            "growth_scalar", &FullConstrainedMixtureFiberBinder::get_growth_scalar, nullptr)
        .def_property("dcauchy_stress_dlambda",
            &FullConstrainedMixtureFiberBinder::get_d_cauchy_stress_d_lambda, nullptr)
        .def_property("dgrowth_scalar_dlambda",
            &FullConstrainedMixtureFiberBinder::get_d_growth_scalar_d_lambda, nullptr)
        .def_property("adaptive_tolerance",
            &FullConstrainedMixtureFiberBinder::get_adaptive_tolerance,
            &FullConstrainedMixtureFiberBinder::set_adaptive_tolerance)
        .def_property("window_size", &FullConstrainedMixtureFiberBinder::get_window_size,
            &FullConstrainedMixtureFiberBinder::set_window_size);

    py::class_<ImplicitRemodelFiberBinder>(submodule, "ImplicitRemodelFiber")
        .def(py::init<const FiberMaterial&, const LinearGrowthPoissonTurnover&, double>(),
            py::arg("fiber_material"), py::arg("turnover_law"), py::arg("lambda_pre"))
        .def("recompute_state", &ImplicitRemodelFiberBinder::recompute_state, py::arg("lambda_f"),
            py::arg("delta_time"))
        .def("update", &ImplicitRemodelFiberBinder::update)
        .def_property("cauchy_stress", &ImplicitRemodelFiberBinder::get_cauchy_stress, nullptr)
        .def_property("growth_scalar", &ImplicitRemodelFiberBinder::get_growth_scalar, nullptr)
        .def_property("dcauchy_stress_dlambda",
            &ImplicitRemodelFiberBinder::get_d_cauchy_stress_d_lambda, nullptr)
        .def_property("dgrowth_scalar_dlambda",
            &ImplicitRemodelFiberBinder::get_d_growth_scalar_d_lambda, nullptr)
        .def_property("lambda_r", &ImplicitRemodelFiberBinder::get_current_lambda_r, nullptr);
    //.def_property("lambda_e", &ImplicitRemodelFiberBinder::get_current_lambda_e, nullptr);

    py::class_<ExplicitRemodelFiberBinder>(submodule, "ExplicitRemodelFiber")
        .def(py::init<const FiberMaterial&, const LinearGrowthPoissonTurnover&, double>(),
            py::arg("fiber_material"), py::arg("turnover_law"), py::arg("lambda_pre"))
        .def("set_state", &ExplicitRemodelFiberBinder::set_state, py::arg("lambda_f"))
        .def("update", &ExplicitRemodelFiberBinder::update, py::arg("delta_time"))
        .def_property("cauchy_stress", &ExplicitRemodelFiberBinder::get_cauchy_stress, nullptr)
        .def_property("kappa_dot", &ExplicitRemodelFiberBinder::get_kappa_dot, nullptr)
        .def_property("growth_scalar", &ExplicitRemodelFiberBinder::get_growth_scalar, nullptr)
        .def_property("dcauchy_stress_dlambda",
            &ExplicitRemodelFiberBinder::get_d_cauchy_stress_d_lambda, nullptr)
        .def_property("dgrowth_scalar_dlambda",
            &ExplicitRemodelFiberBinder::get_d_growth_scalar_d_lambda, nullptr)
        .def_property("lambda_r", &ExplicitRemodelFiberBinder::get_current_lambda_r, nullptr);
    //.def_property("lambda_e", &ExplicitRemodelFiberBinder::get_current_lambda_e, nullptr);
  }

  FOUR_C_PYBIND_REGISTER_SUBMODULE(
      mixture, "Python bindings to the mixture implementations in 4C.", &init_submodule_mixture);
}  // namespace
FOUR_C_NAMESPACE_CLOSE