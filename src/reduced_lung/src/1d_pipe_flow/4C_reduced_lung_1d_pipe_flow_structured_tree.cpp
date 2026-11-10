// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_1d_pipe_flow_structured_tree.hpp"

#include "4C_global_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <vector>

#ifdef FOUR_C_WITH_FFTW
#include <fftw3.h>

#include <numbers>
#endif

FOUR_C_NAMESPACE_OPEN
namespace ReducedLung1DPipe
{
  namespace TerminalUnit
  {
    namespace
    {
      // --------------- Bessel J0 and J1 for complex arguments ---------------
      // Series expansion: J_n(z) = sum_{m=0}^{inf} (-1)^m (z/2)^{2m+n} / (m! (m+n)!)

      // Define J0(z)
      std::complex<double> bessel_j0(std::complex<double> z)
      {
        std::complex<double> term(1.0, 0.0);
        std::complex<double> sum = term;
        std::complex<double> z2 = -z * z / 4.0;
        constexpr double tol = 1e-8;
        for (int m = 1; m <= 200; ++m)
        {
          term *= z2 / double(m * m);
          sum += term;
          if (std::abs(term) < tol) break;
        }
        return sum;
      }

      // Define J1(z)
      std::complex<double> bessel_j1(std::complex<double> z)
      {
        std::complex<double> term = z / 2.0;
        std::complex<double> sum = term;
        std::complex<double> z2 = -z * z / 4.0;
        constexpr double tol = 1e-8;
        for (int m = 1; m <= 200; ++m)
        {
          term *= z2 / double(m * (m + 1));
          sum += term;
          if (std::abs(term) < tol) break;
        }
        return sum;
      }

      // --------------- Womersley impedance of a structured tree vessel segment ---------------

      // F_J = 2*J1(w) / (w * J0(w)) , w = alpha*i^1.5 , Womersley number a = r*sqrt(omega * rho
      // /mu) i^1.5 = (-1 + i)/sqrt(2). Womersley_fj is evaluated for every node at every frequency.
      const std::complex<double> i_to_1_5 = std::pow(std::complex<double>(0.0, 1.0), 1.5);

      std::complex<double> womersley_fj(double alpha)
      {
        if (alpha < 1e-4) return std::complex<double>(1.0, 0.0);  // F_J -> 1 for omega -> 0

        const std::complex<double> w = alpha * i_to_1_5;
        const std::complex<double> J0 = bessel_j0(w);
        const std::complex<double> J1 = bessel_j1(w);
        return 2.0 * J1 / (w * J0);
      }

      // ------------------------------------------------------------------------
      // Womersley impedance of a structured tree vessel segment
      // ------------------------------------------------------------------------
      //
      // The tree is self-similar: each bifurcation splits a vessel of radius r into
      // two daughter vessels r_large and r_small. Both daughters
      // are constant multiples of the parent radius, so the reachable radii are
      // products of powers of two factors -> tree of distinct radii collapses
      // from ~2^depth paths to a directed acyclic graph (DAG) of only ~depth^2 nodes.
      // The DAG is built once and the impedance is evaluated per frequency in reverse topological
      // order on a flat array.

      struct TreeDagNode
      {
        double radius;
        bool terminal;         /// flag if terminal criterion is true
        int child_large;       /// index of the larger daughter vessel, -1 for terminal nodes
        int child_small;       /// index of the smaller daughter vessel, -1 for terminal nodes
        double length_L;       /// vessel length L = k_l * r^k_exp
        double wave_speed_c0;  /// foot-of-wave speed sqrt(2*Eh/r / (3*rho))
      };

      std::vector<TreeDagNode> build_tree_dag(double root_radius, double xi, double gamma,
          double r_min, double rho, double k_l, double length_exponent,
          const std::function<double(double)>& evaluate_eh_over_r)
      {
        const double radius_ratio = std::sqrt(gamma);

        std::vector<TreeDagNode> nodes;
        std::map<double, int> radius_to_index;
        // vector with radii where daughters need to be created
        std::vector<double> radii_to_expand;

        auto get_or_create = [&](double r) -> int
        {
          const auto it = radius_to_index.find(r);
          if (it != radius_to_index.end()) return it->second;

          const int index = static_cast<int>(nodes.size());
          radius_to_index.emplace(r, index);
          // If terminal, peripheral resistance R_peri is applied at the end
          const bool terminal = (r <= r_min);

          // stiffness function Eh/r
          const double Eh_over_r_val = evaluate_eh_over_r(r);
          nodes.push_back(TreeDagNode{.radius = r,
              .terminal = terminal,
              .child_large = -1,
              .child_small = -1,
              .length_L = k_l * std::pow(r, length_exponent),
              .wave_speed_c0 = std::sqrt(2.0 * Eh_over_r_val / (3.0 * rho))});
          radii_to_expand.push_back(r);
          return index;
        };

        get_or_create(root_radius);

        int radius_index = 0;
        while (radius_index < static_cast<int>(radii_to_expand.size()))
        {
          const double r = radii_to_expand[radius_index++];
          const int index = radius_to_index[r];

          if (nodes[index].terminal) continue;

          // Recursive creation of tree vessels of structured tree
          const double r_large = r / std::pow(std::pow(radius_ratio, xi) + 1.0, 1.0 / xi);
          const double r_small = radius_ratio * r_large;
          const int child_large = get_or_create(r_large);
          const int child_small = get_or_create(r_small);
          nodes[index].child_large = child_large;
          nodes[index].child_small = child_small;
        }

        return nodes;
      }

      // The impedance converges to the (real) characteristic impedance of the root vessel for
      // large omega. 256 harmonics cover the relevant physiological range.
      constexpr int max_frequencies = 256;

      // Compute Z(omega_k) for each harmonic
      std::vector<std::complex<double>> compute_root_impedance_spectrum(
          const std::vector<TreeDagNode>& dag, double R_peri, double rho, double mu, int n_cycle,
          double dt)
      {
        const double T = n_cycle * dt;
        const double domega = 2.0 * std::numbers::pi / T;
        const int n_half = n_cycle / 2;
        const int n_compute = std::min(n_half, max_frequencies);
        const int n_nodes = static_cast<int>(dag.size());

        std::vector<std::complex<double>> Z_full(n_cycle);
        // Z_node starts as the far-end impedance of each vessel. For terminal vessels the far
        // end (x = L) is closed by the peripheral resistance R_peri; for non-terminal vessels
        // the far end is a bifurcation and Z_node is overwritten with the input impedance seen
        // from the vessel entrance as the reverse-index loop below walks up the tree.
        std::vector<std::complex<double>> Z_node(n_nodes, std::complex<double>(R_peri, 0.0));

        // Frequency-independent node data hoisted out of the frequency loop.
        struct NodeData
        {
          double alpha_factor;  /// radius*sqrt(rho/mu), so that alpha = alpha_factor*sqrt(omega)
          double z0_norm;       /// rho*c0/(pi*r^2), the omega-independent part of Z0
          double k_gamma;       /// L/c0, the omega-independent part of gamma_L
          double R_poiseuille;  /// 8*mu*L/(pi*r^4), only used in the alpha -> 0 limit
        };
        std::vector<NodeData> node_data(n_nodes);
        for (int i = n_nodes - 1; i >= 0; --i)
        {
          const TreeDagNode& node = dag[i];
          node_data[i] = {node.radius * std::sqrt(rho / mu),
              rho * node.wave_speed_c0 / (std::numbers::pi * node.radius * node.radius),
              node.length_L / node.wave_speed_c0,
              8.0 * mu * node.length_L /
                  (std::numbers::pi * node.radius * node.radius * node.radius * node.radius)};
        }

        for (int k = 0; k <= n_compute; ++k)
        {
          const double omega = k * domega;
          const double sqrt_omega = std::sqrt(omega);

          // Lower indices are assigned to parents than to children, so iterating the nodes in
          // reverse index order resolves children before parents.
          for (int i = n_nodes - 1; i >= 0; --i)
          {
            const TreeDagNode& node = dag[i];
            const NodeData& data = node_data[i];

            std::complex<double> Z_bif;
            if (node.terminal)
            {
              // Terminal vessels are full-length vessels whose far end is closed by the
              // peripheral resistance.
              Z_bif = std::complex<double>(R_peri, 0.0);
            }
            else
            {
              const std::complex<double> Z_L = Z_node[node.child_large];
              const std::complex<double> Z_R = Z_node[node.child_small];
              const std::complex<double> Z_sum = Z_L + Z_R;
              if (std::abs(Z_sum) < 1e-8)
                Z_bif = 0.0;
              else
                Z_bif = (Z_L * Z_R) / Z_sum;
            }

            // F_J(alpha) = 2*J1(alpha*i^1.5) / (alpha*i^1.5*J0(alpha*i^1.5)) evaluated exactly
            // from the Bessel series.
            const double alpha = data.alpha_factor * sqrt_omega;
            const std::complex<double> F_J = womersley_fj(alpha);
            const std::complex<double> one_minus_FJ = 1.0 - F_J;

            if (std::abs(one_minus_FJ) < 1e-8)
            {
              Z_node[i] = std::complex<double>(data.R_poiseuille, 0.0) + Z_bif;
              continue;
            }

            const std::complex<double> sqrt_inv = 1.0 / std::sqrt(one_minus_FJ);
            const std::complex<double> Z0 = data.z0_norm * sqrt_inv;
            const std::complex<double> gamma_L =
                std::complex<double>(0.0, omega * data.k_gamma) * sqrt_inv;

            // Exponential form of the transmission line equation:
            // Z_in = Z0 * (1 - R * exp(-2*gamma_L)) / (1 + R * exp(-2*gamma_L))
            // with R = (Z0 - Z_bif) / (Z0 + Z_bif). For 2*Re(gamma_L) > 37 the factor
            // exp(-2*gamma_L) is < 1e-16 and Z_in collapses to Z0
            if (2.0 * gamma_L.real() > 37.0)
            {
              Z_node[i] = Z0;
              continue;
            }

            const std::complex<double> R = (Z0 - Z_bif) / (Z0 + Z_bif);
            const std::complex<double> exp_m2gL = std::exp(-2.0 * gamma_L);
            Z_node[i] = Z0 * (1.0 - R * exp_m2gL) / (1.0 + R * exp_m2gL);
          }

          Z_full[k] = Z_node[0];
        }

        // Harmonics above max_frequencies are set to the boundary value Z[n_compute].
        for (int k = n_compute + 1; k <= n_half; ++k) Z_full[k] = Z_full[n_compute];

        // For even n_cycle the Nyquist bin must be real for the spectrum to be Hermitian
        if (n_cycle % 2 == 0) Z_full[n_half] = std::complex<double>(Z_full[n_half].real(), 0.0);

        for (int k = n_half + 1; k < n_cycle; ++k)
        {
          Z_full[k] = std::conj(Z_full[n_cycle - k]);
        }

        return Z_full;
      }

#ifdef FOUR_C_WITH_FFTW
      std::vector<double> impulse_response_via_fftw(
          const std::vector<std::complex<double>>& Z, int n_cycle)
      {
        // Z has Hermitian symmetry, so the inverse transform z[n] = 1/n * sum_k Z[k] *
        // exp(2*pi*i*k*n/n) is real and FFTW's half-complex c2r transform can be used.
        const int n_half = n_cycle / 2;
        std::vector<std::complex<double>> input(Z.begin(), Z.begin() + n_half + 1);
        std::vector<double> z(n_cycle, 0.0);
        fftw_plan plan = fftw_plan_dft_c2r_1d(
            n_cycle, reinterpret_cast<fftw_complex*>(input.data()), z.data(), FFTW_ESTIMATE);
        if (plan == nullptr)
          FOUR_C_THROW("Structured tree: FFTW plan creation failed (c2r of length {})", n_cycle);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        for (double& zk : z) zk /= double(n_cycle);
        return z;
      }
#endif

      std::vector<double> compute_impulse_response(
          const std::vector<std::complex<double>>& Z, int n_cycle)
      {
#ifdef FOUR_C_WITH_FFTW
        return impulse_response_via_fftw(Z, n_cycle);
#else
        FOUR_C_THROW(
            "Structured tree: FFTW is required for the structured-tree boundary condition. "
            "Please enable FFTW during the configure process.");
#endif
      }

    }  // anonymous namespace

    std::vector<std::complex<double>> compute_structured_tree_impedance(
        const StructuredTreeTerminalUnit& st_model, double density_rho, double viscosity_mu,
        const std::function<double(double)>& evaluate_eh_over_r)
    {
      std::vector<TreeDagNode> dag = build_tree_dag(st_model.root_radius, st_model.exponent_xi,
          st_model.asymmetry_ratio_gamma, st_model.termination_radius_r_min, density_rho,
          st_model.length_coefficient, st_model.length_exponent, evaluate_eh_over_r);
      return compute_root_impedance_spectrum(dag, st_model.peripheral_resistance_R_peri,
          density_rho, viscosity_mu, st_model.n_cycle, st_model.dt);
    }

    StructuredTreeTerminalUnit create_structured_tree_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::StructuredTreeModel& st_model,
        int global_id, double root_radius, double density_rho, double viscosity_mu, double dt,
        double period)
    {
      StructuredTreeTerminalUnit model;

      model.root_radius = root_radius;
      model.exponent_xi = st_model.exponent_xi.at(global_id);
      model.asymmetry_ratio_gamma = st_model.asymmetry_ratio_gamma.at(global_id);
      model.termination_radius_r_min = st_model.termination_radius_r_min.at(global_id);
      model.peripheral_resistance_R_peri = st_model.peripheral_resistance_R_peri.at(global_id);
      const int stiffness_function_id = st_model.stiffness_function_id.at(global_id);
      const auto* eh_over_r_function =
          &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfAnything>(
              stiffness_function_id);
      std::function<double(double)> evaluate_eh_over_r = [eh_over_r_function](double r)
      { return eh_over_r_function->evaluate({{"r", r}}, {}, 0); };
      model.length_coefficient = st_model.length_coefficient.at(global_id);
      model.length_exponent = st_model.length_exponent.at(global_id);

      model.n_cycle = static_cast<int>(std::round(period / dt));
      model.dt = dt;
      if (model.n_cycle < 2)
        FOUR_C_THROW("Structured tree: period/dt must be >= 2 (got {})", model.n_cycle);

      model.impedance_spectrum =
          compute_structured_tree_impedance(model, density_rho, viscosity_mu, evaluate_eh_over_r);
      model.impulse_response = compute_impulse_response(model.impedance_spectrum, model.n_cycle);

      // Shorten impulse response to its significant part.
      const double z0_magnitude = std::abs(model.impulse_response.front());
      constexpr double truncation_tolerance = 1e-8;
      int cutoff_index = 0;
      for (int k = 0; k < model.n_cycle; ++k)
        if (std::abs(model.impulse_response[k]) >= truncation_tolerance * z0_magnitude)
          cutoff_index = k;
      model.impulse_response.resize(cutoff_index + 1);

      model.flow_history.resize(model.n_cycle, 0.0);
      model.flow_history_write_index = 0;
      model.flow_history_count = 0;
      model.convolved_pressure = 0.0;

      return model;
    }

    double evaluate_structured_tree_residual(double area_A, double flow_Q, double reference_area_A0,
        double beta, double Pext, const StructuredTreeTerminalUnit& st_model)
    {
      // 1D pressure
      double p_1D = beta * (std::sqrt(area_A) - std::sqrt(reference_area_A0)) + Pext;
      // 0D pressure depending on flow history
      double p_ST = st_model.impulse_response[0] * flow_Q + st_model.convolved_pressure;

      return p_1D - p_ST;
    }

    double evaluate_structured_tree_jacobian(
        double area_A, double dQdA, double beta, const StructuredTreeTerminalUnit& st_model)
    {
      double dp1D_dA = 0.5 * beta / std::sqrt(area_A);
      double dpST_dA = st_model.impulse_response[0] * dQdA;

      return dp1D_dA - dpST_dA;
    }

    void update_structured_tree_data(StructuredTreeTerminalUnit& st_model, double flow_Q)
    {
      st_model.flow_history[st_model.flow_history_write_index] = flow_Q;
      st_model.flow_history_write_index =
          (st_model.flow_history_write_index + 1) % st_model.n_cycle;
      if (st_model.flow_history_count < st_model.n_cycle) st_model.flow_history_count++;

      // Update flow history for the upcoming time step.
      st_model.convolved_pressure = 0.0;
      const int n_history = std::min({st_model.flow_history_count, st_model.n_cycle - 1,
          static_cast<int>(st_model.impulse_response.size()) - 1});

      int k = 1;
      int flow_history_index = st_model.flow_history_write_index - 1;
      const int n_first = std::min(n_history, st_model.flow_history_write_index);
      for (; k <= n_first; ++k, --flow_history_index)
        st_model.convolved_pressure +=
            st_model.impulse_response[k] * st_model.flow_history[flow_history_index];
      flow_history_index = st_model.n_cycle - 1;
      for (; k <= n_history; ++k, --flow_history_index)
        st_model.convolved_pressure +=
            st_model.impulse_response[k] * st_model.flow_history[flow_history_index];
    }
  }  // namespace TerminalUnit
}  // namespace ReducedLung1DPipe
FOUR_C_NAMESPACE_CLOSE
