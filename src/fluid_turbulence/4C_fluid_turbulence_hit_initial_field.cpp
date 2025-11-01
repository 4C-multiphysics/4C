// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_hit_initial_field.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_timint_hdg.hpp"
#include "4C_global_data.hpp"

#include <cmath>
#include <complex>

#ifdef FOUR_C_WITH_FFTW
#include <fftw3.h>
#endif
FOUR_C_NAMESPACE_OPEN

namespace FLD
{
  /*--------------------------------------------------------------*
   | constructor                                  rasthofer 04/13 |
   *--------------------------------------------------------------*/
  HomoIsoTurbInitialField::HomoIsoTurbInitialField(
      FluidImplicitTimeInt& timeint, const Inpar::FLUID::InitialField initfield)
      : discret_(timeint.discret_),
        velnp_(timeint.velnp_),
        veln_(timeint.veln_),
        velnm_(timeint.velnm_),
        type_(initfield)
  {
    // determine number of modes
    // number of modes equal to number of elements in one spatial direction
    // this does not yield the correct value
    // nummodes_ = (int) pow((double) discret_->NumGlobalElements(),1.0/3.0);
    switch (discret_->num_global_elements())
    {
      case 512:
      {
        nummodes_ = 8;
        break;
      }
      case 1728:
      {
        nummodes_ = 12;
        break;
      }
      case 4096:
      {
        nummodes_ = 16;
        break;
      }
      case 13824:
      {
        nummodes_ = 24;
        break;
      }
      case 32768:
      {
        nummodes_ = 32;
        break;
      }
      case 110592:
      {
        nummodes_ = 48;
        break;
      }
      case 262144:
      {
        nummodes_ = 64;
        break;
      }
      default:
      {
        FOUR_C_THROW("Set problem size! {}", discret_->num_global_elements());
        break;
      }
    }

    //-------------------------------------------------
    // create set of node coordinates
    //-------------------------------------------------

    // the criterion allows differences in coordinates by 1e-9
    std::set<double, LineSortCriterion> coords;
    // loop all nodes and store x1-coordinate
    for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
    {
      Core::Nodes::Node* node = discret_->l_row_node(inode);
      if ((node->x()[1] < 2e-9 && node->x()[1] > -2e-9) and
          (node->x()[2] < 2e-9 && node->x()[2] > -2e-9))
        coords.insert(node->x()[0]);
    }
    // communicate coordinates to all procs via round Robin loop
    {
      int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());
      int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

      std::vector<char> sblock;
      std::vector<char> rblock;

      // create an exporter for point to point communication
      Core::Communication::Exporter exporter(discret_->get_comm());

      // communicate coordinates
      for (int np = 0; np < numprocs; ++np)
      {
        Core::Communication::PackBuffer data;

        for (std::set<double, LineSortCriterion>::iterator x1line = coords.begin();
            x1line != coords.end(); ++x1line)
        {
          add_to_pack(data, *x1line);
        }
        std::swap(sblock, data());

        MPI_Request request;
        int tag = myrank;

        int frompid = myrank;
        int topid = (myrank + 1) % numprocs;

        int length = sblock.size();

        exporter.i_send(frompid, topid, sblock.data(), sblock.size(), tag, request);

        rblock.clear();

        // receive from predecessor
        frompid = (myrank + numprocs - 1) % numprocs;
        exporter.receive_any(frompid, tag, rblock, length);

        if (tag != (myrank + numprocs - 1) % numprocs)
        {
          FOUR_C_THROW("received wrong message (ReceiveAny)");
        }

        exporter.wait(request);

        {
          // for safety
          Core::Communication::barrier(exporter.get_comm());
        }

        // unpack received block into set of all coordinates
        {
          std::vector<double> coordsvec;

          coordsvec.clear();

          Core::Communication::UnpackBuffer buffer(rblock);
          while (!buffer.at_end())
          {
            double onecoord;
            extract_from_pack(buffer, onecoord);
            coords.insert(onecoord);
          }
        }
      }
    }
    // push coordinates in vectors
    {
      coordinates_ = std::make_shared<std::vector<double>>();

      for (std::set<double, LineSortCriterion>::iterator coord1 = coords.begin();
          coord1 != coords.end(); ++coord1)
      {
        coordinates_->push_back(*coord1);
      }
    }

    //-------------------------------------------------
    // set-up experimental data
    //-------------------------------------------------
    // non-dimensionalize and store experimental data

    if (type_ == Inpar::FLUID::initfield_hit_comte_bellot_corrsin)
      // of Comte-Bellot-Corrsin experiment
      prepare_exparimental_data();

    return;
  }

  // also consider routine for HDG further down
  /*--------------------------------------------------------------*
   | calculate initial field using fft            rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbInitialField::calculate_initial_field()
  {
#ifdef FOUR_C_WITH_FFTW

    // set and initialize working arrays
    Teuchos::Array<std::complex<double>> u1_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u2_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u3_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

    Teuchos::Array<double> u1(nummodes_ * nummodes_ * nummodes_);
    Teuchos::Array<double> u2(nummodes_ * nummodes_ * nummodes_);
    Teuchos::Array<double> u3(nummodes_ * nummodes_ * nummodes_);

    //-------------------------------------------------
    // construction of initial field in spectral space
    //-------------------------------------------------

    // as given in the respective literature, the Fourier coefficients are
    // evaluated in the following intervals
    // k_1: [-nummodes_/2,(nummodes_/2-1)]
    // k_2: [-nummodes_/2,(nummodes_/2-1)]
    // k_3: [-nummodes_/2,0]
    for (int k_1 = (-nummodes_ / 2); k_1 <= (nummodes_ / 2 - 1); k_1++)
    {
      for (int k_2 = (-nummodes_ / 2); k_2 <= (nummodes_ / 2 - 1); k_2++)
      {
        for (int k_3 = (-nummodes_ / 2); k_3 <= 0; k_3++)
        {
          // get position in u1_hat
          const int pos =
              (k_3 + nummodes_ / 2) +
              (nummodes_ / 2 + 1) * ((k_2 + nummodes_ / 2) + nummodes_ * (k_1 + nummodes_ / 2));

          if (k_1 == (-nummodes_ / 2) or k_2 == (-nummodes_ / 2) or k_3 == (-nummodes_ / 2))
          {
            // odd-ball wave numbers are set to zero to ensure that solution is a real function
            ((u1_hat)[pos]).real(0.0);
            // this is important to have here
            ((u1_hat)[pos]).imag(0.0);
            // remaining analogously
            ((u2_hat)[pos]).real(0.0);
            ((u2_hat)[pos]).imag(0.0);
            ((u3_hat)[pos]).real(0.0);
            ((u3_hat)[pos]).imag(0.0);
          }
          else if (k_1 == 0 and k_2 == 0 and k_3 == 0)
          {
            // likewise set to zero since there will not be any conjugate complex
            ((u1_hat)[pos]).real(0.0);
            // this is important to have here
            ((u1_hat)[pos]).imag(0.0);
            // remaining analogously
            ((u2_hat)[pos]).real(0.0);
            ((u2_hat)[pos]).imag(0.0);
            ((u3_hat)[pos]).real(0.0);
            ((u3_hat)[pos]).imag(0.0);
          }
          else
          {
            bool calculate = true;
            // check if conjugate complex has already been set
            int pos_conj = -999;
            if (k_3 == 0)
            {
              pos_conj = (-k_3 + nummodes_ / 2) +
                         (nummodes_ / 2 + 1) *
                             ((-k_2 + nummodes_ / 2) + nummodes_ * (-k_1 + nummodes_ / 2));

              if (pos_conj < pos) calculate = false;
            }

            if (calculate)
            {
              const double k_12 = sqrt(k_1 * k_1 + k_2 * k_2);
              const double k = sqrt(k_1 * k_1 + k_2 * k_2 + k_3 * k_3);

              double random_theta1 = 0.0;
              double random_theta2 = 0.0;
              double random_phi = 0.0;

              // random numbers are created by one processor and
              // then send to the other processors
              // this ensures that all processors construct the same
              // initial field, which is important to get a matching
              // velocity field in physical space
              if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
              {
                Core::Utils::Random* random = Global::Problem::instance()->random();
                // set range [0;1] (default: [-1;1])
                //              random->SetRandRange(0.0,1.0);
                //              random_theta1 = random->Uni();
                //              random_theta2 = random->Uni();
                //              random_phi = random->Uni();
                random_theta1 = 0.5 * random->uni() + 0.5;
                random_theta2 = 0.5 * random->uni() + 0.5;
                random_phi = 0.5 * random->uni() + 0.5;
              }
              Core::Communication::broadcast(&random_theta1, 1, 0, discret_->get_comm());
              Core::Communication::broadcast(&random_theta2, 1, 0, discret_->get_comm());
              Core::Communication::broadcast(&random_phi, 1, 0, discret_->get_comm());

              // estimate energy at wave number from energy spectrum
              double energy = 0.0;
              if (type_ == Inpar::FLUID::initfield_hit_comte_bellot_corrsin)
                energy = interpolate_energy_from_spectrum(k);
              else
                energy = calculate_energy_from_spectrum(k);

              if (energy < 0.0)
              {
                std::cout << "k " << k << std::endl;
                std::cout << "k1  " << k_1 << std::endl;
                std::cout << "k2  " << k_2 << std::endl;
                std::cout << "k3  " << k_3 << std::endl;
                FOUR_C_THROW("Negative energy!");
              }

              // remark on the literature:
              // Collis 2002: sqrt(energy/(2*PI*k))
              const double fac = sqrt(energy / (2 * std::numbers::pi * k * k));
              // Rogallo 1981: sqrt(energy/(4*PI*k*k))
              // the missing factor 1/2 of Collis version compared to Rogallo version
              // is related to the definition of E from u_i
              // here, we have E = 1/2 * u_i *u_i (see statistics manager)

              // real part, imaginary part
              std::complex<double> alpha(fac * cos(2 * std::numbers::pi * random_theta1) *
                                             cos(2 * std::numbers::pi * random_phi),
                  fac * sin(2 * std::numbers::pi * random_theta1) *
                      cos(2 * std::numbers::pi * random_phi));
              std::complex<double> beta(fac * cos(2 * std::numbers::pi * random_theta2) *
                                            sin(2 * std::numbers::pi * random_phi),
                  fac * sin(2 * std::numbers::pi * random_theta2) *
                      sin(2 * std::numbers::pi * random_phi));

              // construct velocity from alpha, beta and k
              if (k_12 > 1.0e-9)
              {
                (u1_hat)[pos] =
                    (alpha * k * ((double)k_2) + beta * ((double)k_1) * ((double)k_3)) / (k * k_12);
                (u2_hat)[pos] =
                    (beta * ((double)k_2) * ((double)k_3) - alpha * k * ((double)k_1)) / (k * k_12);
                if (k_3 == 0)
                  (u3_hat)[pos] = -beta * k_12 / k;
                else
                  (u3_hat)[pos] =
                      -(((double)k_1) * ((u1_hat)[pos]) + ((double)k_2) * ((u2_hat)[pos])) /
                      ((double)k_3);
              }
              else
              {
                (u1_hat)[pos] = alpha;
                (u2_hat)[pos] = beta;
                (u3_hat)[pos] =
                    -(((double)k_1) * ((u1_hat)[pos]) + ((double)k_2) * ((u2_hat)[pos])) /
                    ((double)k_3);
              }
            }
            else
            {
              (u1_hat)[pos] = conj((u1_hat)[pos_conj]);
              (u2_hat)[pos] = conj((u2_hat)[pos_conj]);
              (u3_hat)[pos] = conj((u3_hat)[pos_conj]);
            }
          }
        }
      }
    }

    // transfer to FFTW structure
    // FFTW assumes wave numbers in the following intervals
    // k_1: [0,(nummodes_-1)]
    // k_2: [0,(nummodes_-1)]
    // k_3: [0,nummodes_/2]
    // using peridocity and conjugate symmetry allows for setting
    // the Fourier coefficients in the required interval
    Teuchos::Array<std::complex<double>> u1_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u2_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u3_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

    for (int fftw_k_1 = 0; fftw_k_1 <= (nummodes_ - 1); fftw_k_1++)
    {
      for (int fftw_k_2 = 0; fftw_k_2 <= (nummodes_ - 1); fftw_k_2++)
      {
        for (int fftw_k_3 = 0; fftw_k_3 <= (nummodes_ / 2); fftw_k_3++)
        {
          int k_1 = -999;
          int k_2 = -999;
          int k_3 = -999;
          bool conjugate = false;

          if ((fftw_k_1 >= 0 and fftw_k_1 <= (nummodes_ / 2 - 1)) and
              (fftw_k_2 >= 0 and fftw_k_2 <= (nummodes_ / 2 - 1)) and fftw_k_3 == 0)
          {
            // wave number vector is part of construction domain
            // and this value is taken
            k_1 = fftw_k_1;
            k_2 = fftw_k_2;
            k_3 = fftw_k_3;
          }
          else
          {
            // see whether the negative wave vector is in the construction domain
            k_1 = -fftw_k_1;
            k_2 = -fftw_k_2;
            k_3 = -fftw_k_3;
            if ((k_1 >= (-nummodes_ / 2) and k_1 <= (nummodes_ / 2 - 1)) and
                (k_2 >= (-nummodes_ / 2) and k_2 <= (nummodes_ / 2 - 1)) and
                (k_3 >= (-nummodes_ / 2) and k_3 <= 0))
            {
              // negative wave vector is in the construction domain
              // conjugate complex have to be used here
              conjugate = true;
            }
            else
            {
              // if negative wave vector is not in the construction domain
              // we have to shift it into the domain using the periodicity of the
              // wave number field
              // -k_3 always lies within the construction domain!
              if (k_1 < (-nummodes_ / 2)) k_1 += nummodes_;
              if (k_2 < (-nummodes_ / 2)) k_2 += nummodes_;

              conjugate = true;
            }
          }

          // get position in u1_hat_fftw
          const int pos = fftw_k_3 + (nummodes_ / 2 + 1) * (fftw_k_2 + nummodes_ * fftw_k_1);
          // and in u1_hat
          const int pos_cond =
              (k_3 + nummodes_ / 2) +
              (nummodes_ / 2 + 1) * ((k_2 + nummodes_ / 2) + nummodes_ * (k_1 + nummodes_ / 2));

          // set value
          if (not conjugate)
          {
            (u1_hat_fftw)[pos] = (u1_hat)[pos_cond];
            (u2_hat_fftw)[pos] = (u2_hat)[pos_cond];
            (u3_hat_fftw)[pos] = (u3_hat)[pos_cond];
          }
          else
          {
            (u1_hat_fftw)[pos] = conj((u1_hat)[pos_cond]);
            (u2_hat_fftw)[pos] = conj((u2_hat)[pos_cond]);
            (u3_hat_fftw)[pos] = conj((u3_hat)[pos_cond]);
          }
        }
      }
    }

    //----------------------------------------
    // fast Fourier transformation using FFTW
    //----------------------------------------

#ifdef FOUR_C_WITH_FFTW
    // set-up
    fftw_plan fft = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
        (reinterpret_cast<fftw_complex*>(u1_hat_fftw.data())), u1.data(), FFTW_ESTIMATE);
    // fft
    fftw_execute(fft);
    // free memory
    fftw_destroy_plan(fft);

    // similar for the remaining two directions
    fftw_plan fft_2 = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
        (reinterpret_cast<fftw_complex*>(u2_hat_fftw.data())), u2.data(), FFTW_ESTIMATE);
    fftw_execute(fft_2);
    // free memory
    fftw_destroy_plan(fft_2);
    fftw_plan fft_3 = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
        (reinterpret_cast<fftw_complex*>(u3_hat_fftw.data())), u3.data(), FFTW_ESTIMATE);
    fftw_execute(fft_3);
    // free memory
    fftw_destroy_plan(fft_3);
    fftw_cleanup();
#else
    FOUR_C_THROW("FFTW required for HIT!");
#endif

    //----------------------------------------
    // set velocity field
    //----------------------------------------

    for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
    {
      // get node
      Core::Nodes::Node* node = discret_->l_row_node(inode);

      // get coordinates
      Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);
      for (int idim = 0; idim < 3; idim++) xyz(idim, 0) = node->x()[idim];

      //    std::cout << "coords " << xyz << std::endl;

      // get global ids of all dofs of the node
      std::vector<int> dofs = discret_->dof(node);

      // determine position
      std::vector<int> loc(3);
      for (int idim = 0; idim < 3; idim++)
      {
        for (std::size_t rr = 0; rr < coordinates_->size(); rr++)
        {
          if ((xyz(idim, 0) <= ((*coordinates_)[rr] + 2e-9)) and
              (xyz(idim, 0) >= ((*coordinates_)[rr] - 2e-9)))
          {
            // due to periodic boundary conditions,
            // the value at the last node is equal to the one at the first node
            // using this strategy, no special care is required for slave nodes
            if ((int)rr < nummodes_)
              loc[idim] = rr;
            else
              loc[idim] = 0;

            break;
          }
        }
      }

      // get position in transferred velocity vectors u_1, u_2 and u_3
      const int pos = loc[2] + nummodes_ * (loc[1] + nummodes_ * loc[0]);

      // get local dof id corresponding to the global id
      int lid = discret_->dof_row_map()->lid(dofs[0]);
      // set value
      velnp_->replace_local_value(lid, u1[pos]);
      // analogous for remaining directions
      lid = discret_->dof_row_map()->lid(dofs[1]);
      velnp_->replace_local_value(lid, u2[pos]);
      lid = discret_->dof_row_map()->lid(dofs[2]);
      velnp_->replace_local_value(lid, u3[pos]);
    }

    // initialize veln_ as well
    veln_->update(1.0, *velnp_, 0.0);
    velnm_->update(1.0, *velnp_, 0.0);

    return;
#else
    FOUR_C_THROW("FFTW required");
#endif
  }


  /*--------------------------------------------------------------*
   | set and non-dimensionalize experimental data rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbInitialField::prepare_exparimental_data()
  {
    //----------------------------------------
    // set-up wave numbers
    //----------------------------------------

    // wave number given from experiment [cm-1]
    // have to be transferred to [m-1] and non-dimensionalized
    k_exp_.resize(19);
    k_exp_[0] = 0.20;
    k_exp_[1] = 0.25;
    k_exp_[2] = 0.30;
    k_exp_[3] = 0.40;
    k_exp_[4] = 0.50;
    k_exp_[5] = 0.70;
    k_exp_[6] = 1.00;
    k_exp_[7] = 1.50;
    k_exp_[8] = 2.00;
    k_exp_[9] = 2.50;
    k_exp_[10] = 3.00;
    k_exp_[11] = 4.00;
    k_exp_[12] = 6.00;
    k_exp_[13] = 8.00;
    k_exp_[14] = 10.00;
    k_exp_[15] = 12.50;
    k_exp_[16] = 15.00;
    k_exp_[17] = 17.50;
    k_exp_[18] = 20.00;

    // non-dimensionalize wave number (according to Collis 2002)
    // grid size of experiment
    const double M = 0.0508;
    // domain length
    const double L = 10.0 * M;
    // reference length
    const double L_ref = L / (2.0 * std::numbers::pi);

    for (std::size_t rr = 0; rr < k_exp_.size(); rr++) k_exp_[rr] *= (L_ref / 0.01);

    //----------------------------------------
    // set-up energy
    //----------------------------------------

    // energy spectrum given from experiment [cm3/s2]
    // have to be transferred to [m3/s2] and non-dimensionalized
    E_exp_.resize(19);
    E_exp_[0] = 129.0;
    E_exp_[1] = 230.0;
    E_exp_[2] = 322.0;
    E_exp_[3] = 435.0;
    E_exp_[4] = 457.0;
    E_exp_[5] = 380.0;
    E_exp_[6] = 270.0;
    E_exp_[7] = 168.0;
    E_exp_[8] = 120.0;
    E_exp_[9] = 89.0;
    E_exp_[10] = 70.3;
    E_exp_[11] = 47.0;
    E_exp_[12] = 24.7;
    E_exp_[13] = 12.6;
    E_exp_[14] = 7.42;
    E_exp_[15] = 3.96;
    E_exp_[16] = 2.33;
    E_exp_[17] = 1.34;
    E_exp_[18] = 0.80;

    // non-dimensionalize energy spectrum
    // inlet velocity of experiment
    const double U_0 = 10.0;
    // reference time
    const double t_ref = 64.0 * M / U_0;

    for (std::size_t rr = 0; rr < E_exp_.size(); rr++)
      E_exp_[rr] *= ((0.01 * 0.01 * 0.01) * (t_ref * t_ref) / (L_ref * L_ref * L_ref));

    return;
  }


  /*--------------------------------------------------------------*
   | get energy for given wave number             rasthofer 04/13 |
   *--------------------------------------------------------------*/
  double HomoIsoTurbInitialField::interpolate_energy_from_spectrum(double k)
  {
    double energy = 0.0;

    // the smallest value for k=1, the next smaller 1.41
    // the following required extrapolation for k<1.6 yields
    // negative values for k=1, which are not physical
    // therefore, energy is set to zero for this case

    // determine position of k
    int position = -1;
    for (std::size_t rr = 0; rr < k_exp_.size(); rr++)
    {
      if (k < k_exp_[rr])
      {
        position = rr;
        break;
      }
    }

    if (position == -1) FOUR_C_THROW("Could not determine wave number!");

    if (position > 0)
      // interpolate energy
      energy = E_exp_[position] + (E_exp_[position - 1] - E_exp_[position]) /
                                      (k_exp_[position] - k_exp_[position - 1]) *
                                      (k_exp_[position] - k);
    else
      // extrapolate energy
      energy = E_exp_[position + 1] - (E_exp_[position + 1] - E_exp_[position]) /
                                          (k_exp_[position + 1] - k_exp_[position]) *
                                          (k_exp_[position + 1] - k);

    // see above
    if (k == 1) energy = 0.0;

    return energy;
  }


  /*--------------------------------------------------------------*
   | get energy for given wave number             rasthofer 05/13 |
   *--------------------------------------------------------------*/
  double HomoIsoTurbInitialField::calculate_energy_from_spectrum(double k)
  {
    // remark: k>0 here!

    double energy = 0.0;

    if (type_ == Inpar::FLUID::initfield_forced_hit_simple_algebraic_spectrum)
    {
      // initial spectrum as used in Hickel et al. 2006
      energy = 0.5 * pow(k, -5.0 / 3.0);
    }
    else if (type_ == Inpar::FLUID::initfield_passive_hit_const_input)
    {
      if (k <= 2)
        energy = 0.1 * 1.0;
      else
        energy = 0.1 * pow(2.0, 5.0 / 3.0) * pow(k, -5.0 / 3.0);
    }
    else if (type_ == Inpar::FLUID::initfield_forced_hit_numeric_spectrum)
    {
      // initial spectrum as used in Bazilevs et al. 2007 (from Langford & Moser 1999)
      std::vector<double> k_vec(48);
      k_vec[0] = 1.43551;
      k_vec[1] = 1.59008;
      k_vec[2] = 1.74499;
      k_vec[3] = 1.93294;
      k_vec[4] = 2.10165;
      k_vec[5] = 2.3064;
      k_vec[6] = 2.50765;
      k_vec[7] = 2.70109;
      k_vec[8] = 2.90939;
      k_vec[9] = 3.1926;
      k_vec[10] = 3.47102;
      k_vec[11] = 3.77363;
      k_vec[12] = 4.14107;
      k_vec[13] = 4.58665;
      k_vec[14] = 5.22421;
      k_vec[15] = 5.78648;
      k_vec[16] = 6.40926;
      k_vec[17] = 7.0994;
      k_vec[18] = 7.93698;
      k_vec[19] = 8.87338;
      k_vec[20] = 9.92049;
      k_vec[21] = 11.0909;
      k_vec[22] = 12.6323;
      k_vec[23] = 14.2543;
      k_vec[24] = 16.085;
      k_vec[25] = 17.8161;
      k_vec[26] = 19.7332;
      k_vec[27] = 21.6541;
      k_vec[28] = 24.2077;
      k_vec[29] = 26.8124;
      k_vec[30] = 29.4218;
      k_vec[31] = 32.8899;
      k_vec[32] = 36.4263;
      k_vec[33] = 40.343;
      k_vec[34] = 44.6798;
      k_vec[35] = 48.5728;
      k_vec[36] = 52.805;
      k_vec[37] = 55.826;
      k_vec[38] = 60.131;
      k_vec[39] = 64.1623;
      k_vec[40] = 69.1053;
      k_vec[41] = 73.7417;
      k_vec[42] = 78.6891;
      k_vec[43] = 82.4224;
      k_vec[44] = 87.1378;
      k_vec[45] = 92.1229;
      k_vec[46] = 98.3035;
      k_vec[47] = 103.935;

      std::vector<double> e_vec(48);
      e_vec[0] = 14.2756;
      e_vec[1] = 12.8539;
      e_vec[2] = 11.5745;
      e_vec[3] = 10.7899;
      e_vec[4] = 10.0599;
      e_vec[5] = 9.05865;
      e_vec[6] = 8.15762;
      e_vec[7] = 6.85399;
      e_vec[8] = 5.56221;
      e_vec[9] = 4.51327;
      e_vec[10] = 3.79176;
      e_vec[11] = 3.07691;
      e_vec[12] = 2.58484;
      e_vec[13] = 2.09723;
      e_vec[14] = 1.82356;
      e_vec[15] = 1.53182;
      e_vec[16] = 1.28676;
      e_vec[17] = 1.15861;
      e_vec[18] = 0.973181;
      e_vec[19] = 0.817432;
      e_vec[20] = 0.710861;
      e_vec[21] = 0.597094;
      e_vec[22] = 0.501464;
      e_vec[23] = 0.406811;
      e_vec[24] = 0.34168;
      e_vec[25] = 0.287018;
      e_vec[26] = 0.232874;
      e_vec[27] = 0.188958;
      e_vec[28] = 0.148072;
      e_vec[29] = 0.12014;
      e_vec[30] = 0.0941574;
      e_vec[31] = 0.0688353;
      e_vec[32] = 0.0503267;
      e_vec[33] = 0.0367947;
      e_vec[34] = 0.0259835;
      e_vec[35] = 0.0196708;
      e_vec[36] = 0.0148917;
      e_vec[37] = 0.0112761;
      e_vec[38] = 0.00915091;
      e_vec[39] = 0.00646395;
      e_vec[40] = 0.0047269;
      e_vec[41] = 0.00357899;
      e_vec[42] = 0.00270985;
      e_vec[43] = 0.00212455;
      e_vec[44] = 0.00160872;
      e_vec[45] = 0.00121814;
      e_vec[46] = 0.000922317;
      e_vec[47] = 0.000775034;

      // determine position of k
      int position = -1;
      for (std::size_t rr = 0; rr < k_vec.size(); rr++)
      {
        if (k < k_vec[rr])
        {
          position = rr;
          break;
        }
      }

      if (position == -1) FOUR_C_THROW("Could not determine wave number!");

      if (position > 0)
        // interpolate energy
        energy = e_vec[position] + (e_vec[position - 1] - e_vec[position]) /
                                       (k_vec[position] - k_vec[position - 1]) *
                                       (k_vec[position] - k);
      else
        // extrapolate energy
        energy = e_vec[position + 1] - (e_vec[position + 1] - e_vec[position]) /
                                           (k_vec[position + 1] - k_vec[position]) *
                                           (k_vec[position + 1] - k);
    }

    return energy;
  }

  /*--------------------------------------------------------------*
   | constructor                                         bk 03/15 |
   *--------------------------------------------------------------*/
  HomoIsoTurbInitialFieldHDG::HomoIsoTurbInitialFieldHDG(
      FluidImplicitTimeInt& timeint, const Inpar::FLUID::InitialField initfield)
      : HomoIsoTurbInitialField(timeint, initfield)
  {
    // here we are using the interior velocity
    TimIntHDG* hdgfluid = dynamic_cast<TimIntHDG*>(&timeint);
    if (hdgfluid == nullptr) FOUR_C_THROW("this should be a hdg time integer");

    // we want to use the interior velocity here
    intvelnp_ = hdgfluid->return_int_velnp();
    intveln_ = hdgfluid->return_int_veln();
    intvelnm_ = hdgfluid->return_int_velnm();

    //-----------------------------------
    // determine number of modes
    //-----------------------------------

    // number of modes equal to 5 times number of elements in one spatial direction
    nummodes_ *= 5;

    //-------------------------------------------------
    // create set of node coordinates
    //-------------------------------------------------

    // push coordinates in vector
    {
      std::vector<double> copycoordinates;

      for (std::vector<double>::iterator coord1 = coordinates_->begin();
          coord1 != coordinates_->end(); ++coord1)
      {
        copycoordinates.push_back(*coord1);
      }

      coordinates_->clear();

      double elesize = abs(copycoordinates.at(1) - copycoordinates.at(0));
      // use 5 sampling locations in each element in each direction
      const std::array<double, 5> localcoords = {0.9, 0.7, 0.5, 0.3, 0.1};
      for (std::vector<double>::iterator coord1 = copycoordinates.begin();
          coord1 != copycoordinates.end(); ++coord1)
      {
        if (coord1 != copycoordinates.begin())
          for (int i = 0; i < 5; i++) coordinates_->push_back(*coord1 - elesize * localcoords[i]);
      }
    }

    return;
  }


  /*--------------------------------------------------------------*
   | calculate initial field using fft                   bk 03/15 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbInitialFieldHDG::calculate_initial_field()
  {
#ifdef FOUR_C_WITH_FFTW

    // set and initialize working arrays
    Teuchos::Array<std::complex<double>> u1_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u2_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u3_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

    Teuchos::Array<double> u1(nummodes_ * nummodes_ * nummodes_);
    Teuchos::Array<double> u2(nummodes_ * nummodes_ * nummodes_);
    Teuchos::Array<double> u3(nummodes_ * nummodes_ * nummodes_);

    //-------------------------------------------------
    // construction of initial field in spectral space
    //-------------------------------------------------

    // as given in the respective literature, the Fourier coefficients are
    // evaluated in the following intervals
    // k_1: [-nummodes_/2,(nummodes_/2-1)]
    // k_2: [-nummodes_/2,(nummodes_/2-1)]
    // k_3: [-nummodes_/2,0]
    for (int k_1 = (-nummodes_ / 2); k_1 <= (nummodes_ / 2 - 1); k_1++)
    {
      for (int k_2 = (-nummodes_ / 2); k_2 <= (nummodes_ / 2 - 1); k_2++)
      {
        for (int k_3 = (-nummodes_ / 2); k_3 <= 0; k_3++)
        {
          // get position in u1_hat
          const int pos =
              (k_3 + nummodes_ / 2) +
              (nummodes_ / 2 + 1) * ((k_2 + nummodes_ / 2) + nummodes_ * (k_1 + nummodes_ / 2));

          if (k_1 == (-nummodes_ / 2) or k_2 == (-nummodes_ / 2) or k_3 == (-nummodes_ / 2))
          {
            // odd-ball wave numbers are set to zero to ensure that solution is a real function
            ((u1_hat)[pos]).real(0.0);
            // this is important to have here
            ((u1_hat)[pos]).imag(0.0);
            // remaining analogously
            ((u2_hat)[pos]).real(0.0);
            ((u2_hat)[pos]).imag(0.0);
            ((u3_hat)[pos]).real(0.0);
            ((u3_hat)[pos]).imag(0.0);
          }
          else if (k_1 == 0 and k_2 == 0 and k_3 == 0)
          {
            // likewise set to zero since there will not be any conjugate complex
            ((u1_hat)[pos]).real(0.0);
            // this is important to have here
            ((u1_hat)[pos]).imag(0.0);
            // remaining analogously
            ((u2_hat)[pos]).real(0.0);
            ((u2_hat)[pos]).imag(0.0);
            ((u3_hat)[pos]).real(0.0);
            ((u3_hat)[pos]).imag(0.0);
          }
          else
          {
            bool calculate = true;
            // check if conjugate complex has already been set
            int pos_conj = -999;
            if (k_3 == 0)
            {
              pos_conj = (-k_3 + nummodes_ / 2) +
                         (nummodes_ / 2 + 1) *
                             ((-k_2 + nummodes_ / 2) + nummodes_ * (-k_1 + nummodes_ / 2));

              if (pos_conj < pos) calculate = false;
            }

            if (calculate)
            {
              const double k_12 = sqrt(k_1 * k_1 + k_2 * k_2);
              const double k = sqrt(k_1 * k_1 + k_2 * k_2 + k_3 * k_3);

              double random_theta1 = 0.0;
              double random_theta2 = 0.0;
              double random_phi = 0.0;

              // random numbers are created by one processor and
              // then send to the other processors
              // this ensures that all processors construct the same
              // initial field, which is important to get a matching
              // velocity field in physical space
              if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
              {
                Core::Utils::Random* random = Global::Problem::instance()->random();
                // set range [0;1] (default: [-1;1])
                //              random->SetRandRange(0.0,1.0);
                //              random_theta1 = random->Uni();
                //              random_theta2 = random->Uni();
                //              random_phi = random->Uni();
                random_theta1 = 0.5 * random->uni() + 0.5;
                random_theta2 = 0.5 * random->uni() + 0.5;
                random_phi = 0.5 * random->uni() + 0.5;
              }
              Core::Communication::broadcast(&random_theta1, 1, 0, discret_->get_comm());
              Core::Communication::broadcast(&random_theta2, 1, 0, discret_->get_comm());
              Core::Communication::broadcast(&random_phi, 1, 0, discret_->get_comm());

              // estimate energy at wave number from energy spectrum
              double energy = 0.0;
              if (type_ == Inpar::FLUID::initfield_hit_comte_bellot_corrsin)
                energy = interpolate_energy_from_spectrum(k);
              else
                energy = calculate_energy_from_spectrum(k);

              if (energy < 0.0)
              {
                std::cout << "k " << k << std::endl;
                std::cout << "k1  " << k_1 << std::endl;
                std::cout << "k2  " << k_2 << std::endl;
                std::cout << "k3  " << k_3 << std::endl;
                FOUR_C_THROW("Negative energy!");
              }

              // remark on the literature:
              // Collis 2002: sqrt(energy/(2*PI*k))
              const double fac = sqrt(energy / (2 * std::numbers::pi * k * k));
              // Rogallo 1981: sqrt(energy/(4*PI*k*k))
              // the missing factor 1/2 of Collis version compared to Rogallo version
              // is related to the definition of E from u_i
              // here, we have E = 1/2 * u_i *u_i (see statistics manager)

              // real part, imaginary part
              std::complex<double> alpha(fac * cos(2 * std::numbers::pi * random_theta1) *
                                             cos(2 * std::numbers::pi * random_phi),
                  fac * sin(2 * std::numbers::pi * random_theta1) *
                      cos(2 * std::numbers::pi * random_phi));
              std::complex<double> beta(fac * cos(2 * std::numbers::pi * random_theta2) *
                                            sin(2 * std::numbers::pi * random_phi),
                  fac * sin(2 * std::numbers::pi * random_theta2) *
                      sin(2 * std::numbers::pi * random_phi));

              // construct velocity from alpha, beta and k
              if (k_12 > 1.0e-9)
              {
                (u1_hat)[pos] =
                    (alpha * k * ((double)k_2) + beta * ((double)k_1) * ((double)k_3)) / (k * k_12);
                (u2_hat)[pos] =
                    (beta * ((double)k_2) * ((double)k_3) - alpha * k * ((double)k_1)) / (k * k_12);
                if (k_3 == 0)
                  (u3_hat)[pos] = -beta * k_12 / k;
                else
                  (u3_hat)[pos] =
                      -(((double)k_1) * ((u1_hat)[pos]) + ((double)k_2) * ((u2_hat)[pos])) /
                      ((double)k_3);
              }
              else
              {
                (u1_hat)[pos] = alpha;
                (u2_hat)[pos] = beta;
                (u3_hat)[pos] =
                    -(((double)k_1) * ((u1_hat)[pos]) + ((double)k_2) * ((u2_hat)[pos])) /
                    ((double)k_3);
              }
            }
            else
            {
              (u1_hat)[pos] = conj((u1_hat)[pos_conj]);
              (u2_hat)[pos] = conj((u2_hat)[pos_conj]);
              (u3_hat)[pos] = conj((u3_hat)[pos_conj]);
            }
          }
        }
      }
    }

    // transfer to FFTW structure
    // FFTW assumes wave numbers in the following intervals
    // k_1: [0,(nummodes_-1)]
    // k_2: [0,(nummodes_-1)]
    // k_3: [0,nummodes_/2]
    // using peridocity and conjugate symmetry allows for setting
    // the Fourier coefficients in the required interval
    Teuchos::Array<std::complex<double>> u1_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u2_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<std::complex<double>> u3_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

    for (int fftw_k_1 = 0; fftw_k_1 <= (nummodes_ - 1); fftw_k_1++)
    {
      for (int fftw_k_2 = 0; fftw_k_2 <= (nummodes_ - 1); fftw_k_2++)
      {
        for (int fftw_k_3 = 0; fftw_k_3 <= (nummodes_ / 2); fftw_k_3++)
        {
          int k_1 = -999;
          int k_2 = -999;
          int k_3 = -999;
          bool conjugate = false;

          if ((fftw_k_1 >= 0 and fftw_k_1 <= (nummodes_ / 2 - 1)) and
              (fftw_k_2 >= 0 and fftw_k_2 <= (nummodes_ / 2 - 1)) and fftw_k_3 == 0)
          {
            // wave number vector is part of construction domain
            // and this value is taken
            k_1 = fftw_k_1;
            k_2 = fftw_k_2;
            k_3 = fftw_k_3;
          }
          else
          {
            // see whether the negative wave vector is in the construction domain
            k_1 = -fftw_k_1;
            k_2 = -fftw_k_2;
            k_3 = -fftw_k_3;
            if ((k_1 >= (-nummodes_ / 2) and k_1 <= (nummodes_ / 2 - 1)) and
                (k_2 >= (-nummodes_ / 2) and k_2 <= (nummodes_ / 2 - 1)) and
                (k_3 >= (-nummodes_ / 2) and k_3 <= 0))
            {
              // negative wave vector is in the construction domain
              // conjugate complex have to be used here
              conjugate = true;
            }
            else
            {
              // if negative wave vector is not in the construction domain
              // we have to shift it into the domain using the periodicity of the
              // wave number field
              // -k_3 always lies within the construction domain!
              if (k_1 < (-nummodes_ / 2)) k_1 += nummodes_;
              if (k_2 < (-nummodes_ / 2)) k_2 += nummodes_;

              conjugate = true;
            }
          }

          // get position in u1_hat_fftw
          const int pos = fftw_k_3 + (nummodes_ / 2 + 1) * (fftw_k_2 + nummodes_ * fftw_k_1);
          // and in u1_hat
          const int pos_cond =
              (k_3 + nummodes_ / 2) +
              (nummodes_ / 2 + 1) * ((k_2 + nummodes_ / 2) + nummodes_ * (k_1 + nummodes_ / 2));

          // set value
          if (not conjugate)
          {
            (u1_hat_fftw)[pos] = (u1_hat)[pos_cond];
            (u2_hat_fftw)[pos] = (u2_hat)[pos_cond];
            (u3_hat_fftw)[pos] = (u3_hat)[pos_cond];
          }
          else
          {
            (u1_hat_fftw)[pos] = conj((u1_hat)[pos_cond]);
            (u2_hat_fftw)[pos] = conj((u2_hat)[pos_cond]);
            (u3_hat_fftw)[pos] = conj((u3_hat)[pos_cond]);
          }
        }
      }
    }

    //----------------------------------------
    // fast Fourier transformation using FFTW
    //----------------------------------------

#ifdef FOUR_C_WITH_FFTW
    // set-up
    fftw_plan fft = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
        (reinterpret_cast<fftw_complex*>(u1_hat_fftw.data())), u1.data(), FFTW_ESTIMATE);
    // fft
    fftw_execute(fft);
    // free memory
    fftw_destroy_plan(fft);

    // similar for the remaining two directions
    fftw_plan fft_2 = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
        (reinterpret_cast<fftw_complex*>(u2_hat_fftw.data())), u2.data(), FFTW_ESTIMATE);
    fftw_execute(fft_2);
    // free memory
    fftw_destroy_plan(fft_2);
    fftw_plan fft_3 = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
        (reinterpret_cast<fftw_complex*>(u3_hat_fftw.data())), u3.data(), FFTW_ESTIMATE);
    fftw_execute(fft_3);
    // free memory
    fftw_destroy_plan(fft_3);
    fftw_cleanup();
#else
    FOUR_C_THROW("FFTW required for HIT!");
#endif

    //----------------------------------------
    // set velocity field
    //----------------------------------------

    // for 1st evaluate
    Teuchos::ParameterList params;
    params.set<FLD::Action>("action", FLD::interpolate_hdg_for_hit);

    std::vector<int> dummy;
    Core::LinAlg::SerialDenseMatrix dummyMat;
    Core::LinAlg::SerialDenseVector dummyVec;
    // this is a dummy, should be zero is written in the first components of interpolVec
    intvelnp_->put_scalar(0.0);
    // set dummy
    discret_->set_state(1, "intvelnp", *intvelnp_);

    // for 2nd evaluate
    Core::LinAlg::SerialDenseVector elevec1, elevec3;
    Core::LinAlg::SerialDenseMatrix elemat1, elemat2;
    Teuchos::ParameterList initParams;
    initParams.set<FLD::Action>("action", FLD::project_hdg_initial_field_for_hit);

    // loop over all elements on the processor
    Core::Elements::LocationArray la(2);
    double error = 0;
    for (int el = 0; el < discret_->num_my_row_elements(); ++el)
    {
      // 1st evaluate
      Core::Elements::Element* ele = discret_->l_row_element(el);

      Core::LinAlg::SerialDenseVector interpolVec;
      interpolVec.resize(5 * 5 * 5 * 6);  // 5*5*5 points: velx, vely, velz, x, y, z

      ele->evaluate(params, *discret_, dummy, dummyMat, dummyMat, interpolVec, dummyVec, dummyVec);

      // sum values on nodes into vectors and record the touch count (build average of values)
      for (int i = 0; i < 5 * 5 * 5; ++i)
      {
        // get coordinates
        Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);
        for (int d = 0; d < 3; ++d) xyz(d) = interpolVec(i * 6 + d + 3);
        // determine position
        std::vector<int> loc(3);

        for (int idim = 0; idim < 3; idim++)
        {
          for (std::size_t rr = 0; rr < coordinates_->size(); rr++)
          {
            if ((xyz(idim, 0) <= ((*coordinates_)[rr] + 2e-9)) and
                (xyz(idim, 0) >= ((*coordinates_)[rr] - 2e-9)))
            {
              // due to periodic boundary conditions,
              // the value at the last node is equal to the one at the first node
              // using this strategy, no special care is required for slave nodes
              if ((int)rr < nummodes_)
                loc[idim] = rr;
              else
                FOUR_C_THROW("I think that this should not happen");

              break;
            }
          }
        }

        // get position in velocity vectors local_u_1, local_u_2 and local_u_3
        const int pos = loc[2] + nummodes_ * (loc[1] + nummodes_ * loc[0]);

        // set value
        interpolVec(i * 6 + 0) = (u1)[pos];
        interpolVec(i * 6 + 1) = (u2)[pos];
        interpolVec(i * 6 + 2) = (u3)[pos];
      }

      // 2nd evaluate
      ele->location_vector(*discret_, la);
      if (elevec1.numRows() != discret_->num_dof(1, ele)) elevec1.size(discret_->num_dof(1, ele));
      if (static_cast<std::size_t>(elevec3.numRows()) != la[0].lm_.size())
        elevec3.size(la[0].lm_.size());

      ele->evaluate(
          initParams, *discret_, la[0].lm_, elemat1, elemat2, elevec1, interpolVec, elevec3);

      if (ele->owner() == Core::Communication::my_mpi_rank(discret_->get_comm()))
      {
        std::vector<int> globalDofs = discret_->dof(1, ele);
        FOUR_C_ASSERT(
            globalDofs.size() == static_cast<std::size_t>(elevec1.numRows()), "Internal error");
        intvelnp_->replace_global_values(globalDofs.size(), elevec1.values(), globalDofs.data());
      }

      // now fill the ele vector into the discretization
      for (unsigned int i = 0; i < la[0].lm_.size(); ++i)
      {
        const int lid = discret_->dof_row_map()->lid(la[0].lm_[i]);
        if (lid >= 0)
        {
          // Here we are facing a difficulty:
          // the initial field is not continuous but the projection results in jumps at element
          // boundaries the question is, which value should be used for the traces I am here just
          // using the average of both elements
          if (velnp_->local_values_as_span()[lid] != 0)
          {
            double tmp = velnp_->local_values_as_span()[lid];
            (*velnp_).get_values()[lid] = 0.5 * (tmp + elevec3(i));
          }
          else
            (*velnp_).get_values()[lid] = elevec3(i);
        }
      }
    }
    std::cout << "the error due to projection of the solution from only one side is " << error
              << std::endl;

    // initialize veln_ as well
    intveln_->update(1.0, *intvelnp_, 0.0);
    intvelnm_->update(1.0, *intvelnp_, 0.0);
    veln_->update(1.0, *velnp_, 0.0);
    velnm_->update(1.0, *velnp_, 0.0);
    discret_->clear_state(true);
    return;
#else
    FOUR_C_THROW("FFTW required");
#endif
  }

};  // namespace FLD

FOUR_C_NAMESPACE_CLOSE
