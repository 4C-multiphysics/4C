// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_turbulence_hit_initial_scalar_field.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <cmath>
#include <complex>

#ifdef FOUR_C_WITH_FFTW
#include <fftw3.h>
#endif

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  /*--------------------------------------------------------------*
   | constructor                                  rasthofer 04/13 |
   *--------------------------------------------------------------*/
  HomoIsoTurbInitialScalarField::HomoIsoTurbInitialScalarField(
      ScaTraTimIntImpl& timeint, const Inpar::ScaTra::InitialField initfield)
      : discret_(timeint.discret_), phinp_(timeint.phinp_), phin_(timeint.phin_), type_(initfield)
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

    return;
  }


  /*--------------------------------------------------------------*
   | calculate initial field using fft            rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbInitialScalarField::calculate_initial_field()
  {
#ifdef FOUR_C_WITH_FFTW
    // set and initialize working arrays
    Teuchos::Array<std::complex<double>> phi_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

    Teuchos::Array<double> phi(nummodes_ * nummodes_ * nummodes_);

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
          // get position in phi_hat
          const int pos =
              (k_3 + nummodes_ / 2) +
              (nummodes_ / 2 + 1) * ((k_2 + nummodes_ / 2) + nummodes_ * (k_1 + nummodes_ / 2));

          if (k_1 == (-nummodes_ / 2) or k_2 == (-nummodes_ / 2) or k_3 == (-nummodes_ / 2))
          {
            // odd-ball wave numbers are set to zero to ensure that solution is real function
            ((phi_hat)[pos]).real(0.0);
            // this is important to have here
            ((phi_hat)[pos]).imag(0.0);
          }
          else if (k_1 == 0 and k_2 == 0 and k_3 == 0)
          {
            // likewise set to zero since there will not be any conjugate complex
            ((phi_hat)[pos]).real(0.0);
            // this is important to have here
            ((phi_hat)[pos]).imag(0.0);
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
              const double k = sqrt(k_1 * k_1 + k_2 * k_2 + k_3 * k_3);

              double random_theta = 0.0;

              // random numbers are created by one processor and
              // then send to the other processors
              // this ensures that all processors construct the same
              // initial field, which is important to get a matching
              // scalar field in physical space
              if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
              {
                Core::Utils::Random* random = Global::Problem::instance()->random();
                // set range [0;1] (default: [-1;1])
                //              random->SetRandRange(0.0,1.0);
                //              random_theta = random->Uni();
                // use this version to get random field different from fluid
                random_theta = 0.5 * random->uni() + 0.5;
              }
              Core::Communication::broadcast(&random_theta, 1, 0, discret_->get_comm());

              // estimate energy at wave number from energy spectrum
              const double energy = calculate_energy_from_spectrum(k);

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
              // is related to the definition of E from phi
              // here, we have E = 1/2 * phi * phi (see statistics manager)

              // real part, imaginary part
              std::complex<double> alpha(fac * cos(2 * std::numbers::pi * random_theta),
                  fac * sin(2 * std::numbers::pi * random_theta));
              (phi_hat)[pos] = alpha;
            }
            else
            {
              (phi_hat)[pos] = conj((phi_hat)[pos_conj]);
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
    Teuchos::Array<std::complex<double>> phi_hat_fftw(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

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

          // get position in phi_hat_fftw
          const int pos = fftw_k_3 + (nummodes_ / 2 + 1) * (fftw_k_2 + nummodes_ * fftw_k_1);
          // and in phi_hat
          const int pos_cond =
              (k_3 + nummodes_ / 2) +
              (nummodes_ / 2 + 1) * ((k_2 + nummodes_ / 2) + nummodes_ * (k_1 + nummodes_ / 2));

          // set value
          if (not conjugate)
          {
            (phi_hat_fftw)[pos] = (phi_hat)[pos_cond];
          }
          else
          {
            (phi_hat_fftw)[pos] = conj((phi_hat)[pos_cond]);
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
        (reinterpret_cast<fftw_complex*>(phi_hat_fftw.data())), phi.data(), FFTW_ESTIMATE);
    // fft
    fftw_execute(fft);
    // free memory
    fftw_destroy_plan(fft);
    fftw_cleanup();
#else
    FOUR_C_THROW("FFTW required for HIT!");
#endif

    //----------------------------------------
    // set scalar field
    //----------------------------------------

    for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
    {
      // get node
      Core::Nodes::Node* node = discret_->l_row_node(inode);

      // get coordinates
      Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);
      for (int idim = 0; idim < 3; idim++) xyz(idim, 0) = node->x()[idim];

      // get global ids of all dofs of the node
      std::vector<int> dofs = discret_->dof(0, node);
      if (dofs.size() > 1)
        FOUR_C_THROW("Only one dof per node for homogeneous isotropic turbulence!");

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

      // get position in transferred phi vector
      const int pos = loc[2] + nummodes_ * (loc[1] + nummodes_ * loc[0]);

      // get local dof id corresponding to the global id
      int lid = discret_->dof_row_map()->lid(dofs[0]);
      // set value
      phinp_->replace_local_value(lid, phi[pos]);
    }

    // initialize phin_ as well
    phin_->update(1.0, *phinp_, 0.0);

    return;
#else
    FOUR_C_THROW("FFTW required");
#endif
  }


  /*--------------------------------------------------------------*
   | get energy for given wave number             rasthofer 05/13 |
   *--------------------------------------------------------------*/
  double HomoIsoTurbInitialScalarField::calculate_energy_from_spectrum(double k)
  {
    // remark: k > 0 here
    double energy = 0.0;

    if (type_ == Inpar::ScaTra::initialfield_forced_hit_low_Sc)
    {
      if (k > 2.0)
        energy = 0.1 * pow(2.0, 5.0 / 3.0) * pow(k, -5.0 / 3.0);
      else
        energy = 0.1 * 1.0;
    }
    else if (type_ == Inpar::ScaTra::initialfield_forced_hit_high_Sc)
    {
      if (k > 2.0)
        energy = 0.1 * 2.0 * pow(k, -1.0);
      else
        energy = 0.1 * 1.0;
    }
    else
      FOUR_C_THROW("Unknown initial field!");

    return energy;
  }


};  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE
