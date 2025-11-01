// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_scatra_timint_genalpha.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <complex>

#ifdef FOUR_C_WITH_FFTW
#include <fftw3.h>
#endif

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

#define USE_TARGET_SPECTRUM
// #define TIME_UPDATE_FORCING_SPECTRUM

namespace ScaTra
{
  /*--------------------------------------------------------------*
   | constructor                                  rasthofer 04/13 |
   *--------------------------------------------------------------*/
  HomoIsoTurbScalarForcing::HomoIsoTurbScalarForcing(ScaTraTimIntImpl* timeint)
      : forcing_type_(Teuchos::getIntegralValue<Inpar::FLUID::ForcingType>(
            timeint->extraparams_->sublist("TURBULENCE MODEL"), "FORCING_TYPE")),
        discret_(timeint->discret_),
        forcing_(timeint->forcing_),
        phinp_(timeint->phinp_),
        phiaf_(nullptr),
        threshold_wavenumber_(timeint->extraparams_->sublist("TURBULENCE MODEL")
                .get<double>("THRESHOLD_WAVENUMBER", 0)),
        is_genalpha_(false),
        dt_(timeint->dta_)
  {
    // set gen-alpha
    TimIntGenAlpha* timeint_genalpha = dynamic_cast<TimIntGenAlpha*>(timeint);
    if (timeint_genalpha != nullptr)
    {
      is_genalpha_ = true;
      phiaf_ = timeint_genalpha->phiaf();
    }

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
    // create set of wave numbers
    //-------------------------------------------------

    // push wave numbers in vector
    {
      wavenumbers_ = std::make_shared<std::vector<double>>();

      wavenumbers_->resize((std::size_t)nummodes_);
      for (std::size_t rr = 0; rr < wavenumbers_->size(); rr++) (*wavenumbers_)[rr] = rr;
    }

    // set size of energy-spectrum vector
    scalarvariancespectrum_n_ = std::make_shared<std::vector<double>>();
    scalarvariancespectrum_n_->resize(wavenumbers_->size());
    scalarvariancespectrum_np_ = std::make_shared<std::vector<double>>();
    scalarvariancespectrum_np_->resize(wavenumbers_->size());
    // and initialize with zeros, just to be sure
    for (std::size_t rr = 0; rr < scalarvariancespectrum_n_->size(); rr++)
    {
      (*scalarvariancespectrum_n_)[rr] = 0.0;
      (*scalarvariancespectrum_np_)[rr] = 0.0;
    }

    // linear compensation factor for isotropic forcing
    force_fac_ = std::make_shared<Core::LinAlg::SerialDenseVector>(
        nummodes_ * nummodes_ * (nummodes_ / 2 + 1));

    return;
  }


  /*--------------------------------------------------------------*
   | initialize energy spectrum by initial field  rasthofer 05/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbScalarForcing::set_initial_spectrum(Inpar::ScaTra::InitialField init_field_type)
  {
#ifdef USE_TARGET_SPECTRUM
    (*scalarvariancespectrum_n_)[0] = 0.0;
    for (std::size_t rr = 1; rr < wavenumbers_->size(); rr++)
    {
      if (init_field_type == Inpar::ScaTra::initialfield_forced_hit_low_Sc)
      {
        if ((*wavenumbers_)[rr] > 0.0 and (*wavenumbers_)[rr] <= 2.0)
          (*scalarvariancespectrum_n_)[rr] = 0.1 * 1.0;
        else
          (*scalarvariancespectrum_n_)[rr] =
              0.1 * pow(2.0, 5.0 / 3.0) * pow((*wavenumbers_)[rr], -5.0 / 3.0);
      }
      else if (init_field_type == Inpar::ScaTra::initialfield_forced_hit_high_Sc)
      {
        if ((*wavenumbers_)[rr] > 0.0 and (*wavenumbers_)[rr] <= 2.0)
          (*scalarvariancespectrum_n_)[rr] = 0.1 * 1.0;
        else
          (*scalarvariancespectrum_n_)[rr] = 0.1 * 2.0 * pow((*wavenumbers_)[rr], -1.0);
      }
      else
        FOUR_C_THROW("Other initial spectra than simple algebraic spectrum not yet implemented!");
    }
#else
    CalculateForcing(0);
    if (forcing_type_ == Inpar::FLUID::linear_compensation_from_intermediate_spectrum)
      TimeUpdateForcing();
#endif
    return;
  }


  /*--------------------------------------------------------------*
   | activate calculation of forcing              rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbScalarForcing::activate_forcing(const bool activate)
  {
    activate_ = activate;
    return;
  }


  /*--------------------------------------------------------------*
   | calculate volume force                       rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbScalarForcing::calculate_forcing(const int step)
  {
#ifdef FOUR_C_WITH_FFTW
    //-------------------------------------------------------------------------------
    // calculate Fourier transformation of velocity
    //-------------------------------------------------------------------------------

    // set and initialize working arrays
    Teuchos::Array<std::complex<double>> phi_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
    Teuchos::Array<double> local_phi(nummodes_ * nummodes_ * nummodes_);
    Teuchos::Array<double> global_phi(nummodes_ * nummodes_ * nummodes_);

    //-----------------------------------
    // prepare Fourier transformation
    //-----------------------------------

    // set solution in local vectors for phi

    for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
    {
      // get node
      Core::Nodes::Node* node = discret_->l_row_node(inode);

      // get coordinates
      Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);
      for (int idim = 0; idim < 3; idim++) xyz(idim, 0) = node->x()[idim];

      // get global ids of all dofs of the node
      std::vector<int> dofs = discret_->dof(0, node);

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

      // get position in velocity vectors local_u_1, local_u_2 and local_u_3
      const int pos = loc[2] + nummodes_ * (loc[1] + nummodes_ * loc[0]);

      // get local dof id corresponding to the global id
      int lid = discret_->dof_row_map()->lid(dofs[0]);
      // set value
      // since we are interested at E at n+1, we use phi at n+1 also for gen-alpha
      (local_phi)[pos] = phinp_->local_values_as_span()[lid];
    }

    global_phi = Core::Communication::sum_all(local_phi, discret_->get_comm());

    //----------------------------------------
    // fast Fourier transformation using FFTW
    //----------------------------------------

    // note: this is not very efficient, since each
    // processor does the fft and there is no communication

#ifdef FOUR_C_WITH_FFTW
    // set-up
    fftw_plan fft = fftw_plan_dft_r2c_3d(nummodes_, nummodes_, nummodes_, global_phi.data(),
        (reinterpret_cast<fftw_complex*>(phi_hat.data())), FFTW_ESTIMATE);
    // fft
    fftw_execute(fft);
    // free memory
    fftw_destroy_plan(fft);
    fftw_cleanup();

    // scale solution (not done in the fftw routine)
    for (int i = 0; i < phi_hat.size(); i++)
    {
      (phi_hat)[i] /= nummodes_ * nummodes_ * nummodes_;
    }
#else
    FOUR_C_THROW("FFTW required for HIT!");
#endif

    //----------------------------------------
    // compute energy spectrum
    //----------------------------------------

    // transfer from FFTW structure to intervals around zero
    // FFTW assumes wave numbers in the following intervals
    // k_1: [0,(nummodes_-1)]
    // k_2: [0,(nummodes_-1)]
    // k_3: [0,nummodes_/2]
    // here, we would like to have
    // k_1: [-nummodes_/2,(nummodes_/2-1)]
    // k_2: [-nummodes_/2,(nummodes_/2-1)]
    // k_3: [-nummodes_/2,0]
    // using peridocity and conjugate symmetry allows for setting
    // the Fourier coefficients in the required interval

    // reset energy spectrum at time n+1
    for (std::size_t rr = 0; rr < scalarvariancespectrum_np_->size(); rr++)
      (*scalarvariancespectrum_np_)[rr] = 0.0;

    // the complete number of modes is required here
    // hence, we have k_3: [-nummodes_/2,(nummodes_/2-1)]
    for (int k_1 = (-nummodes_ / 2); k_1 <= (nummodes_ / 2 - 1); k_1++)
    {
      for (int k_2 = (-nummodes_ / 2); k_2 <= (nummodes_ / 2 - 1); k_2++)
      {
        for (int k_3 = (-nummodes_ / 2); k_3 <= (nummodes_ / 2 - 1); k_3++)
        {
          // initialize position in FFTW vectors
          int pos_fftw_k_1 = -999;
          int pos_fftw_k_2 = -999;
          int pos_fftw_k_3 = -999;

          // check if current wave vector lies within the fftw domain
          if ((k_1 >= 0 and k_1 <= (nummodes_ / 2 - 1)) and
              (k_2 >= 0 and k_2 <= (nummodes_ / 2 - 1)) and
              (k_3 >= 0 and k_3 <= (nummodes_ / 2 - 1)))
          {
            pos_fftw_k_1 = k_1;
            pos_fftw_k_2 = k_2;
            pos_fftw_k_3 = k_3;
          }
          else
          {
            // if k_3 is < 0, we have to take the conjugate
            // to get into the FFTW domain
            if (k_3 < 0)
            {
              int k_conj_1 = -k_1;
              int k_conj_2 = -k_2;
              int k_conj_3 = -k_3;

              // check if conjugate wave vector lies within the fftw domain
              // this has to be fulfilled for k_3 but not for k_1 and k_2
              if ((k_conj_1 >= 0 and k_conj_1 <= (nummodes_ - 1)) and
                  (k_conj_2 >= 0 and k_conj_2 <= (nummodes_ - 1)) and
                  (k_conj_3 >= 0 and k_conj_3 <= (nummodes_ - 1)))
              {
                pos_fftw_k_1 = k_conj_1;
                pos_fftw_k_2 = k_conj_2;
                pos_fftw_k_3 = k_conj_3;
              }
              else
              {
                if (not(k_conj_3 >= 0 and k_conj_3 <= (nummodes_ - 1)))
                  FOUR_C_THROW("k_3 in fftw domain expected!");

                // shift k_1 and k_2 into fftw domain
                if (k_conj_1 < 0) k_conj_1 += nummodes_;
                if (k_conj_2 < 0) k_conj_2 += nummodes_;

                if ((k_conj_1 >= 0 and k_conj_1 <= (nummodes_ - 1)) and
                    (k_conj_2 >= 0 and k_conj_2 <= (nummodes_ - 1)) and
                    (k_conj_3 >= 0 and k_conj_3 <= (nummodes_ - 1)))
                {
                  pos_fftw_k_1 = k_conj_1;
                  pos_fftw_k_2 = k_conj_2;
                  pos_fftw_k_3 = k_conj_3;
                }
                else
                  FOUR_C_THROW("Position in fftw domain expected!");
              }
            }
            else
            {
              int k_shift_1 = k_1;
              int k_shift_2 = k_2;
              int k_shift_3 = k_3;

              if (not(k_shift_3 >= 0 and k_shift_3 <= (nummodes_ - 1)))
                FOUR_C_THROW("k_3 in fftw domain expected!");

              // shift k_1 and k_2 into fftw domain
              if (k_shift_1 < 0) k_shift_1 += nummodes_;
              if (k_shift_2 < 0) k_shift_2 += nummodes_;

              if ((k_shift_1 >= 0 and k_shift_1 <= (nummodes_ - 1)) and
                  (k_shift_2 >= 0 and k_shift_2 <= (nummodes_ - 1)) and
                  (k_shift_3 >= 0 and k_shift_3 <= (nummodes_ - 1)))
              {
                pos_fftw_k_1 = k_shift_1;
                pos_fftw_k_2 = k_shift_2;
                pos_fftw_k_3 = k_shift_3;
              }
              else
                FOUR_C_THROW("Position in fftw domain expected!");
            }
          }

          // get position in phi_hat
          const int pos =
              pos_fftw_k_3 + (nummodes_ / 2 + 1) * (pos_fftw_k_2 + nummodes_ * pos_fftw_k_1);

          // calculate energy
          // E = 1/2 * u_i * conj(u_i)
          // u_i * conj(u_i) = real(u_i)^2 + imag(u_i)^2
          // const std::complex<double> energy = 0.5 * ((*phi_hat)[pos] * conj((*phi_hat)[pos]);
          // instead
          const double scalarvariance = 0.5 * norm((phi_hat)[pos]);

          if (forcing_type_ == Inpar::FLUID::linear_compensation_from_intermediate_spectrum)
          {
            // get wave number
            const double k = sqrt(k_1 * k_1 + k_2 * k_2 + k_3 * k_3);

            // insert into sampling vector
            // find position via k
            for (std::size_t rr = 0; rr < wavenumbers_->size(); rr++)
            {
              if (k > ((*wavenumbers_)[rr] - 0.5) and k <= ((*wavenumbers_)[rr] + 0.5))
              {
                (*scalarvariancespectrum_np_)[rr] += scalarvariance;
              }
            }
          }
          else
            FOUR_C_THROW("Unknown forcing type!");
        }
      }
    }

    //--------------------------------------------------------
    // forcing factor from energy spectrum
    //--------------------------------------------------------

    for (int fftw_k_1 = 0; fftw_k_1 <= (nummodes_ - 1); fftw_k_1++)
    {
      for (int fftw_k_2 = 0; fftw_k_2 <= (nummodes_ - 1); fftw_k_2++)
      {
        for (int fftw_k_3 = 0; fftw_k_3 <= (nummodes_ / 2 + 1); fftw_k_3++)
        {
          int k_1 = -999;
          int k_2 = -999;
          int k_3 = -999;

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
            }
            else
            {
              // if negative wave vector is not in the construction domain
              // we have to shift it into the domain using the periodicity of the
              // wave number field
              // -k_3 always lies within the construction domain!
              if (k_1 < (-nummodes_ / 2)) k_1 += nummodes_;
              if (k_2 < (-nummodes_ / 2)) k_2 += nummodes_;
            }
          }

          if (forcing_type_ == Inpar::FLUID::linear_compensation_from_intermediate_spectrum)
          {
            // compute linear compensation factor from energy spectrum

            // following Hickel 2007
            //         (1/(2*E))* dE/dt if k <= k_t
            // C(k) =
            //         0                else
            // threshold wave number k_t
            // E intermediate energy spectrum, i.e. solution without forcing

            // get wave number
            double k = sqrt(k_1 * k_1 + k_2 * k_2 + k_3 * k_3);
            if (k <= threshold_wavenumber_ and k > 0)
            {
              int rr_k = 0;
              for (std::size_t rr = 0; rr < wavenumbers_->size(); rr++)
              {
                if ((*wavenumbers_)[rr] > k)
                {
                  rr_k = rr;
                  break;
                }
              }

              // interpolate spectrum to wave number
              double E_n = interpolate(k, (*wavenumbers_)[rr_k - 1], (*wavenumbers_)[rr_k],
                  (*scalarvariancespectrum_n_)[rr_k - 1], (*scalarvariancespectrum_n_)[rr_k]);
              double E_np = interpolate(k, (*wavenumbers_)[rr_k - 1], (*wavenumbers_)[rr_k],
                  (*scalarvariancespectrum_np_)[rr_k - 1], (*scalarvariancespectrum_np_)[rr_k]);

              // get position in fac-vector
              const int pos = fftw_k_3 + (nummodes_ / 2 + 1) * (fftw_k_2 + nummodes_ * fftw_k_1);

              // calculate C
              (*force_fac_)(pos) = (1.0 / (2.0 * E_np)) * (E_np - E_n) / dt_;
            }
          }
          else
            FOUR_C_THROW("Unknown forcing type!");
        }
      }
    }

    return;
#else
    FOUR_C_THROW("FFTW required");
#endif
  }


  /*--------------------------------------------------------------*
   | get forcing                                   rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbScalarForcing::update_forcing(const int step)
  {
#ifdef FOUR_C_WITH_FFTW
    // check if forcing is selected
    if (activate_)
    {
      //-------------------------------------------------------------------------------
      // calculate Fourier transformation of velocity
      //-------------------------------------------------------------------------------

      // set and initialize working arrays
      Teuchos::Array<std::complex<double>> phi_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
      Teuchos::Array<double> local_phi(nummodes_ * nummodes_ * nummodes_);
      Teuchos::Array<double> global_phi(nummodes_ * nummodes_ * nummodes_);

      //-----------------------------------
      // prepare Fourier transformation
      //-----------------------------------

      // set solution in local vectors for velocity

      for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
      {
        // get node
        Core::Nodes::Node* node = discret_->l_row_node(inode);

        // get coordinates
        Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);
        for (int idim = 0; idim < 3; idim++) xyz(idim, 0) = node->x()[idim];

        // get global ids of all dofs of the node
        std::vector<int> dofs = discret_->dof(0, node);

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

        // get position in velocity vectors local_u_1, local_u_2 and local_u_3
        const int pos = loc[2] + nummodes_ * (loc[1] + nummodes_ * loc[0]);

        // get local dof id corresponding to the global id
        int lid = discret_->dof_row_map()->lid(dofs[0]);
        // set value
        if (not is_genalpha_)
        {
          (local_phi)[pos] = phinp_->local_values_as_span()[lid];
        }
        else
        {
          (local_phi)[pos] = phiaf_->local_values_as_span()[lid];
        }
      }

      global_phi = Core::Communication::sum_all(local_phi, discret_->get_comm());

      //----------------------------------------
      // fast Fourier transformation using FFTW
      //----------------------------------------

      // note: this is not very efficient, since each
      // processor does the fft and there is no communication

#ifdef FOUR_C_WITH_FFTW
      // set-up
      fftw_plan fft = fftw_plan_dft_r2c_3d(nummodes_, nummodes_, nummodes_, global_phi.data(),
          (reinterpret_cast<fftw_complex*>(phi_hat.data())), FFTW_ESTIMATE);
      // fft
      fftw_execute(fft);
      // free memory
      fftw_destroy_plan(fft);
      fftw_cleanup();
#else
      FOUR_C_THROW("FFTW required for HIT!");
#endif

      // scale solution (not done in the fftw routine)
      for (int i = 0; i < phi_hat.size(); i++)
      {
        (phi_hat)[i] /= nummodes_ * nummodes_ * nummodes_;
      }

      //----------------------------------------
      // set forcing vector
      //----------------------------------------

      // Fourier coefficients of forcing
      Teuchos::Array<std::complex<double>> fphi_hat(nummodes_ * nummodes_ * (nummodes_ / 2 + 1));
      // where f_hat = -C(k) * u_hat according to Hickel 2007
      // C denotes a linear compensation factor

      Teuchos::Array<double> fphi(nummodes_ * nummodes_ * nummodes_);

      for (int rr = 0; rr < (nummodes_ * nummodes_ * (nummodes_ / 2 + 1)); rr++)
      {
        (fphi_hat)[rr] = -((*force_fac_)(rr)) * ((phi_hat)[rr]);
      }

      //----------------------------------------
      // fast Fourier transformation using FFTW
      //----------------------------------------

#ifdef FOUR_C_WITH_FFTW
      // setup
      fftw_plan fft_back = fftw_plan_dft_c2r_3d(nummodes_, nummodes_, nummodes_,
          (reinterpret_cast<fftw_complex*>(fphi_hat.data())), fphi.data(), FFTW_ESTIMATE);
      // fft
      fftw_execute(fft_back);
      // free memory
      fftw_destroy_plan(fft_back);
      fftw_cleanup();
#else
      FOUR_C_THROW("FFTW required for HIT!");
#endif

      //----------------------------------------
      // set force
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
        forcing_->replace_local_value(lid, fphi[pos]);
      }
    }
    else
      // set force to zero
      forcing_->put_scalar(0.0);

    return;
#else
    FOUR_C_THROW("FFTW required");
#endif
  }


  /*--------------------------------------------------------------*
   | time update of energy spectrum               rasthofer 04/13 |
   *--------------------------------------------------------------*/
  void HomoIsoTurbScalarForcing::time_update_forcing()
  {
#ifdef TIME_UPDATE_FORCING_SPECTRUM
    // update energy spectrum
    for (std::size_t rr = 0; rr < scalarvariancespectrum_n_->size(); rr++)
    {
      // compute E^n+1 from forced solution
      CalculateForcing(0);

      (*scalarvariancespectrum_n_)[rr] = (*scalarvariancespectrum_np_)[rr];
      (*scalarvariancespectrum_np_)[rr] = 0.0;
    }
#endif

    // reset to zero
    for (int rr = 0; rr < (nummodes_ * nummodes_ * (nummodes_ / 2 + 1)); rr++)
      (*force_fac_)(rr) = 0.0;

    forcing_->put_scalar(0.0);

    return;
  }


}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE
