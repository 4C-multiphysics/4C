// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_statistics_ph.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"

#include <Teuchos_ParameterList.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN

#define SAMP_ALL

//----------------------------------------------------------------------
// constructor
//----------------------------------------------------------------------
FLD::TurbulenceStatisticsPh::TurbulenceStatisticsPh(
    std::shared_ptr<Core::FE::Discretization> actdis, Teuchos::ParameterList& params,
    const std::string& statistics_outfilename)
    : discret_(actdis), params_(params), statistics_outfilename_(statistics_outfilename)
{
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::cout << "This is the turbulence statistics manager of periodic hill problem" << std::endl;
    std::cout << "based on the geometry of ERCOFTAC" << std::endl;
  }

  //----------------------------------------------------------------------
  // plausibility check
  int numdim = params_.get<int>("number of velocity degrees of freedom");
  if (numdim != 3) FOUR_C_THROW("Evaluation of turbulence statistics only for 3d flow problems!");


  //----------------------------------------------------------------------
  // allocate some (toggle) vectors
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  squaredvelnp_ = Core::LinAlg::create_vector(*dofrowmap, true);

  toggleu_ = Core::LinAlg::create_vector(*dofrowmap, true);
  togglev_ = Core::LinAlg::create_vector(*dofrowmap, true);
  togglew_ = Core::LinAlg::create_vector(*dofrowmap, true);
  togglep_ = Core::LinAlg::create_vector(*dofrowmap, true);

  //----------------------------------------------------------------------
  // create sets of coordinates
  //----------------------------------------------------------------------
  x1coordinates_ = std::make_shared<std::vector<double>>();
  x2coordinates_ = std::make_shared<std::vector<double>>();

  // the criterion allows differences in coordinates by 1e-9
  std::set<double, LineSortCriterion> x1avcoords;
  std::set<double, LineSortCriterion> x2avcoords;
  std::set<double, LineSortCriterion> x2loccc;
  std::set<double, LineSortCriterion> x2statlocat;


  for (int i = 0; i < discret_->num_my_row_nodes(); ++i)
  {
    Core::Nodes::Node* node = discret_->l_row_node(i);

    if (node->x()[1] < 85.008 + 2e-9 && node->x()[1] > 85.008 - 2e-9)
      x1avcoords.insert(node->x()[0]);

    if (node->x()[0] < 2e-9 && node->x()[0] > -2e-9) x2avcoords.insert(node->x()[1]);
  }

  //--------------------------------------------------------------------
  // round robin loop to communicate coordinates to all procs
  //--------------------------------------------------------------------
  {
    int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());
    int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

    std::vector<char> sblock;
    std::vector<char> rblock;

    // create an exporter for point to point communication
    Core::Communication::Exporter exporter(discret_->get_comm());

    // first, communicate coordinates in x1-direction
    for (int np = 0; np < numprocs; ++np)
    {
      Core::Communication::PackBuffer data;

      for (std::set<double, LineSortCriterion>::iterator x1line = x1avcoords.begin();
          x1line != x1avcoords.end(); ++x1line)
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

      //--------------------------------------------------
      // Unpack received block into set of all planes.
      {
        std::vector<double> coordsvec;

        coordsvec.clear();

        Core::Communication::UnpackBuffer buffer(rblock);
        while (!buffer.at_end())
        {
          double onecoord;
          extract_from_pack(buffer, onecoord);
          x1avcoords.insert(onecoord);
        }
      }
    }

    // second, communicate coordinates in x2-direction
    for (int np = 0; np < numprocs; ++np)
    {
      Core::Communication::PackBuffer data;
      for (std::set<double, LineSortCriterion>::iterator x2line = x2avcoords.begin();
          x2line != x2avcoords.end(); ++x2line)
      {
        add_to_pack(data, *x2line);
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

      //--------------------------------------------------
      // Unpack received block into set of all planes.
      {
        std::vector<double> coordsvec;

        coordsvec.clear();

        Core::Communication::UnpackBuffer buffer(rblock);
        while (!buffer.at_end())
        {
          double onecoord;
          extract_from_pack(buffer, onecoord);
          x2avcoords.insert(onecoord);
        }
      }
    }
  }
  //----------------------------------------------------------------------
  // push coordinates in vectors
  //----------------------------------------------------------------------
  {
    for (std::set<double, LineSortCriterion>::iterator coord1 = x1avcoords.begin();
        coord1 != x1avcoords.end(); ++coord1)
    {
      x1coordinates_->push_back(*coord1);
    }

    for (std::set<double, LineSortCriterion>::iterator coord2 = x2avcoords.begin();
        coord2 != x2avcoords.end(); ++coord2)
    {
      x2coordinates_->push_back(*coord2);
    }
  }
  //----------------------------------------------------------------------
  // number of coordinates in x1- and x2-direction
  //----------------------------------------------------------------------
  numx1coor_ = x1coordinates_->size();
  numx2coor_ = x2coordinates_->size();

  //----------------------------------------------------------------------
  // define locations in x1-direction for statistical evaluation
  //----------------------------------------------------------------------

#ifndef SAMP_ALL
  // define vector containing x1-coords for sampling

  x1setstatlocations_ = std::shared_ptr(new std::vector<double>);

  //! x1-coordinates for statistical sampling  !!! in sampling routine the offwall distance has to
  //! be modified for different statlocations
  x1setstatlocations_->push_back(0);
  x1setstatlocations_->push_back(14);
  x1setstatlocations_->push_back(28);
  x1setstatlocations_->push_back(56);
  x1setstatlocations_->push_back(84);
  x1setstatlocations_->push_back(112);
  x1setstatlocations_->push_back(140);
  x1setstatlocations_->push_back(168);
  x1setstatlocations_->push_back(196);
  x1setstatlocations_->push_back(224);
//
#endif
#ifdef SAMP_ALL
  x1setstatlocations_ = x1coordinates_;
#endif
  numx1statlocations_ = x1setstatlocations_->size();

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::cout << "Sample at " << numx1statlocations_ << " numx1statlocations_ \n" << std::endl;
  }

  // Resize matrix for x1statlocations and x2statlocations coordinates
  x1statlocations_.reshape(numx1statlocations_, 1);
  x2statlocations_.reshape(numx1statlocations_, numx2coor_);

#ifndef SAMP_ALL
  x1elemlengthhalf_ = (x1max_ - x1min_) / (numx1coor_ - 1);
  for (int x1line = 0; x1line < numx1statlocations_; ++x1line)
  {
    int pos = 0;
    mindist_ = x1elemlengthhalf_;
    for (int x1linecoords = 0; x1linecoords < numx1coor_; ++x1linecoords)
    {
      dist_ = abs(x1setstatlocations_->at(x1line) - x1coordinates_->at(x1linecoords));
      if (dist_ < mindist_)
      {
        mindist_ = dist_;
        pos = x1linecoords;
      }
    }
    x1statlocations(x1line, 0) = x1coordinates_->at(pos);
  }
#endif
#ifdef SAMP_ALL
  for (int x1line = 0; x1line < numx1statlocations_; ++x1line)
  {
    x1statlocations_(x1line, 0) = x1setstatlocations_->at(x1line);
  }
#endif

  //----------------------------------------------------------------------
  // define locations in x2-direction for statistical evaluation
  //----------------------------------------------------------------------
  // define vector containing x2-coords for sampling

  for (int x1stat = 0; x1stat < numx1statlocations_; ++x1stat)
  {
    x2statlocat.clear();

    for (int i = 0; i < discret_->num_my_row_nodes(); ++i)
    {
      Core::Nodes::Node* node = discret_->l_row_node(i);
      if (node->x()[2] < +2e-9 && node->x()[2] > -2e-9 &&
          node->x()[0] < x1statlocations_(x1stat, 0) + 2e-9 &&
          node->x()[0] > x1statlocations_(x1stat, 0) - 2e-9)
      {
        x2statlocat.insert(node->x()[1]);
      }
    }

    //////////////////////////////////////////////
    //  communication of x2-coordinates
    //////////////////////////////////////////////
    {
      int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());
      int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

      std::vector<char> sblock;
      std::vector<char> rblock;

      // create an exporter for point to point communication
      Core::Communication::Exporter exporter(discret_->get_comm());

      // first, communicate coordinates in x1-direction
      for (int np = 0; np < numprocs; ++np)
      {
        Core::Communication::PackBuffer data;
        for (std::set<double, LineSortCriterion>::iterator x2 = x2statlocat.begin();
            x2 != x2statlocat.end(); ++x2)
        {
          add_to_pack(data, *x2);
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

        //--------------------------------------------------
        // Unpack received block into set of all planes.
        {
          Core::Communication::UnpackBuffer buffer(rblock);
          while (!buffer.at_end())
          {
            double onecoord;
            extract_from_pack(buffer, onecoord);
            x2statlocat.insert(onecoord);
          }
        }
      }
    }

    int x2it = -1;
    for (std::set<double, LineSortCriterion>::iterator x2locc = x2statlocat.begin();
        x2locc != x2statlocat.end(); ++x2locc)
    {
      x2it += 1;
      x2statlocations_(x1stat, x2it) = *x2locc;
    }
  }

  ///////////////////////////////////////////
  //    end of communication in x2-direction
  ///////////////////////////////////////////

  //----------------------------------------------------------------------
  // allocate arrays for sums of mean values
  //----------------------------------------------------------------------

  // x1-direction
  x1sump_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x1sump_->reshape(numx1statlocations_, 2);

  x1sumu_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x1sumu_->reshape(numx1statlocations_, 1);

  x1sumf_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x1sumf_->reshape(numx1statlocations_, 3);


  // x2-direction
  // first-order moments
  x2sumu_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumu_->reshape(numx1statlocations_, numx2coor_);

  x2sumv_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumv_->reshape(numx1statlocations_, numx2coor_);

  x2sumw_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumw_->reshape(numx1statlocations_, numx2coor_);

  x2sump_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sump_->reshape(numx1statlocations_, numx2coor_);

  // second-order moments
  x2sumsqu_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumsqu_->reshape(numx1statlocations_, numx2coor_);

  x2sumsqv_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumsqv_->reshape(numx1statlocations_, numx2coor_);

  x2sumsqw_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumsqw_->reshape(numx1statlocations_, numx2coor_);

  x2sumsqp_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumsqp_->reshape(numx1statlocations_, numx2coor_);

  x2sumuv_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumuv_->reshape(numx1statlocations_, numx2coor_);

  x2sumuw_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumuw_->reshape(numx1statlocations_, numx2coor_);

  x2sumvw_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
  x2sumvw_->reshape(numx1statlocations_, numx2coor_);

  // set number of samples to zero
  numsamp_ = 0;

  //----------------------------------------------------------------------
  // define homogeneous direction to compute averages of Smagorinsky constant

  Teuchos::ParameterList* modelparams = &(params_.sublist("TURBULENCE MODEL"));
  // check if we want to compute averages of Smagorinsky constant
  if (modelparams->get<std::string>("PHYSICAL_MODEL", "no_model") == "Dynamic_Smagorinsky")
  {
    // store them in parameterlist for access on the element
    modelparams->set<std::shared_ptr<std::vector<double>>>("dir1coords_", x1coordinates_);
    modelparams->set<std::shared_ptr<std::vector<double>>>("dir2coords_", x2coordinates_);
  }

  //----------------------------------------------------------------------
  // initialize output and initially open respective statistics output file

  std::shared_ptr<std::ofstream> log;

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::string s(statistics_outfilename_);

    s.append(".flow_statistics");

    log = std::make_shared<std::ofstream>(s.c_str(), std::ios::out);
    (*log) << "# Statistics for turbulent incompressible flow over periodic hill (first- and "
              "second-order moments)\n\n";

    log->flush();
  }

  return;
}  // TurbulenceStatisticsPh::TurbulenceStatisticsPh



//----------------------------------------------------------------------
// sampling of velocity/pressure values
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsPh::do_time_sample(
    Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& stresses)
{
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    std::cout << "------------Time Sampling Routine begins---------" << std::endl;


  // compute squared values of velocity
  squaredvelnp_->multiply(1.0, velnp, velnp, 0.0);

  //----------------------------------------------------------------------
  // increase sample counter
  //----------------------------------------------------------------------
  numsamp_++;

#ifdef SAMP_ALL

  for (int k = 0; k < numx1statlocations_; ++k)
  {
    // current x2-coordinate of respective wall
    // constant offset from wall at the middle bottom; has to be adapted for different statlocations
    double x2cwall = x2statlocations_(k, 1);
    double x1cwall = x1statlocations_(k, 0);
    double x2cwall_puw = x2statlocations_(k, numx2coor_ - 1);
    double x2cwall_plw = x2statlocations_(k, 0);

    // toggle vectors are one in the position of a dof of this node,
    toggleu_->put_scalar(0.0);

    // du1/dx
    {
      // count the number of nodes in x3-direction contributing to this nodal value
      int countnodes = 0;

      for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
      {
        Core::Nodes::Node* node = discret_->l_row_node(nn);
        // this is the wall node
        if (node->x()[0] < (x1cwall + 2e-9) and node->x()[0] > (x1cwall - 2e-9) and
            node->x()[1] < (x2cwall + 2e-9) and node->x()[1] > (x2cwall - 2e-9))
        {
          std::vector<int> dof = discret_->dof(node);
          double one = 1.0;

          toggleu_->replace_global_values(1, &one, dof.data());
          countnodes++;
        }
      }
      int countnodesonallprocs = 0;

      countnodesonallprocs = Core::Communication::sum_all(countnodes, discret_->get_comm());

      // reduce by 1 due to periodic boundary condition
      countnodesonallprocs -= 1;
      if (countnodesonallprocs)
      {
        //----------------------------------------------------------------------
        // get values for velocity derivative and pressure
        //----------------------------------------------------------------------
        double u;
        velnp.dot(*toggleu_, &u);

        //----------------------------------------------------------------------
        // calculate spatial means
        //----------------------------------------------------------------------
        double usm = u / countnodesonallprocs;
        //----------------------------------------------------------------------
        // add spatial mean values to statistical sample
        //----------------------------------------------------------------------
        (*x1sumu_)(k, 0) += usm;
      }
    }

    togglep_->put_scalar(0.0);
    // p, lower wall
    {
      // count the number of nodes in x3-direction contributing to this nodal value
      int countnodes = 0;

      for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
      {
        Core::Nodes::Node* node = discret_->l_row_node(nn);
        // this is the wall node
        if (node->x()[0] < (x1cwall + 2e-9) and node->x()[0] > (x1cwall - 2e-9) and
            node->x()[1] < (x2cwall_plw + 2e-9) and node->x()[1] > (x2cwall_plw - 2e-9))
        {
          std::vector<int> dof = discret_->dof(node);
          double one = 1.0;

          togglep_->replace_global_values(1, &one, &(dof[3]));
          countnodes++;
        }
      }
      int countnodesonallprocs = 0;

      countnodesonallprocs = Core::Communication::sum_all(countnodes, discret_->get_comm());

      // reduce by 1 due to periodic boundary condition
      countnodesonallprocs -= 1;
      if (countnodesonallprocs)
      {
        //----------------------------------------------------------------------
        // get values for velocity derivative and pressure
        //----------------------------------------------------------------------
        double p;
        velnp.dot(*togglep_, &p);

        //----------------------------------------------------------------------
        // calculate spatial means
        //----------------------------------------------------------------------
        double lwpsm = p / countnodesonallprocs;
        //----------------------------------------------------------------------
        // add spatial mean values to statistical sample
        //----------------------------------------------------------------------
        (*x1sump_)(k, 0) += lwpsm;
      }
    }

    togglep_->put_scalar(0.0);
    // p, upper wall
    {
      // count the number of nodes in x3-direction contributing to this nodal value
      int countnodes = 0;

      for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
      {
        Core::Nodes::Node* node = discret_->l_row_node(nn);
        // this is the wall node
        if (node->x()[0] < (x1cwall + 2e-9) and node->x()[0] > (x1cwall - 2e-9) and
            node->x()[1] < (x2cwall_puw + 2e-9) and node->x()[1] > (x2cwall_puw - 2e-9))
        {
          std::vector<int> dof = discret_->dof(node);
          double one = 1.0;

          togglep_->replace_global_values(1, &one, &(dof[3]));
          countnodes++;
        }
      }
      int countnodesonallprocs = 0;

      countnodesonallprocs = Core::Communication::sum_all(countnodes, discret_->get_comm());

      // reduce by 1 due to periodic boundary condition
      countnodesonallprocs -= 1;
      if (countnodesonallprocs)
      {
        //----------------------------------------------------------------------
        // get values for velocity derivative and pressure
        //----------------------------------------------------------------------
        double p;
        velnp.dot(*togglep_, &p);

        //----------------------------------------------------------------------
        // calculate spatial means
        //----------------------------------------------------------------------
        double uwpsm = p / countnodesonallprocs;
        //----------------------------------------------------------------------
        // add spatial mean values to statistical sample
        //----------------------------------------------------------------------
        (*x1sump_)(k, 1) += uwpsm;
      }
    }
  }

  // wall stresses
  for (int k = 0; k < numx1statlocations_; ++k)
  {
    // current x2-coordinate of respective wall
    // constant offset from wall at the middle bottom; has to be adapted for different statlocations
    double x2cwall = x2statlocations_(k, 0);
    double x1cwall = x1statlocations_(k, 0);

    // toggle vectors are one in the position of a dof of this node,
    // else 0
    toggleu_->put_scalar(0.0);
    togglev_->put_scalar(0.0);
    togglew_->put_scalar(0.0);

    // count the number of nodes in x3-direction contributing to this nodal value
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);
      // this is the wall node
      if (node->x()[0] < (x1cwall + 2e-9) and node->x()[0] > (x1cwall - 2e-9) and
          node->x()[1] < (x2cwall + 2e-9) and node->x()[1] > (x2cwall - 2e-9))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->replace_global_values(1, &one, &(dof[0]));
        togglev_->replace_global_values(1, &one, &(dof[1]));
        togglew_->replace_global_values(1, &one, &(dof[2]));
        countnodes++;
      }
    }
    int countnodesonallprocs = 0;

    countnodesonallprocs = Core::Communication::sum_all(countnodes, discret_->get_comm());

    // reduce by 1 due to periodic boundary condition
    countnodesonallprocs -= 1;
    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity derivative and pressure
      //----------------------------------------------------------------------
      double fx;
      stresses.dot(*toggleu_, &fx);
      double fy;
      stresses.dot(*togglev_, &fy);
      double fz;
      stresses.dot(*togglew_, &fz);

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x1sumf_)(k, 0) += fx / (double)countnodesonallprocs;
      (*x1sumf_)(k, 1) += fy / (double)countnodesonallprocs;
      (*x1sumf_)(k, 2) += fz / (double)countnodesonallprocs;
    }
  }

  //----------------------------------------------------------------------
  // end of loop velocity gradient at lower wall
  //----------------------------------------------------------------------
#endif

  //----------------------------------------------------------------------
  // loop locations for statistical evaluation in x1-direction
  //----------------------------------------------------------------------

  for (int x1nodnum = 0; x1nodnum < numx1statlocations_; ++x1nodnum)
  {
    //----------------------------------------------------------------------
    // loop nodes in x2-direction
    //----------------------------------------------------------------------
    for (int x2line = 0; x2line < numx2coor_; ++x2line)
    {
      // toggle vectors are one in the position of a dof of this node,
      // else 0
      toggleu_->put_scalar(0.0);
      togglev_->put_scalar(0.0);
      togglew_->put_scalar(0.0);
      togglep_->put_scalar(0.0);

      // count the number of nodes in x3-direction contributing to this nodal value
      int countnodes = 0;

      for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
      {
        Core::Nodes::Node* node = discret_->l_row_node(nn);

        // this is the node
        if (node->x()[0] < x1statlocations_(x1nodnum, 0) + 2e-5 &&
            node->x()[0] > x1statlocations_(x1nodnum, 0) - 2e-5 &&
            node->x()[1] < x2statlocations_(x1nodnum, x2line) + 2e-9 &&
            node->x()[1] > x2statlocations_(x1nodnum, x2line) -
                               2e-9)  // the last node in x3 direction has to be sampled
        {
          std::vector<int> dof = discret_->dof(node);
          double one = 1.0;

          toggleu_->replace_global_values(1, &one, &(dof[0]));
          togglev_->replace_global_values(1, &one, &(dof[1]));
          togglew_->replace_global_values(1, &one, &(dof[2]));
          togglep_->replace_global_values(1, &one, &(dof[3]));

          countnodes++;
        }
      }

      int countnodesonallprocs = 0;

      countnodesonallprocs = Core::Communication::sum_all(countnodes, discret_->get_comm());

      // reduce by 1 due to periodic boundary condition
      countnodesonallprocs -= 1;

      if (countnodesonallprocs)
      {
        //----------------------------------------------------------------------
        // get values for velocity and pressure on this line
        //----------------------------------------------------------------------
        double u;
        double v;
        double w;
        double p;
        velnp.dot(*toggleu_, &u);
        velnp.dot(*togglev_, &v);
        velnp.dot(*togglew_, &w);
        velnp.dot(*togglep_, &p);

        double uu;
        double vv;
        double ww;
        double pp;
        squaredvelnp_->dot(*toggleu_, &uu);
        squaredvelnp_->dot(*togglev_, &vv);
        squaredvelnp_->dot(*togglew_, &ww);
        squaredvelnp_->dot(*togglep_, &pp);

        double uv;
        double uw;
        double vw;
        double locuv = 0.0;
        double locuw = 0.0;
        double locvw = 0.0;
        for (int rr = 1; rr < velnp.local_length(); ++rr)
        {
          locuv += velnp.local_values_as_span()[rr - 1] * toggleu_->local_values_as_span()[rr - 1] *
                   velnp.local_values_as_span()[rr] * togglev_->local_values_as_span()[rr];
        }
        uv = Core::Communication::sum_all(locuv, discret_->get_comm());
        for (int rr = 2; rr < velnp.local_length(); ++rr)
        {
          locuw += velnp.local_values_as_span()[rr - 2] * toggleu_->local_values_as_span()[rr - 2] *
                   velnp.local_values_as_span()[rr] * togglew_->local_values_as_span()[rr];
        }
        uw = Core::Communication::sum_all(locuw, discret_->get_comm());
        for (int rr = 2; rr < velnp.local_length(); ++rr)
        {
          locvw += velnp.local_values_as_span()[rr - 1] * togglev_->local_values_as_span()[rr - 1] *
                   velnp.local_values_as_span()[rr] * togglew_->local_values_as_span()[rr];
        }
        vw = Core::Communication::sum_all(locvw, discret_->get_comm());

        //----------------------------------------------------------------------
        // calculate spatial means on this line
        // add spatial mean values to statistical sample
        //----------------------------------------------------------------------
        (*x2sumu_)(x1nodnum, x2line) += u / countnodesonallprocs;
        (*x2sumv_)(x1nodnum, x2line) += v / countnodesonallprocs;
        (*x2sumw_)(x1nodnum, x2line) += w / countnodesonallprocs;
        (*x2sump_)(x1nodnum, x2line) += p / countnodesonallprocs;

        (*x2sumsqu_)(x1nodnum, x2line) += uu / countnodesonallprocs;
        (*x2sumsqv_)(x1nodnum, x2line) += vv / countnodesonallprocs;
        (*x2sumsqw_)(x1nodnum, x2line) += ww / countnodesonallprocs;
        (*x2sumsqp_)(x1nodnum, x2line) += pp / countnodesonallprocs;

        (*x2sumuv_)(x1nodnum, x2line) += uv / countnodesonallprocs;
        (*x2sumuw_)(x1nodnum, x2line) += uw / countnodesonallprocs;
        (*x2sumvw_)(x1nodnum, x2line) += vw / countnodesonallprocs;
      }
    }
  }

  return;
}  // TurbulenceStatisticsPh::DoTimeSample


/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsPh::dump_statistics(int step)
{
  //----------------------------------------------------------------------
  // output to log-file
  std::shared_ptr<std::ofstream> log;

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::string s(statistics_outfilename_);
    s.append(".flow_statistics");

    log = std::make_shared<std::ofstream>(s.c_str(), std::ios::out);
    (*log) << "# Statistics for turbulent incompressible flow for a periodic hill (first- and "
              "second-order moments)";
    (*log) << "\n\n";
    (*log) << "# Statistics record ";
    (*log) << " (Steps " << step - numsamp_ + 1 << "--" << step << ")\n\n\n";
    (*log) << std::scientific;
    (*log) << "\n\n\n";

#ifdef SAMP_ALL
    (*log) << "# lower wall behind step\n";
    (*log) << "#     x1";
    (*log) << "           duxdy       pmean (lw)    pmean (uw)     tau_w          dist\n";
    // distance from wall to first node off wall
    for (int i = 0; i < numx1statlocations_; ++i)
    {
      double lwx1u = (*x1sumu_)(i, 0) / numsamp_;
      double dist = x2statlocations_(i, 1) - x2statlocations_(i, 0);
      double lwx1duxdy = lwx1u / dist;
      double lwx1p = (*x1sump_)(i, 0) / numsamp_;
      double uwx1p = (*x1sump_)(i, 1) / numsamp_;
      double fx = (*x1sumf_)(i, 0) / numsamp_;
      double fy = (*x1sumf_)(i, 1) / numsamp_;
      double fz = (*x1sumf_)(i, 2) / numsamp_;
      double f = std::sqrt(fx * fx + fy * fy + fz * fz);
      if (fx < 0.0) f *= -1.0;  // recover sign
      (*log) << " " << std::setw(11) << std::setprecision(4) << x1statlocations_(i, 0);
      (*log) << "   " << std::setw(11) << std::setprecision(4) << lwx1duxdy;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << lwx1p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << uwx1p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << f;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << dist;
      (*log) << "\n";
    }
#endif
    for (int i = 0; i < numx1statlocations_; ++i)
    {
      // current x1-coordinate
      double x1 = 1.0e20;
      x1 = x1statlocations_(i, 0);

      (*log) << "\n\n\n";
      (*log) << "# line in x2-direction at x1 = " << std::setw(11) << std::setprecision(4) << x1
             << "\n";
      (*log) << "#     x2";
      (*log) << "           umean         vmean         wmean         pmean";
      (*log) << "         urms          vrms          wrms          prms";
      (*log) << "          u'v'          u'w'          v'w'\n";
      for (unsigned j = 0; j < x2coordinates_->size(); ++j)
      {
        double x2u = (*x2sumu_)(i, j) / numsamp_;
        double x2v = (*x2sumv_)(i, j) / numsamp_;
        double x2w = (*x2sumw_)(i, j) / numsamp_;
        double x2p = (*x2sump_)(i, j) / numsamp_;
        double x2urms = 0.0;
        double x2vrms = 0.0;
        double x2wrms = 0.0;
        double x2prms = 0.0;
        if (((*x2sumsqu_)(i, j) / numsamp_ - x2u * x2u) > 0.0)
          x2urms = std::sqrt((*x2sumsqu_)(i, j) / numsamp_ - x2u * x2u);
        if (((*x2sumsqv_)(i, j) / numsamp_ - x2v * x2v) > 0.0)
          x2vrms = std::sqrt((*x2sumsqv_)(i, j) / numsamp_ - x2v * x2v);
        if (((*x2sumsqw_)(i, j) / numsamp_ - x2w * x2w) > 0.0)
          x2wrms = std::sqrt((*x2sumsqw_)(i, j) / numsamp_ - x2w * x2w);
        if (((*x2sumsqp_)(i, j) / numsamp_ - x2p * x2p) > 0.0)
          x2prms = std::sqrt((*x2sumsqp_)(i, j) / numsamp_ - x2p * x2p);
        double x2uv = (*x2sumuv_)(i, j) / numsamp_ - x2u * x2v;
        double x2uw = (*x2sumuw_)(i, j) / numsamp_ - x2u * x2w;
        double x2vw = (*x2sumvw_)(i, j) / numsamp_ - x2v * x2w;
        (*log) << " " << std::setw(11) << std::setprecision(4) << x2statlocations_(i, j);
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2u;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2v;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2w;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2p;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2urms;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vrms;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2wrms;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2prms;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uv;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uw;
        (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vw;
        (*log) << "\n";
      }
    }
    log->flush();
  }

  return;

}  // TurbulenceStatisticsPh::DumpStatistics

FOUR_C_NAMESPACE_CLOSE
