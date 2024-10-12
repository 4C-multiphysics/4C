/*----------------------------------------------------------------------*/
/*! \file

\brief calculate mean values and fluctuations for turbulent flow in a
lid-driven cavity.


\level 2

*/
/*----------------------------------------------------------------------*/
#include "4C_fluid_turbulence_statistics_ldc.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*!
  \brief Standard Constructor (public)

    o Create sets for centerlines in x1- and x2-direction

  o Allocate distributed vector for squares

*/
/*----------------------------------------------------------------------*/
FLD::TurbulenceStatisticsLdc::TurbulenceStatisticsLdc(Teuchos::RCP<Core::FE::Discretization> actdis,
    Teuchos::ParameterList& params, const std::string& statistics_outfilename)
    : discret_(actdis), params_(params), statistics_outfilename_(statistics_outfilename)
{
  //----------------------------------------------------------------------
  // plausibility check
  int numdim = params_.get<int>("number of velocity degrees of freedom");
  if (numdim != 3) FOUR_C_THROW("Evaluation of turbulence statistics only for 3d flow problems!");

  const auto physicaltype =
      Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(params_, "Physical Type");

  //----------------------------------------------------------------------
  // allocate some (toggle) vectors
  const Epetra_Map* dofrowmap = discret_->dof_row_map();

  toggleu_ = Core::LinAlg::create_vector(*dofrowmap, true);
  togglev_ = Core::LinAlg::create_vector(*dofrowmap, true);
  togglew_ = Core::LinAlg::create_vector(*dofrowmap, true);
  togglep_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // bounds for extension of cavity in x1-direction
  x1min_ = +10e+19;
  x1max_ = -10e+19;
  // bounds for extension of cavity in x2-direction
  x2min_ = +10e+19;
  x2max_ = -10e+19;
  // bounds for extension of cavity in x3-direction
  x3min_ = +10e+19;
  x3max_ = -10e+19;

  //----------------------------------------------------------------------
  // create sets of coordinates for centerlines in x1-, x2- and x3-direction
  //----------------------------------------------------------------------
  x1coordinates_ = Teuchos::make_rcp<std::vector<double>>();
  x2coordinates_ = Teuchos::make_rcp<std::vector<double>>();
  x3coordinates_ = Teuchos::make_rcp<std::vector<double>>();

  // the criterion allows differences in coordinates by 1e-9
  std::set<double, LineSortCriterion> x1avcoords;
  std::set<double, LineSortCriterion> x2avcoords;
  std::set<double, LineSortCriterion> x3avcoords;

  // loop nodes, build sets of centerlines accessible on this proc
  for (int i = 0; i < discret_->num_my_row_nodes(); ++i)
  {
    Core::Nodes::Node* node = discret_->l_row_node(i);
    x1avcoords.insert(node->x()[0]);
    x2avcoords.insert(node->x()[1]);
    x3avcoords.insert(node->x()[2]);

    if (x1min_ > node->x()[0])
    {
      x1min_ = node->x()[0];
    }
    if (x1max_ < node->x()[0])
    {
      x1max_ = node->x()[0];
    }
    if (x2min_ > node->x()[1])
    {
      x2min_ = node->x()[1];
    }
    if (x2max_ < node->x()[1])
    {
      x2max_ = node->x()[1];
    }
    if (x3min_ > node->x()[2])
    {
      x3min_ = node->x()[2];
    }
    if (x3max_ < node->x()[2])
    {
      x3max_ = node->x()[2];
    }
  }

  // communicate x1mins and x1maxs
  double min1;
  discret_->get_comm().MinAll(&x1min_, &min1, 1);
  x1min_ = min1;

  double max1;
  discret_->get_comm().MaxAll(&x1max_, &max1, 1);
  x1max_ = max1;

  // communicate x2mins and x2maxs
  double min2;
  discret_->get_comm().MinAll(&x2min_, &min2, 1);
  x2min_ = min2;

  double max2;
  discret_->get_comm().MaxAll(&x2max_, &max2, 1);
  x2max_ = max2;

  // communicate x3mins and x3maxs
  double min3;
  discret_->get_comm().MinAll(&x3min_, &min3, 1);
  x3min_ = min3;

  double max3;
  discret_->get_comm().MaxAll(&x3max_, &max3, 1);
  x3max_ = max3;

  //--------------------------------------------------------------------
  // round robin loop to communicate coordinates in x1-, x2- and x3-
  // direction to all procs
  //--------------------------------------------------------------------
  {
    int myrank = discret_->get_comm().MyPID();
    int numprocs = discret_->get_comm().NumProc();

    std::vector<char> sblock;
    std::vector<char> rblock;

    // create an exporter for point to point comunication
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
      swap(sblock, data());

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
        exporter.get_comm().Barrier();
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
      swap(sblock, data());

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
        exporter.get_comm().Barrier();
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

    // third, communicate coordinates in x3-direction
    for (int np = 0; np < numprocs; ++np)
    {
      Core::Communication::PackBuffer data;

      for (std::set<double, LineSortCriterion>::iterator x3line = x3avcoords.begin();
           x3line != x3avcoords.end(); ++x3line)
      {
        add_to_pack(data, *x3line);
      }
      swap(sblock, data());

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
        exporter.get_comm().Barrier();
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
          x3avcoords.insert(onecoord);
        }
      }
    }
  }

  //----------------------------------------------------------------------
  // push coordinates in x1-, x2- and x3-direction in a vector
  //----------------------------------------------------------------------
  {
    x1coordinates_ = Teuchos::make_rcp<std::vector<double>>();
    x2coordinates_ = Teuchos::make_rcp<std::vector<double>>();
    x3coordinates_ = Teuchos::make_rcp<std::vector<double>>();

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

    for (std::set<double, LineSortCriterion>::iterator coord3 = x3avcoords.begin();
         coord3 != x3avcoords.end(); ++coord3)
    {
      x3coordinates_->push_back(*coord3);
    }
  }

  //----------------------------------------------------------------------
  // allocate arrays for sums of mean values
  //----------------------------------------------------------------------
  int size1 = x1coordinates_->size();
  int size2 = x2coordinates_->size();
  int size3 = x3coordinates_->size();

  // first-order moments
  x1sumu_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumu_->resize(size1, 0.0);
  x2sumu_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumu_->resize(size2, 0.0);
  x3sumu_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumu_->resize(size3, 0.0);

  x1sumv_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumv_->resize(size1, 0.0);
  x2sumv_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumv_->resize(size2, 0.0);
  x3sumv_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumv_->resize(size3, 0.0);

  x1sumw_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumw_->resize(size1, 0.0);
  x2sumw_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumw_->resize(size2, 0.0);
  x3sumw_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumw_->resize(size3, 0.0);

  x1sump_ = Teuchos::make_rcp<std::vector<double>>();
  x1sump_->resize(size1, 0.0);
  x2sump_ = Teuchos::make_rcp<std::vector<double>>();
  x2sump_->resize(size2, 0.0);
  x3sump_ = Teuchos::make_rcp<std::vector<double>>();
  x3sump_->resize(size3, 0.0);

  // second-order moments
  x1sumsqu_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumsqu_->resize(size1, 0.0);
  x2sumsqu_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumsqu_->resize(size2, 0.0);
  x3sumsqu_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumsqu_->resize(size3, 0.0);

  x1sumsqv_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumsqv_->resize(size1, 0.0);
  x2sumsqv_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumsqv_->resize(size2, 0.0);
  x3sumsqv_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumsqv_->resize(size3, 0.0);

  x1sumsqw_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumsqw_->resize(size1, 0.0);
  x2sumsqw_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumsqw_->resize(size2, 0.0);
  x3sumsqw_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumsqw_->resize(size3, 0.0);

  x1sumsqp_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumsqp_->resize(size1, 0.0);
  x2sumsqp_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumsqp_->resize(size2, 0.0);
  x3sumsqp_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumsqp_->resize(size3, 0.0);

  x1sumuv_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumuv_->resize(size1, 0.0);
  x2sumuv_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumuv_->resize(size2, 0.0);
  x3sumuv_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumuv_->resize(size3, 0.0);

  x1sumuw_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumuw_->resize(size1, 0.0);
  x2sumuw_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumuw_->resize(size2, 0.0);
  x3sumuw_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumuw_->resize(size3, 0.0);

  x1sumvw_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumvw_->resize(size1, 0.0);
  x2sumvw_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumvw_->resize(size2, 0.0);
  x3sumvw_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumvw_->resize(size3, 0.0);

  // the following vectors are only necessary for low-Mach-number flow
  // first-order moments
  x1sumrho_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumrho_->resize(size1, 0.0);
  x2sumrho_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumrho_->resize(size2, 0.0);
  x3sumrho_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumrho_->resize(size3, 0.0);

  x1sum_t_ = Teuchos::make_rcp<std::vector<double>>();
  x1sum_t_->resize(size1, 0.0);
  x2sum_t_ = Teuchos::make_rcp<std::vector<double>>();
  x2sum_t_->resize(size2, 0.0);
  x3sum_t_ = Teuchos::make_rcp<std::vector<double>>();
  x3sum_t_->resize(size3, 0.0);

  // second-order moments
  x1sumsqrho_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumsqrho_->resize(size1, 0.0);
  x2sumsqrho_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumsqrho_->resize(size2, 0.0);
  x3sumsqrho_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumsqrho_->resize(size3, 0.0);

  x1sumsq_t_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumsq_t_->resize(size1, 0.0);
  x2sumsq_t_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumsq_t_->resize(size2, 0.0);
  x3sumsq_t_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumsq_t_->resize(size3, 0.0);

  x1sumu_t_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumu_t_->resize(size1, 0.0);
  x2sumu_t_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumu_t_->resize(size2, 0.0);
  x3sumu_t_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumu_t_->resize(size3, 0.0);

  x1sumv_t_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumv_t_->resize(size1, 0.0);
  x2sumv_t_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumv_t_->resize(size2, 0.0);
  x3sumv_t_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumv_t_->resize(size3, 0.0);

  x1sumw_t_ = Teuchos::make_rcp<std::vector<double>>();
  x1sumw_t_->resize(size1, 0.0);
  x2sumw_t_ = Teuchos::make_rcp<std::vector<double>>();
  x2sumw_t_->resize(size2, 0.0);
  x3sumw_t_ = Teuchos::make_rcp<std::vector<double>>();
  x3sumw_t_->resize(size3, 0.0);

  //----------------------------------------------------------------------
  // initialize output and initially open respective statistics output file

  Teuchos::RCP<std::ofstream> log;

  if (discret_->get_comm().MyPID() == 0)
  {
    std::string s(statistics_outfilename_);

    if (physicaltype == Inpar::FLUID::loma)
    {
      s.append(".loma_statistics");

      log = Teuchos::make_rcp<std::ofstream>(s.c_str(), std::ios::out);
      (*log) << "# Statistics for turbulent variable-density flow in a lid-driven cavity at low "
                "Mach number (first- and second-order moments)\n\n";
    }
    else
    {
      s.append(".flow_statistics");

      log = Teuchos::make_rcp<std::ofstream>(s.c_str(), std::ios::out);
      (*log) << "# Statistics for turbulent incompressible flow in a lid-driven cavity (first- and "
                "second-order moments)\n\n";
    }

    log->flush();
  }

  // clear statistics
  clear_statistics();

  return;
}  // TurbulenceStatisticsLdc::TurbulenceStatisticsLdc


/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::do_time_sample(Core::LinAlg::Vector<double>& velnp)
{
  //----------------------------------------------------------------------
  // increase sample counter
  //----------------------------------------------------------------------
  numsamp_++;

  int x1nodnum = 0;
  //----------------------------------------------------------------------
  // loop nodes on centerline in x1-direction and calculate pointwise means
  //----------------------------------------------------------------------
  for (std::vector<double>::iterator x1line = x1coordinates_->begin();
       x1line != x1coordinates_->end(); ++x1line)
  {
    // toggle vectors are one in the position of a dof for this line,
    // else 0
    toggleu_->PutScalar(0.0);
    togglev_->PutScalar(0.0);
    togglew_->PutScalar(0.0);
    togglep_->PutScalar(0.0);

    // count the number of nodes contributing to this nodal value on x1-centerline
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);

      // this node belongs to the centerline in x1-direction
      if ((node->x()[0] < (*x1line + 2e-9) && node->x()[0] > (*x1line - 2e-9)) &&
          (node->x()[1] < ((x2max_ + x2min_) / 2.0) + 2e-9 &&
              node->x()[1] > ((x2max_ + x2min_) / 2.0) - 2e-9) &&
          (node->x()[2] < ((x3max_ + x3min_) / 2.0) + 2e-9 &&
              node->x()[2] > ((x3max_ + x3min_) / 2.0) - 2e-9))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->ReplaceGlobalValues(1, &one, &(dof[0]));
        togglev_->ReplaceGlobalValues(1, &one, &(dof[1]));
        togglew_->ReplaceGlobalValues(1, &one, &(dof[2]));
        togglep_->ReplaceGlobalValues(1, &one, &(dof[3]));

        countnodes++;
      }
    }

    int countnodesonallprocs = 0;

    discret_->get_comm().SumAll(&countnodes, &countnodesonallprocs, 1);

    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity and pressure on this centerline
      //----------------------------------------------------------------------
      double u;
      double v;
      double w;
      double p;
      velnp.Dot(*toggleu_, &u);
      velnp.Dot(*togglev_, &v);
      velnp.Dot(*togglew_, &w);
      velnp.Dot(*togglep_, &p);

      //----------------------------------------------------------------------
      // calculate spatial means for velocity and pressure on this centerline
      // (if more than one node contributing to this centerline node)
      //----------------------------------------------------------------------
      double usm = u / countnodesonallprocs;
      double vsm = v / countnodesonallprocs;
      double wsm = w / countnodesonallprocs;
      double psm = p / countnodesonallprocs;

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x1sumu_)[x1nodnum] += usm;
      (*x1sumv_)[x1nodnum] += vsm;
      (*x1sumw_)[x1nodnum] += wsm;
      (*x1sump_)[x1nodnum] += psm;

      (*x1sumsqu_)[x1nodnum] += usm * usm;
      (*x1sumsqv_)[x1nodnum] += vsm * vsm;
      (*x1sumsqw_)[x1nodnum] += wsm * wsm;
      (*x1sumsqp_)[x1nodnum] += psm * psm;

      (*x1sumuv_)[x1nodnum] += usm * vsm;
      (*x1sumuw_)[x1nodnum] += usm * wsm;
      (*x1sumvw_)[x1nodnum] += vsm * wsm;
    }
    x1nodnum++;
  }

  int x2nodnum = 0;
  //----------------------------------------------------------------------
  // loop nodes on centerline in x2-direction and calculate pointwise means
  //----------------------------------------------------------------------
  for (std::vector<double>::iterator x2line = x2coordinates_->begin();
       x2line != x2coordinates_->end(); ++x2line)
  {
    // toggle vectors are one in the position of a dof for this line,
    // else 0
    toggleu_->PutScalar(0.0);
    togglev_->PutScalar(0.0);
    togglew_->PutScalar(0.0);
    togglep_->PutScalar(0.0);

    // count the number of nodes contributing to this nodal value on x2-centerline
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);

      // this node belongs to the centerline in x2-direction
      if ((node->x()[1] < (*x2line + 2e-9) && node->x()[1] > (*x2line - 2e-9)) &&
          (node->x()[0] < ((x1max_ + x1min_) / 2.0) + 2e-9 &&
              node->x()[0] > ((x1max_ + x1min_) / 2.0) - 2e-9) &&
          (node->x()[2] < ((x3max_ + x3min_) / 2.0) + 2e-9 &&
              node->x()[2] > ((x3max_ + x3min_) / 2.0) - 2e-9))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->ReplaceGlobalValues(1, &one, &(dof[0]));
        togglev_->ReplaceGlobalValues(1, &one, &(dof[1]));
        togglew_->ReplaceGlobalValues(1, &one, &(dof[2]));
        togglep_->ReplaceGlobalValues(1, &one, &(dof[3]));

        countnodes++;
      }
    }

    int countnodesonallprocs = 0;

    discret_->get_comm().SumAll(&countnodes, &countnodesonallprocs, 1);

    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity and pressure on this centerline
      //----------------------------------------------------------------------
      double u;
      double v;
      double w;
      double p;
      velnp.Dot(*toggleu_, &u);
      velnp.Dot(*togglev_, &v);
      velnp.Dot(*togglew_, &w);
      velnp.Dot(*togglep_, &p);

      //----------------------------------------------------------------------
      // calculate spatial means for velocity and pressure on this centerline
      // (if more than one node contributing to this centerline node)
      //----------------------------------------------------------------------
      double usm = u / countnodesonallprocs;
      double vsm = v / countnodesonallprocs;
      double wsm = w / countnodesonallprocs;
      double psm = p / countnodesonallprocs;

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x2sumu_)[x2nodnum] += usm;
      (*x2sumv_)[x2nodnum] += vsm;
      (*x2sumw_)[x2nodnum] += wsm;
      (*x2sump_)[x2nodnum] += psm;

      (*x2sumsqu_)[x2nodnum] += usm * usm;
      (*x2sumsqv_)[x2nodnum] += vsm * vsm;
      (*x2sumsqw_)[x2nodnum] += wsm * wsm;
      (*x2sumsqp_)[x2nodnum] += psm * psm;

      (*x2sumuv_)[x2nodnum] += usm * vsm;
      (*x2sumuw_)[x2nodnum] += usm * wsm;
      (*x2sumvw_)[x2nodnum] += vsm * wsm;
    }
    x2nodnum++;
  }

  int x3nodnum = 0;
  //----------------------------------------------------------------------
  // loop nodes on centerline in x3-direction and calculate pointwise means
  //----------------------------------------------------------------------
  for (std::vector<double>::iterator x3line = x3coordinates_->begin();
       x3line != x3coordinates_->end(); ++x3line)
  {
    // toggle vectors are one in the position of a dof for this line,
    // else 0
    toggleu_->PutScalar(0.0);
    togglev_->PutScalar(0.0);
    togglew_->PutScalar(0.0);
    togglep_->PutScalar(0.0);

    // count the number of nodes contributing to this nodal value on x3-centerline
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);

      // this node belongs to the centerline in x3-direction
      if ((node->x()[2] < (*x3line + 2e-9) && node->x()[2] > (*x3line - 2e-9)) &&
          (node->x()[0] < ((x1max_ + x1min_) / 2.0) + 2e-9 &&
              node->x()[0] > ((x1max_ + x1min_) / 2.0) - 2e-9) &&
          (node->x()[1] < ((x2max_ + x2min_) / 2.0) + 2e-9 &&
              node->x()[1] > ((x2max_ + x2min_) / 2.0) - 2e-9))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->ReplaceGlobalValues(1, &one, &(dof[0]));
        togglev_->ReplaceGlobalValues(1, &one, &(dof[1]));
        togglew_->ReplaceGlobalValues(1, &one, &(dof[2]));
        togglep_->ReplaceGlobalValues(1, &one, &(dof[3]));

        countnodes++;
      }
    }

    int countnodesonallprocs = 0;

    discret_->get_comm().SumAll(&countnodes, &countnodesonallprocs, 1);

    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity and pressure on this centerline
      //----------------------------------------------------------------------
      double u;
      double v;
      double w;
      double p;
      velnp.Dot(*toggleu_, &u);
      velnp.Dot(*togglev_, &v);
      velnp.Dot(*togglew_, &w);
      velnp.Dot(*togglep_, &p);

      //----------------------------------------------------------------------
      // calculate spatial means for velocity and pressure on this centerline
      // (if more than one node contributing to this centerline node)
      //----------------------------------------------------------------------
      double usm = u / countnodesonallprocs;
      double vsm = v / countnodesonallprocs;
      double wsm = w / countnodesonallprocs;
      double psm = p / countnodesonallprocs;

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x3sumu_)[x3nodnum] += usm;
      (*x3sumv_)[x3nodnum] += vsm;
      (*x3sumw_)[x3nodnum] += wsm;
      (*x3sump_)[x3nodnum] += psm;

      (*x3sumsqu_)[x3nodnum] += usm * usm;
      (*x3sumsqv_)[x3nodnum] += vsm * vsm;
      (*x3sumsqw_)[x3nodnum] += wsm * wsm;
      (*x3sumsqp_)[x3nodnum] += psm * psm;

      (*x3sumuv_)[x3nodnum] += usm * vsm;
      (*x3sumuw_)[x3nodnum] += usm * wsm;
      (*x3sumvw_)[x3nodnum] += vsm * wsm;
    }
    x3nodnum++;
  }

  return;
}  // TurbulenceStatisticsLdc::DoTimeSample

/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::do_loma_time_sample(Core::LinAlg::Vector<double>& velnp,
    Core::LinAlg::Vector<double>& scanp, Core::LinAlg::Vector<double>& force, const double eosfac)
{
  //----------------------------------------------------------------------
  // increase sample counter
  //----------------------------------------------------------------------
  numsamp_++;

  int x1nodnum = 0;
  //----------------------------------------------------------------------
  // loop nodes on centerline in x1-direction and calculate pointwise means
  //----------------------------------------------------------------------
  for (std::vector<double>::iterator x1line = x1coordinates_->begin();
       x1line != x1coordinates_->end(); ++x1line)
  {
    // toggle vectors are one in the position of a dof for this line,
    // else 0
    toggleu_->PutScalar(0.0);
    togglev_->PutScalar(0.0);
    togglew_->PutScalar(0.0);
    togglep_->PutScalar(0.0);

    // count the number of nodes contributing to this nodal value on x1-centerline
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);

      // this node belongs to the centerline in x1-direction
      if ((node->x()[0] < (*x1line + 2e-9) && node->x()[0] > (*x1line - 2e-9)) &&
          (node->x()[1] < (0.5 + 2e-9) && node->x()[1] > (0.5 - 2e-9)) &&
          (node->x()[2] < ((x3max_ - x3min_) / 2.0) + 2e-9 &&
              node->x()[2] > ((x3max_ - x3min_) / 2.0) - 2e-9))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->ReplaceGlobalValues(1, &one, &(dof[0]));
        togglev_->ReplaceGlobalValues(1, &one, &(dof[1]));
        togglew_->ReplaceGlobalValues(1, &one, &(dof[2]));
        togglep_->ReplaceGlobalValues(1, &one, &(dof[3]));

        countnodes++;
      }
    }

    int countnodesonallprocs = 0;

    discret_->get_comm().SumAll(&countnodes, &countnodesonallprocs, 1);

    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity, pressure, density and temperature
      // on this centerline
      //----------------------------------------------------------------------
      double u;
      double v;
      double w;
      double p;
      velnp.Dot(*toggleu_, &u);
      velnp.Dot(*togglev_, &v);
      velnp.Dot(*togglew_, &w);
      velnp.Dot(*togglep_, &p);

      double T;
      scanp.Dot(*togglep_, &T);

      //----------------------------------------------------------------------
      // calculate spatial means for vel., press., dens. and temp.
      // on this centerline
      // (if more than one node contributing to this centerline node)
      //----------------------------------------------------------------------
      double usm = u / countnodesonallprocs;
      double vsm = v / countnodesonallprocs;
      double wsm = w / countnodesonallprocs;
      double psm = p / countnodesonallprocs;
      double Tsm = T / countnodesonallprocs;
      // compute density: rho = eosfac/T
      double rsm = eosfac / Tsm;

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x1sumu_)[x1nodnum] += usm;
      (*x1sumv_)[x1nodnum] += vsm;
      (*x1sumw_)[x1nodnum] += wsm;
      (*x1sump_)[x1nodnum] += psm;
      (*x1sumrho_)[x1nodnum] += rsm;
      (*x1sum_t_)[x1nodnum] += Tsm;

      (*x1sumsqu_)[x1nodnum] += usm * usm;
      (*x1sumsqv_)[x1nodnum] += vsm * vsm;
      (*x1sumsqw_)[x1nodnum] += wsm * wsm;
      (*x1sumsqp_)[x1nodnum] += psm * psm;
      (*x1sumsqrho_)[x1nodnum] += rsm * rsm;
      (*x1sumsq_t_)[x1nodnum] += Tsm * Tsm;

      (*x1sumuv_)[x1nodnum] += usm * vsm;
      (*x1sumuw_)[x1nodnum] += usm * wsm;
      (*x1sumvw_)[x1nodnum] += vsm * wsm;
      (*x1sumu_t_)[x1nodnum] += usm * Tsm;
      (*x1sumv_t_)[x1nodnum] += vsm * Tsm;
      (*x1sumw_t_)[x1nodnum] += wsm * Tsm;
    }
    x1nodnum++;
  }

  int x2nodnum = 0;
  //----------------------------------------------------------------------
  // loop nodes on centerline in x2-direction and calculate pointwise means
  //----------------------------------------------------------------------
  for (std::vector<double>::iterator x2line = x2coordinates_->begin();
       x2line != x2coordinates_->end(); ++x2line)
  {
    // toggle vectors are one in the position of a dof for this line,
    // else 0
    toggleu_->PutScalar(0.0);
    togglev_->PutScalar(0.0);
    togglew_->PutScalar(0.0);
    togglep_->PutScalar(0.0);

    // count the number of nodes contributing to this nodal value on x2-centerline
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);

      // this node belongs to the centerline in x2-direction
      if ((node->x()[1] < (*x2line + 2e-9) && node->x()[1] > (*x2line - 2e-9)) &&
          (node->x()[0] < (0.5 + 2e-9) && node->x()[0] > (0.5 - 2e-9)) &&
          (node->x()[2] < ((x3max_ - x3min_) / 2.0) + 2e-9 &&
              node->x()[2] > ((x3max_ - x3min_) / 2.0) - 2e-9))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->ReplaceGlobalValues(1, &one, &(dof[0]));
        togglev_->ReplaceGlobalValues(1, &one, &(dof[1]));
        togglew_->ReplaceGlobalValues(1, &one, &(dof[2]));
        togglep_->ReplaceGlobalValues(1, &one, &(dof[3]));

        countnodes++;
      }
    }

    int countnodesonallprocs = 0;

    discret_->get_comm().SumAll(&countnodes, &countnodesonallprocs, 1);

    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity, pressure, density and temperature
      // on this centerline
      //----------------------------------------------------------------------
      double u;
      double v;
      double w;
      double p;
      velnp.Dot(*toggleu_, &u);
      velnp.Dot(*togglev_, &v);
      velnp.Dot(*togglew_, &w);
      velnp.Dot(*togglep_, &p);

      double T;
      scanp.Dot(*togglep_, &T);

      //----------------------------------------------------------------------
      // calculate spatial means for vel., press., dens. and temp.
      // on this centerline
      // (if more than one node contributing to this centerline node)
      //----------------------------------------------------------------------
      double usm = u / countnodesonallprocs;
      double vsm = v / countnodesonallprocs;
      double wsm = w / countnodesonallprocs;
      double psm = p / countnodesonallprocs;
      double Tsm = T / countnodesonallprocs;
      // compute density: rho = eosfac/T
      double rsm = eosfac / Tsm;

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x2sumu_)[x2nodnum] += usm;
      (*x2sumv_)[x2nodnum] += vsm;
      (*x2sumw_)[x2nodnum] += wsm;
      (*x2sump_)[x2nodnum] += psm;
      (*x2sumrho_)[x2nodnum] += rsm;
      (*x2sum_t_)[x2nodnum] += Tsm;

      (*x2sumsqu_)[x2nodnum] += usm * usm;
      (*x2sumsqv_)[x2nodnum] += vsm * vsm;
      (*x2sumsqw_)[x2nodnum] += wsm * wsm;
      (*x2sumsqp_)[x2nodnum] += psm * psm;
      (*x2sumsqrho_)[x2nodnum] += rsm * rsm;
      (*x2sumsq_t_)[x2nodnum] += Tsm * Tsm;

      (*x2sumuv_)[x2nodnum] += usm * vsm;
      (*x2sumuw_)[x2nodnum] += usm * wsm;
      (*x2sumvw_)[x2nodnum] += vsm * wsm;
      (*x2sumu_t_)[x2nodnum] += usm * Tsm;
      (*x2sumv_t_)[x2nodnum] += vsm * Tsm;
      (*x2sumw_t_)[x2nodnum] += wsm * Tsm;
    }
    x2nodnum++;
  }

  int x3nodnum = 0;
  //----------------------------------------------------------------------
  // loop nodes on centerline in x3-direction and calculate pointwise means
  //----------------------------------------------------------------------
  for (std::vector<double>::iterator x3line = x3coordinates_->begin();
       x3line != x3coordinates_->end(); ++x3line)
  {
    // toggle vectors are one in the position of a dof for this line,
    // else 0
    toggleu_->PutScalar(0.0);
    togglev_->PutScalar(0.0);
    togglew_->PutScalar(0.0);
    togglep_->PutScalar(0.0);

    // count the number of nodes contributing to this nodal value on x3-centerline
    int countnodes = 0;

    for (int nn = 0; nn < discret_->num_my_row_nodes(); ++nn)
    {
      Core::Nodes::Node* node = discret_->l_row_node(nn);

      // this node belongs to the centerline in x3-direction
      if ((node->x()[2] < (*x3line + 2e-9) && node->x()[2] > (*x3line - 2e-9)) &&
          (node->x()[0] < (0.5 + 2e-9) && node->x()[0] > (0.5 - 2e-9)) &&
          (node->x()[1] < (0.5 + 2e-9) && node->x()[1] > (0.5 - 2e-9)))
      {
        std::vector<int> dof = discret_->dof(node);
        double one = 1.0;

        toggleu_->ReplaceGlobalValues(1, &one, &(dof[0]));
        togglev_->ReplaceGlobalValues(1, &one, &(dof[1]));
        togglew_->ReplaceGlobalValues(1, &one, &(dof[2]));
        togglep_->ReplaceGlobalValues(1, &one, &(dof[3]));

        countnodes++;
      }
    }

    int countnodesonallprocs = 0;

    discret_->get_comm().SumAll(&countnodes, &countnodesonallprocs, 1);

    if (countnodesonallprocs)
    {
      //----------------------------------------------------------------------
      // get values for velocity, pressure, density and temperature
      // on this centerline
      //----------------------------------------------------------------------
      double u;
      double v;
      double w;
      double p;
      velnp.Dot(*toggleu_, &u);
      velnp.Dot(*togglev_, &v);
      velnp.Dot(*togglew_, &w);
      velnp.Dot(*togglep_, &p);

      double T;
      scanp.Dot(*togglep_, &T);

      //----------------------------------------------------------------------
      // calculate spatial means for vel., press., dens. and temp.
      // on this centerline
      // (if more than one node contributing to this centerline node)
      //----------------------------------------------------------------------
      double usm = u / countnodesonallprocs;
      double vsm = v / countnodesonallprocs;
      double wsm = w / countnodesonallprocs;
      double psm = p / countnodesonallprocs;
      double Tsm = T / countnodesonallprocs;
      // compute density: rho = eosfac/T
      double rsm = eosfac / Tsm;

      //----------------------------------------------------------------------
      // add spatial mean values to statistical sample
      //----------------------------------------------------------------------
      (*x3sumu_)[x3nodnum] += usm;
      (*x3sumv_)[x3nodnum] += vsm;
      (*x3sumw_)[x3nodnum] += wsm;
      (*x3sump_)[x3nodnum] += psm;
      (*x3sumrho_)[x3nodnum] += rsm;
      (*x3sum_t_)[x3nodnum] += Tsm;

      (*x3sumsqu_)[x3nodnum] += usm * usm;
      (*x3sumsqv_)[x3nodnum] += vsm * vsm;
      (*x3sumsqw_)[x3nodnum] += wsm * wsm;
      (*x3sumsqp_)[x3nodnum] += psm * psm;
      (*x3sumsqrho_)[x3nodnum] += rsm * rsm;
      (*x3sumsq_t_)[x3nodnum] += Tsm * Tsm;

      (*x3sumuv_)[x3nodnum] += usm * vsm;
      (*x3sumuw_)[x3nodnum] += usm * wsm;
      (*x3sumvw_)[x3nodnum] += vsm * wsm;
      (*x3sumu_t_)[x3nodnum] += usm * Tsm;
      (*x3sumv_t_)[x3nodnum] += vsm * Tsm;
      (*x3sumw_t_)[x3nodnum] += wsm * Tsm;
    }
    x3nodnum++;
  }

  return;
}  // TurbulenceStatisticsLdc::DoLomaTimeSample

/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::dump_statistics(int step)
{
  //----------------------------------------------------------------------
  // output to log-file
  Teuchos::RCP<std::ofstream> log;
  if (discret_->get_comm().MyPID() == 0)
  {
    std::string s(statistics_outfilename_);
    s.append(".flow_statistics");

    log = Teuchos::make_rcp<std::ofstream>(s.c_str(), std::ios::out);
    (*log) << "# Statistics for turbulent incompressible flow in a lid-driven cavity (first- and "
              "second-order moments)";
    (*log) << "\n\n\n";
    (*log) << "# Statistics record ";
    (*log) << " (Steps " << step - numsamp_ + 1 << "--" << step << ")\n";

    (*log) << "#     x1";
    (*log) << "           umean         vmean         wmean         pmean";
    (*log) << "         urms          vrms          wrms";
    (*log) << "          u'v'          u'w'          v'w'          prms\n";

    (*log) << std::scientific;
    for (unsigned i = 0; i < x1coordinates_->size(); ++i)
    {
      double x1u = (*x1sumu_)[i] / numsamp_;
      double x1v = (*x1sumv_)[i] / numsamp_;
      double x1w = (*x1sumw_)[i] / numsamp_;
      double x1p = (*x1sump_)[i] / numsamp_;

      double x1urms = 0.0;
      double x1vrms = 0.0;
      double x1wrms = 0.0;
      double x1prms = 0.0;

      if (((*x1sumsqu_)[i] / numsamp_ - x1u * x1u) > 0.0)
        x1urms = std::sqrt((*x1sumsqu_)[i] / numsamp_ - x1u * x1u);
      if (((*x1sumsqv_)[i] / numsamp_ - x1v * x1v) > 0.0)
        x1vrms = std::sqrt((*x1sumsqv_)[i] / numsamp_ - x1v * x1v);
      if (((*x1sumsqw_)[i] / numsamp_ - x1w * x1w) > 0.0)
        x1wrms = std::sqrt((*x1sumsqw_)[i] / numsamp_ - x1w * x1w);
      if (((*x1sumsqp_)[i] / numsamp_ - x1p * x1p) > 0.0)
        x1prms = std::sqrt((*x1sumsqp_)[i] / numsamp_ - x1p * x1p);

      double x1uv = (*x1sumuv_)[i] / numsamp_ - x1u * x1v;
      double x1uw = (*x1sumuw_)[i] / numsamp_ - x1u * x1w;
      double x1vw = (*x1sumvw_)[i] / numsamp_ - x1v * x1w;

      (*log) << " " << std::setw(11) << std::setprecision(4) << (*x1coordinates_)[i];
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1u;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1v;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1w;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1urms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1vrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1wrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1uv;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1uw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1vw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1prms;
      (*log) << "\n";
    }

    (*log) << "\n\n\n";
    (*log) << "#     x2";
    (*log) << "           umean         vmean         wmean         pmean";
    (*log) << "         urms          vrms          wrms";
    (*log) << "          u'v'          u'w'          v'w'          prms\n";

    (*log) << std::scientific;
    for (unsigned i = 0; i < x2coordinates_->size(); ++i)
    {
      double x2u = (*x2sumu_)[i] / numsamp_;
      double x2v = (*x2sumv_)[i] / numsamp_;
      double x2w = (*x2sumw_)[i] / numsamp_;
      double x2p = (*x2sump_)[i] / numsamp_;

      double x2urms = 0.0;
      double x2vrms = 0.0;
      double x2wrms = 0.0;
      double x2prms = 0.0;

      if (((*x2sumsqu_)[i] / numsamp_ - x2u * x2u) > 0.0)
        x2urms = std::sqrt((*x2sumsqu_)[i] / numsamp_ - x2u * x2u);
      if (((*x2sumsqv_)[i] / numsamp_ - x2v * x2v) > 0.0)
        x2vrms = std::sqrt((*x2sumsqv_)[i] / numsamp_ - x2v * x2v);
      if (((*x2sumsqw_)[i] / numsamp_ - x2w * x2w) > 0.0)
        x2wrms = std::sqrt((*x2sumsqw_)[i] / numsamp_ - x2w * x2w);
      if (((*x2sumsqp_)[i] / numsamp_ - x2p * x2p) > 0.0)
        x2prms = std::sqrt((*x2sumsqp_)[i] / numsamp_ - x2p * x2p);

      double x2uv = (*x2sumuv_)[i] / numsamp_ - x2u * x2v;
      double x2uw = (*x2sumuw_)[i] / numsamp_ - x2u * x2w;
      double x2vw = (*x2sumvw_)[i] / numsamp_ - x2v * x2w;

      (*log) << " " << std::setw(11) << std::setprecision(4) << (*x2coordinates_)[i];
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2u;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2v;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2w;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2urms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2wrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uv;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2prms;
      (*log) << "\n";
    }

    (*log) << "\n\n\n";
    (*log) << "#     x3";
    (*log) << "           umean         vmean         wmean         pmean";
    (*log) << "         urms          vrms          wrms";
    (*log) << "          u'v'          u'w'          v'w'          prms\n";

    (*log) << std::scientific;
    for (unsigned i = 0; i < x3coordinates_->size(); ++i)
    {
      double x3u = (*x3sumu_)[i] / numsamp_;
      double x3v = (*x3sumv_)[i] / numsamp_;
      double x3w = (*x3sumw_)[i] / numsamp_;
      double x3p = (*x3sump_)[i] / numsamp_;

      double x3urms = 0.0;
      double x3vrms = 0.0;
      double x3wrms = 0.0;
      double x3prms = 0.0;

      if (((*x3sumsqu_)[i] / numsamp_ - x3u * x3u) > 0.0)
        x3urms = std::sqrt((*x3sumsqu_)[i] / numsamp_ - x3u * x3u);
      if (((*x3sumsqv_)[i] / numsamp_ - x3v * x3v) > 0.0)
        x3vrms = std::sqrt((*x3sumsqv_)[i] / numsamp_ - x3v * x3v);
      if (((*x3sumsqw_)[i] / numsamp_ - x3w * x3w) > 0.0)
        x3wrms = std::sqrt((*x3sumsqw_)[i] / numsamp_ - x3w * x3w);
      if (((*x3sumsqp_)[i] / numsamp_ - x3p * x3p) > 0.0)
        x3prms = std::sqrt((*x3sumsqp_)[i] / numsamp_ - x3p * x3p);

      double x3uv = (*x3sumuv_)[i] / numsamp_ - x3u * x3v;
      double x3uw = (*x3sumuw_)[i] / numsamp_ - x3u * x3w;
      double x3vw = (*x3sumvw_)[i] / numsamp_ - x3v * x3w;

      (*log) << " " << std::setw(11) << std::setprecision(4) << (*x3coordinates_)[i];
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3u;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3v;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3w;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3urms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3vrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3wrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3uv;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3uw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3vw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3prms;
      (*log) << "\n";
    }
    log->flush();
  }

  return;

}  // TurbulenceStatisticsLdc::DumpStatistics


/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::dump_loma_statistics(int step)
{
  //----------------------------------------------------------------------
  // output to log-file
  Teuchos::RCP<std::ofstream> log;
  if (discret_->get_comm().MyPID() == 0)
  {
    std::string s(statistics_outfilename_);
    s.append(".loma_statistics");

    log = Teuchos::make_rcp<std::ofstream>(s.c_str(), std::ios::out);
    (*log) << "# Statistics for turbulent variable-density flow in a lid-driven cavity at low Mach "
              "number (first- and second-order moments)";
    (*log) << "\n\n\n";
    (*log) << "# Statistics record ";
    (*log) << " (Steps " << step - numsamp_ + 1 << "--" << step << ")\n";

    (*log) << "#     x1";
    (*log)
        << "           umean         vmean         wmean         pmean       rhomean         Tmean";
    (*log) << "         urms          vrms          wrms          prms        rhorms          Trms";
    (*log)
        << "          u'v'          u'w'          v'w'          u'T'          v'T'          w'T'\n";

    (*log) << std::scientific;
    for (unsigned i = 0; i < x1coordinates_->size(); ++i)
    {
      double x1u = (*x1sumu_)[i] / numsamp_;
      double x1v = (*x1sumv_)[i] / numsamp_;
      double x1w = (*x1sumw_)[i] / numsamp_;
      double x1p = (*x1sump_)[i] / numsamp_;
      double x1rho = (*x1sumrho_)[i] / numsamp_;
      double x1T = (*x1sum_t_)[i] / numsamp_;

      double x1urms = 0.0;
      double x1vrms = 0.0;
      double x1wrms = 0.0;
      double x1prms = 0.0;
      double x1rhorms = 0.0;
      double x1Trms = 0.0;

      if (((*x1sumsqu_)[i] / numsamp_ - x1u * x1u) > 0.0)
        x1urms = std::sqrt((*x1sumsqu_)[i] / numsamp_ - x1u * x1u);
      if (((*x1sumsqv_)[i] / numsamp_ - x1v * x1v) > 0.0)
        x1vrms = std::sqrt((*x1sumsqv_)[i] / numsamp_ - x1v * x1v);
      if (((*x1sumsqw_)[i] / numsamp_ - x1w * x1w) > 0.0)
        x1wrms = std::sqrt((*x1sumsqw_)[i] / numsamp_ - x1w * x1w);
      if (((*x1sumsqp_)[i] / numsamp_ - x1p * x1p) > 0.0)
        x1prms = std::sqrt((*x1sumsqp_)[i] / numsamp_ - x1p * x1p);
      if (((*x1sumsqrho_)[i] / numsamp_ - x1rho * x1rho) > 0.0)
        x1rhorms = std::sqrt((*x1sumsqrho_)[i] / numsamp_ - x1rho * x1rho);
      if (((*x1sumsq_t_)[i] / numsamp_ - x1T * x1T) > 0.0)
        x1Trms = std::sqrt((*x1sumsq_t_)[i] / numsamp_ - x1T * x1T);

      double x1uv = (*x1sumuv_)[i] / numsamp_ - x1u * x1v;
      double x1uw = (*x1sumuw_)[i] / numsamp_ - x1u * x1w;
      double x1vw = (*x1sumvw_)[i] / numsamp_ - x1v * x1w;
      double x1uT = (*x1sumu_t_)[i] / numsamp_ - x1u * x1T;
      double x1vT = (*x1sumv_t_)[i] / numsamp_ - x1v * x1T;
      double x1wT = (*x1sumw_t_)[i] / numsamp_ - x1w * x1T;

      (*log) << " " << std::setw(11) << std::setprecision(4) << (*x1coordinates_)[i];
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1u;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1v;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1w;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1rho;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1T;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1urms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1vrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1wrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1prms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1rhorms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1Trms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1uv;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1uw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1vw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1uT;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1vT;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x1wT;
      (*log) << "\n";
    }

    (*log) << "#     x2";
    (*log)
        << "           umean         vmean         wmean         pmean       rhomean         Tmean";
    (*log) << "         urms          vrms          wrms          prms        rhorms          Trms";
    (*log)
        << "          u'v'          u'w'          v'w'          u'T'          v'T'          w'T'\n";

    (*log) << std::scientific;
    for (unsigned i = 0; i < x2coordinates_->size(); ++i)
    {
      double x2u = (*x2sumu_)[i] / numsamp_;
      double x2v = (*x2sumv_)[i] / numsamp_;
      double x2w = (*x2sumw_)[i] / numsamp_;
      double x2p = (*x2sump_)[i] / numsamp_;
      double x2rho = (*x2sumrho_)[i] / numsamp_;
      double x2T = (*x2sum_t_)[i] / numsamp_;

      double x2urms = 0.0;
      double x2vrms = 0.0;
      double x2wrms = 0.0;
      double x2prms = 0.0;
      double x2rhorms = 0.0;
      double x2Trms = 0.0;

      if (((*x2sumsqu_)[i] / numsamp_ - x2u * x2u) > 0.0)
        x2urms = std::sqrt((*x2sumsqu_)[i] / numsamp_ - x2u * x2u);
      if (((*x2sumsqv_)[i] / numsamp_ - x2v * x2v) > 0.0)
        x2vrms = std::sqrt((*x2sumsqv_)[i] / numsamp_ - x2v * x2v);
      if (((*x2sumsqw_)[i] / numsamp_ - x2w * x2w) > 0.0)
        x2wrms = std::sqrt((*x2sumsqw_)[i] / numsamp_ - x2w * x2w);
      if (((*x2sumsqp_)[i] / numsamp_ - x2p * x2p) > 0.0)
        x2prms = std::sqrt((*x2sumsqp_)[i] / numsamp_ - x2p * x2p);
      if (((*x2sumsqrho_)[i] / numsamp_ - x2rho * x2rho) > 0.0)
        x2rhorms = std::sqrt((*x2sumsqrho_)[i] / numsamp_ - x2rho * x2rho);
      if (((*x2sumsq_t_)[i] / numsamp_ - x2T * x2T) > 0.0)
        x2Trms = std::sqrt((*x2sumsq_t_)[i] / numsamp_ - x2T * x2T);

      double x2uv = (*x2sumuv_)[i] / numsamp_ - x2u * x2v;
      double x2uw = (*x2sumuw_)[i] / numsamp_ - x2u * x2w;
      double x2vw = (*x2sumvw_)[i] / numsamp_ - x2v * x2w;
      double x2uT = (*x2sumu_t_)[i] / numsamp_ - x2u * x2T;
      double x2vT = (*x2sumv_t_)[i] / numsamp_ - x2v * x2T;
      double x2wT = (*x2sumw_t_)[i] / numsamp_ - x2w * x2T;

      (*log) << " " << std::setw(11) << std::setprecision(4) << (*x2coordinates_)[i];
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2u;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2v;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2w;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2rho;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2T;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2urms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2wrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2prms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2rhorms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2Trms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uv;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2uT;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2vT;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x2wT;
      (*log) << "\n";
    }

    (*log) << "#     x3";
    (*log)
        << "           umean         vmean         wmean         pmean       rhomean         Tmean";
    (*log) << "         urms          vrms          wrms          prms        rhorms          Trms";
    (*log)
        << "          u'v'          u'w'          v'w'          u'T'          v'T'          w'T'\n";

    (*log) << std::scientific;
    for (unsigned i = 0; i < x3coordinates_->size(); ++i)
    {
      double x3u = (*x3sumu_)[i] / numsamp_;
      double x3v = (*x3sumv_)[i] / numsamp_;
      double x3w = (*x3sumw_)[i] / numsamp_;
      double x3p = (*x3sump_)[i] / numsamp_;
      double x3rho = (*x3sumrho_)[i] / numsamp_;
      double x3T = (*x3sum_t_)[i] / numsamp_;

      double x3urms = 0.0;
      double x3vrms = 0.0;
      double x3wrms = 0.0;
      double x3prms = 0.0;
      double x3rhorms = 0.0;
      double x3Trms = 0.0;

      if (((*x3sumsqu_)[i] / numsamp_ - x3u * x3u) > 0.0)
        x3urms = std::sqrt((*x3sumsqu_)[i] / numsamp_ - x3u * x3u);
      if (((*x3sumsqv_)[i] / numsamp_ - x3v * x3v) > 0.0)
        x3vrms = std::sqrt((*x3sumsqv_)[i] / numsamp_ - x3v * x3v);
      if (((*x3sumsqw_)[i] / numsamp_ - x3w * x3w) > 0.0)
        x3wrms = std::sqrt((*x3sumsqw_)[i] / numsamp_ - x3w * x3w);
      if (((*x3sumsqp_)[i] / numsamp_ - x3p * x3p) > 0.0)
        x3prms = std::sqrt((*x3sumsqp_)[i] / numsamp_ - x3p * x3p);
      if (((*x3sumsqrho_)[i] / numsamp_ - x3rho * x3rho) > 0.0)
        x3rhorms = std::sqrt((*x3sumsqrho_)[i] / numsamp_ - x3rho * x3rho);
      if (((*x3sumsq_t_)[i] / numsamp_ - x3T * x3T) > 0.0)
        x3Trms = std::sqrt((*x3sumsq_t_)[i] / numsamp_ - x3T * x3T);

      double x3uv = (*x3sumuv_)[i] / numsamp_ - x3u * x3v;
      double x3uw = (*x3sumuw_)[i] / numsamp_ - x3u * x3w;
      double x3vw = (*x3sumvw_)[i] / numsamp_ - x3v * x3w;
      double x3uT = (*x3sumu_t_)[i] / numsamp_ - x3u * x3T;
      double x3vT = (*x3sumv_t_)[i] / numsamp_ - x3v * x3T;
      double x3wT = (*x3sumw_t_)[i] / numsamp_ - x3w * x3T;

      (*log) << " " << std::setw(11) << std::setprecision(4) << (*x3coordinates_)[i];
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3u;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3v;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3w;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3p;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3rho;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3T;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3urms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3vrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3wrms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3prms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3rhorms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3Trms;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3uv;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3uw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3vw;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3uT;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3vT;
      (*log) << "   " << std::setw(11) << std::setprecision(4) << x3wT;
      (*log) << "\n";
    }
    log->flush();
  }

  return;

}  // TurbulenceStatisticsLdc::DumpLomaStatistics


/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::clear_statistics()
{
  numsamp_ = 0;

  for (unsigned i = 0; i < x1coordinates_->size(); ++i)
  {
    (*x1sumu_)[i] = 0.0;
    (*x1sumv_)[i] = 0.0;
    (*x1sumw_)[i] = 0.0;
    (*x1sump_)[i] = 0.0;
    (*x1sumrho_)[i] = 0.0;
    (*x1sum_t_)[i] = 0.0;

    (*x1sumsqu_)[i] = 0.0;
    (*x1sumsqv_)[i] = 0.0;
    (*x1sumsqw_)[i] = 0.0;
    (*x1sumsqp_)[i] = 0.0;
    (*x1sumsqrho_)[i] = 0.0;
    (*x1sumsq_t_)[i] = 0.0;

    (*x1sumuv_)[i] = 0.0;
    (*x1sumuw_)[i] = 0.0;
    (*x1sumvw_)[i] = 0.0;
    (*x1sumu_t_)[i] = 0.0;
    (*x1sumv_t_)[i] = 0.0;
    (*x1sumw_t_)[i] = 0.0;
  }

  for (unsigned i = 0; i < x2coordinates_->size(); ++i)
  {
    (*x2sumu_)[i] = 0.0;
    (*x2sumv_)[i] = 0.0;
    (*x2sumw_)[i] = 0.0;
    (*x2sump_)[i] = 0.0;
    (*x2sumrho_)[i] = 0.0;
    (*x2sum_t_)[i] = 0.0;

    (*x2sumsqu_)[i] = 0.0;
    (*x2sumsqv_)[i] = 0.0;
    (*x2sumsqw_)[i] = 0.0;
    (*x2sumsqp_)[i] = 0.0;
    (*x2sumsqrho_)[i] = 0.0;
    (*x2sumsq_t_)[i] = 0.0;

    (*x2sumuv_)[i] = 0.0;
    (*x2sumuw_)[i] = 0.0;
    (*x2sumvw_)[i] = 0.0;
    (*x2sumu_t_)[i] = 0.0;
    (*x2sumv_t_)[i] = 0.0;
    (*x2sumw_t_)[i] = 0.0;
  }

  for (unsigned i = 0; i < x3coordinates_->size(); ++i)
  {
    (*x3sumu_)[i] = 0.0;
    (*x3sumv_)[i] = 0.0;
    (*x3sumw_)[i] = 0.0;
    (*x3sump_)[i] = 0.0;
    (*x3sumrho_)[i] = 0.0;
    (*x3sum_t_)[i] = 0.0;

    (*x3sumsqu_)[i] = 0.0;
    (*x3sumsqv_)[i] = 0.0;
    (*x3sumsqw_)[i] = 0.0;
    (*x3sumsqp_)[i] = 0.0;
    (*x3sumsqrho_)[i] = 0.0;
    (*x3sumsq_t_)[i] = 0.0;

    (*x3sumuv_)[i] = 0.0;
    (*x3sumuw_)[i] = 0.0;
    (*x3sumvw_)[i] = 0.0;
    (*x3sumu_t_)[i] = 0.0;
    (*x3sumv_t_)[i] = 0.0;
    (*x3sumw_t_)[i] = 0.0;
  }

  return;
}  // TurbulenceStatisticsLdc::ClearStatistics

/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::read_restart(Core::IO::DiscretizationReader& reader)
{
  numsamp_ = reader.read_double("numsamp");

  reader.read_redundant_double_vector(x1sumu_, "x1sumu");
  reader.read_redundant_double_vector(x1sumv_, "x1sumv");
  reader.read_redundant_double_vector(x1sumw_, "x1sumw");
  reader.read_redundant_double_vector(x1sump_, "x1sump");
  //  reader.read_redundant_double_vector(x1sumrho_ ,"x1sumrho");
  //  reader.read_redundant_double_vector(x1sumT_ ,"x1sumT");

  reader.read_redundant_double_vector(x1sumsqu_, "x1sumsqu");
  reader.read_redundant_double_vector(x1sumsqv_, "x1sumsqv");
  reader.read_redundant_double_vector(x1sumsqw_, "x1sumsqw");
  reader.read_redundant_double_vector(x1sumsqp_, "x1sumsqp");
  //  reader.read_redundant_double_vector(x1sumsqrho_ ,"x1sumsqrho");
  //  reader.read_redundant_double_vector(x1sumsqT_ ,"x1sumsqT");

  reader.read_redundant_double_vector(x1sumuv_, "x1sumuv");
  reader.read_redundant_double_vector(x1sumuw_, "x1sumuw");
  reader.read_redundant_double_vector(x1sumvw_, "x1sumvw");
  //  reader.read_redundant_double_vector(x1sumuT_ ,"x1sumuT");
  //  reader.read_redundant_double_vector(x1sumvT_ ,"x1sumvT");
  //  reader.read_redundant_double_vector(x1sumwT_ ,"x1sumwT");

  reader.read_redundant_double_vector(x2sumu_, "x2sumu");
  reader.read_redundant_double_vector(x2sumv_, "x2sumv");
  reader.read_redundant_double_vector(x2sumw_, "x2sumw");
  reader.read_redundant_double_vector(x2sump_, "x2sump");
  //  reader.read_redundant_double_vector(x2sumrho_ ,"x2sumrho");
  //  reader.read_redundant_double_vector(x2sumT_ ,"x2sumT");

  reader.read_redundant_double_vector(x2sumsqu_, "x2sumsqu");
  reader.read_redundant_double_vector(x2sumsqv_, "x2sumsqv");
  reader.read_redundant_double_vector(x2sumsqw_, "x2sumsqw");
  reader.read_redundant_double_vector(x2sumsqp_, "x2sumsqp");
  //  reader.read_redundant_double_vector(x2sumsqrho_ ,"x2sumsqrho");
  //  reader.read_redundant_double_vector(x2sumsqT_ ,"x2sumsqT");

  reader.read_redundant_double_vector(x2sumuv_, "x2sumuv");
  reader.read_redundant_double_vector(x2sumuw_, "x2sumuw");
  reader.read_redundant_double_vector(x2sumvw_, "x2sumvw");
  //  reader.read_redundant_double_vector(x2sumuT_ ,"x2sumuT");
  //  reader.read_redundant_double_vector(x2sumvT_ ,"x2sumvT");
  //  reader.read_redundant_double_vector(x2sumwT_ ,"x2sumwT");

  reader.read_redundant_double_vector(x3sumu_, "x3sumu");
  reader.read_redundant_double_vector(x3sumv_, "x3sumv");
  reader.read_redundant_double_vector(x3sumw_, "x3sumw");
  reader.read_redundant_double_vector(x3sump_, "x3sump");
  //  reader.read_redundant_double_vector(x3sumrho_ ,"x3sumrho");
  //  reader.read_redundant_double_vector(x3sumT_ ,"x3sumT");

  reader.read_redundant_double_vector(x3sumsqu_, "x3sumsqu");
  reader.read_redundant_double_vector(x3sumsqv_, "x3sumsqv");
  reader.read_redundant_double_vector(x3sumsqw_, "x3sumsqw");
  reader.read_redundant_double_vector(x3sumsqp_, "x3sumsqp");
  //  reader.read_redundant_double_vector(x3sumsqrho_ ,"x3sumsqrho");
  //  reader.read_redundant_double_vector(x3sumsqT_ ,"x3sumsqT");

  reader.read_redundant_double_vector(x3sumuv_, "x3sumuv");
  reader.read_redundant_double_vector(x3sumuw_, "x3sumuw");
  reader.read_redundant_double_vector(x3sumvw_, "x3sumvw");
  //  reader.read_redundant_double_vector(x3sumuT_ ,"x3sumuT");
  //  reader.read_redundant_double_vector(x3sumvT_ ,"x3sumvT");
  //  reader.read_redundant_double_vector(x3sumwT_ ,"x3sumwT");
}

/*----------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsLdc::write_restart(Core::IO::DiscretizationWriter& writer)
{
  writer.write_double("numsamp", numsamp_);

  writer.write_redundant_double_vector("x1sumu", *x1sumu_);
  writer.write_redundant_double_vector("x1sumv", *x1sumv_);
  writer.write_redundant_double_vector("x1sumw", *x1sumw_);
  writer.write_redundant_double_vector("x1sump", *x1sump_);
  //  writer.write_redundant_double_vector("x1sumrho", x1sumrho_);
  //  writer.write_redundant_double_vector("x1sumT", x1sumT_);

  writer.write_redundant_double_vector("x1sumsqu", *x1sumsqu_);
  writer.write_redundant_double_vector("x1sumsqv", *x1sumsqv_);
  writer.write_redundant_double_vector("x1sumsqw", *x1sumsqw_);
  writer.write_redundant_double_vector("x1sumsqp", *x1sumsqp_);
  //  writer.write_redundant_double_vector("x1sumsqrho", x1sumsqrho_);
  //  writer.write_redundant_double_vector("x1sumsqT", x1sumsqT_);

  writer.write_redundant_double_vector("x1sumuv", *x1sumuv_);
  writer.write_redundant_double_vector("x1sumuw", *x1sumuw_);
  writer.write_redundant_double_vector("x1sumvw", *x1sumvw_);
  //  writer.write_redundant_double_vector("x1sumuT", x1sumuT_);
  //  writer.write_redundant_double_vector("x1sumvT", x1sumvT_);
  //  writer.write_redundant_double_vector("x1sumwT", x1sumwT_);

  writer.write_redundant_double_vector("x2sumu", *x2sumu_);
  writer.write_redundant_double_vector("x2sumv", *x2sumv_);
  writer.write_redundant_double_vector("x2sumw", *x2sumw_);
  writer.write_redundant_double_vector("x2sump", *x2sump_);
  //  writer.write_redundant_double_vector("x2sumrho", x2sumrho_);
  //  writer.write_redundant_double_vector("x2sumT", x2sumT_);

  writer.write_redundant_double_vector("x2sumsqu", *x2sumsqu_);
  writer.write_redundant_double_vector("x2sumsqv", *x2sumsqv_);
  writer.write_redundant_double_vector("x2sumsqw", *x2sumsqw_);
  writer.write_redundant_double_vector("x2sumsqp", *x2sumsqp_);
  //  writer.write_redundant_double_vector("x2sumsqrho", x2sumsqrho_);
  //  writer.write_redundant_double_vector("x2sumsqT", x2sumsqT_);

  writer.write_redundant_double_vector("x2sumuv", *x2sumuv_);
  writer.write_redundant_double_vector("x2sumuw", *x2sumuw_);
  writer.write_redundant_double_vector("x2sumvw", *x2sumvw_);
  //  writer.write_redundant_double_vector("x2sumuT", x2sumuT_);
  //  writer.write_redundant_double_vector("x2sumvT", x2sumvT_);
  //  writer.write_redundant_double_vector("x2sumwT", x2sumwT_);

  writer.write_redundant_double_vector("x3sumu", *x3sumu_);
  writer.write_redundant_double_vector("x3sumv", *x3sumv_);
  writer.write_redundant_double_vector("x3sumw", *x3sumw_);
  writer.write_redundant_double_vector("x3sump", *x3sump_);
  //  writer.write_redundant_double_vector("x3sumrho", x3sumrho_);
  //  writer.write_redundant_double_vector("x3sumT", x3sumT_);

  writer.write_redundant_double_vector("x3sumsqu", *x3sumsqu_);
  writer.write_redundant_double_vector("x3sumsqv", *x3sumsqv_);
  writer.write_redundant_double_vector("x3sumsqw", *x3sumsqw_);
  writer.write_redundant_double_vector("x3sumsqp", *x3sumsqp_);
  //  writer.write_redundant_double_vector("x3sumsqrho", x3sumsqrho_);
  //  writer.write_redundant_double_vector("x3sumsqT", x3sumsqT_);

  writer.write_redundant_double_vector("x3sumuv", *x3sumuv_);
  writer.write_redundant_double_vector("x3sumuw", *x3sumuw_);
  writer.write_redundant_double_vector("x3sumvw", *x3sumvw_);
  //  writer.write_redundant_double_vector("x3sumuT", x3sumuT_);
  //  writer.write_redundant_double_vector("x3sumvT", x3sumvT_);
  //  writer.write_redundant_double_vector("x3sumwT", x3sumwT_);
}

FOUR_C_NAMESPACE_CLOSE
