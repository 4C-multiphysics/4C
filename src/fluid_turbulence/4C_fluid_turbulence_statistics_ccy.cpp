// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_statistics_ccy.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_newtonianfluid.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------

                  Standard Constructor (public)

  ---------------------------------------------------------------------*/
FLD::TurbulenceStatisticsCcy::TurbulenceStatisticsCcy(
    std::shared_ptr<Core::FE::Discretization> actdis, bool alefluid,
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp, Teuchos::ParameterList& params,
    const std::string& statistics_outfilename, const bool withscatra)
    : discret_(actdis),
      dispnp_(dispnp),
      params_(params),
      statistics_outfilename_(statistics_outfilename),
      withscatra_(withscatra),
      numscatradofpernode_(0)
{
  //----------------------------------------------------------------------
  // plausibility check
  int numdim = params_.get<int>("number of velocity degrees of freedom");
  if (numdim != 3)
  {
    FOUR_C_THROW("Evaluation of turbulence statistics only for 3d flows!");
  }

  //----------------------------------------------------------------------
  // allocate some vectors
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  meanvelnp_ = Core::LinAlg::create_vector(*dofrowmap, true);

  if (withscatra_)
  {
    meanscanp_ = Core::LinAlg::create_vector(*dofrowmap, true);
    // meanfullphinp_ is initialized in ApplyScatraResults()
  }

  //----------------------------------------------------------------------
  // switches, control parameters, material parameters

  // get the plane normal direction from the parameterlist
  {
    std::string plainstring =
        params_.sublist("TURBULENCE MODEL").get<std::string>("HOMDIR", "not_specified");

    if (plainstring == "z")
    {
      dim_ = 2;
    }
    else
    {
      FOUR_C_THROW("homogeneous direction for this flow was specified incorrectly. (need z)");
    }
  }

  // ---------------------------------------------------------------------
  // up to now, there are no records written
  countrecord_ = 0;

  // ---------------------------------------------------------------------
  // compute all planes for sampling

  // available shells of element corners (Nurbs) of elements
  nodeshells_ = std::make_shared<std::vector<double>>();

  // available homogeneous (sampling) shells --- there are
  // numsubdivisions layers per element layer
  shellcoordinates_ = std::make_shared<std::vector<double>>();

  const int numsubdivisions = 5;

  // try to cast discretisation to nurbs variant
  // this tells you what kind of computation of
  // samples is required
  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*actdis));

  if (nurbsdis == nullptr)
  {
    FOUR_C_THROW("Need Nurbs mesh for turbulent flows around a circular cylinder\n");
  }
  else
  {
    // real pointwise control point sampling does not make any sense
    // for Nurbs discretisations since shape functions are not interpolating

    // radial shellcoordinates are determined by the element
    // (cartesian) number in the second knotspan direction and
    // the number of sampling shells in between are added

    // for nurbs discretisations, all vector sizes are already determined
    // by the knotvector size

    // get nurbs dis' knotvector sizes
    std::vector<int> n_x_m_x_l(nurbsdis->return_n_x_m_x_l(0));

    // get nurbs dis' element numbers
    std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(0));

    // get the knotvector itself
    std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

    // resize and initialise to 0
    {
      (*nodeshells_).resize(nele_x_mele_x_lele[1] + 1);
      (*shellcoordinates_).resize(nele_x_mele_x_lele[1] * (numsubdivisions - 1) + 1);

      std::vector<double>::iterator coord;

      for (coord = (*nodeshells_).begin(); coord != (*nodeshells_).end(); ++coord)
      {
        *coord = 0;
      }
      for (coord = shellcoordinates_->begin(); coord != shellcoordinates_->end(); ++coord)
      {
        *coord = 0;
      }
    }

    // count numbers of nodes in homogeneous shells
    int nodeshellsize = nodeshells_->size();

    std::vector<int> nodeshells_numnodes(nodeshellsize, 0);
    std::vector<int> shellcoordinates_numnodes((*shellcoordinates_).size(), 0);

    // get element map
    const Core::LinAlg::Map* elementmap = nurbsdis->element_row_map();

    // loop all available elements
    for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
    {
      // get element pointer
      Core::Elements::Element* const actele = nurbsdis->g_element(elementmap->gid(iele));

      // want to loop all control points of the element,
      // so get the number of points
      const int numnp = actele->num_node();

      // get the elements control points/nodes
      Core::Nodes::Node** nodes = actele->nodes();

      // acquire weights from nodes
      Core::LinAlg::SerialDenseVector weights(numnp);

      for (int inode = 0; inode < numnp; ++inode)
      {
        Core::FE::Nurbs::ControlPoint* cp =
            dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

        weights(inode) = cp->w();
      }

      // get gid, location in the patch
      int gid = actele->id();

      int patchid = 0;

      std::vector<int> ele_cart_id(3);
      knots->convert_ele_gid_to_knot_ids(gid, patchid, ele_cart_id);

      // access elements knot span
      std::vector<Core::LinAlg::SerialDenseVector> knots(3);
      bool zero_size = (*((*nurbsdis).get_knot_vector())).get_ele_knots(knots, actele->id());

      // zero sized elements have to be skipped
      if (zero_size)
      {
        continue;
      }

      // get shapefunctions, compute all visualisation point positions
      Core::LinAlg::SerialDenseVector nurbs_shape_funct(numnp);

      switch (actele->shape())
      {
        case Core::FE::CellType::nurbs8:
        case Core::FE::CellType::nurbs27:
        {
          // element local point position
          Core::LinAlg::SerialDenseVector uv(3);

          {
            // standard

            //               v
            //              /
            //  w  7       /   8
            //  ^   +---------+
            //  |  /         /|
            //  | /         / |
            // 5|/        6/  |
            //  +---------+   |
            //  |         |   |
            //  |         |   +
            //  |         |  / 4
            //  |         | /
            //  |         |/
            //  +---------+ ----->u
            // 1           2
            // use r-coordinate of point 2 and 4
            // temporary x vector
            std::vector<double> x(3);

            // point 1
            uv(0) = 1.0;
            uv(1) = -1.0;
            uv(2) = -1.0;
            Core::FE::Nurbs::nurbs_get_3d_funct(
                nurbs_shape_funct, uv, knots, weights, actele->shape());
            for (int isd = 0; isd < 3; ++isd)
            {
              double val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
              }
              x[isd] = val;
            }

            (*nodeshells_)[ele_cart_id[1]] += sqrt(x[0] * x[0] + x[1] * x[1]);
            (*shellcoordinates_)[ele_cart_id[1] * (numsubdivisions - 1)] +=
                sqrt(x[0] * x[0] + x[1] * x[1]);

            nodeshells_numnodes[ele_cart_id[1]] += 1;
            shellcoordinates_numnodes[ele_cart_id[1] * (numsubdivisions - 1)] += 1;


            for (int rr = 1; rr < numsubdivisions - 1; ++rr)
            {
              uv(1) += 2.0 / (numsubdivisions - 1);

              Core::FE::Nurbs::nurbs_get_3d_funct(
                  nurbs_shape_funct, uv, knots, weights, actele->shape());
              for (int isd = 0; isd < 3; ++isd)
              {
                double val = 0;
                for (int inode = 0; inode < numnp; ++inode)
                {
                  val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
                }
                x[isd] = val;
              }
              (*shellcoordinates_)[ele_cart_id[1] * (numsubdivisions - 1) + rr] +=
                  sqrt(x[0] * x[0] + x[1] * x[1]);
              ++(shellcoordinates_numnodes[ele_cart_id[1] * (numsubdivisions - 1) + rr]);
            }


            // set upper point of element, too (only for last layer)
            if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
            {
              // point 8
              uv(0) = 1.0;
              uv(1) = 1.0;
              uv(2) = -1.0;
              Core::FE::Nurbs::nurbs_get_3d_funct(
                  nurbs_shape_funct, uv, knots, weights, actele->shape());
              for (int isd = 0; isd < 3; ++isd)
              {
                double val = 0;
                for (int inode = 0; inode < numnp; ++inode)
                {
                  val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
                }
                x[isd] = val;
              }

              (*nodeshells_)[ele_cart_id[1] + 1] += sqrt(x[0] * x[0] + x[1] * x[1]);
              (*shellcoordinates_)[(ele_cart_id[1] + 1) * (numsubdivisions - 1)] +=
                  sqrt(x[0] * x[0] + x[1] * x[1]);
              ++(nodeshells_numnodes[ele_cart_id[1] + 1]);
              ++(shellcoordinates_numnodes[(ele_cart_id[1] + 1) * (numsubdivisions - 1)]);
            }
          }
          break;
        }
        default:
          FOUR_C_THROW(
              "Unknown element shape for a nurbs element or nurbs type not valid for turbulence "
              "calculation\n");
      }
    }

    //----------------------------------------------------------------------
    // add contributions from all processors, normalize

    std::vector<double> lnodeplanes(*nodeshells_);
    std::vector<double> lplanecoordinates(*shellcoordinates_);

    std::vector<int> lnodeshells_numnodes(nodeshells_numnodes);
    std::vector<int> lshellcoordinates_numnodes(shellcoordinates_numnodes);

    *nodeshells_ = Core::Communication::sum_all(lnodeplanes, discret_->get_comm());
    *shellcoordinates_ = Core::Communication::sum_all(lplanecoordinates, discret_->get_comm());

    nodeshells_numnodes = Core::Communication::sum_all(lnodeshells_numnodes, discret_->get_comm());
    shellcoordinates_numnodes =
        Core::Communication::sum_all(lshellcoordinates_numnodes, discret_->get_comm());

    {
      (*nodeshells_).resize(nele_x_mele_x_lele[1] + 1);
      (*shellcoordinates_).resize(nele_x_mele_x_lele[1] * (numsubdivisions - 1) + 1);

      for (unsigned rr = 0; rr < (*nodeshells_).size(); ++rr)
      {
        if (fabs(nodeshells_numnodes[rr]) < 1e-9)
        {
          FOUR_C_THROW("zero nodes in shell layer {}\n", rr);
        }

        (*nodeshells_)[rr] /= (double)(nodeshells_numnodes[rr]);
      }


      for (unsigned rr = 0; rr < (*shellcoordinates_).size(); ++rr)
      {
        if (fabs(shellcoordinates_numnodes[rr]) < 1e-9)
        {
          FOUR_C_THROW("zero nodes in sampling shell layer {}\n", rr);
        }

        (*shellcoordinates_)[rr] /= (double)(shellcoordinates_numnodes[rr]);
      }
    }
  }

  //----------------------------------------------------------------------
  // sort shellcoordinates and nodeshells
  {
    std::set<double, PlaneSortCriterion> shellset;

    std::vector<double>::iterator coord;
    std::set<double, PlaneSortCriterion>::iterator shell;

    {
      for (coord = (*nodeshells_).begin(); coord != (*nodeshells_).end(); ++coord)
      {
        shellset.insert(*coord);
      }

      int rr = 0;
      for (shell = shellset.begin(); shell != shellset.end(); ++shell)
      {
        (*nodeshells_)[rr] = *shell;
        ++rr;
      }
    }

    shellset.clear();

    {
      for (coord = (*shellcoordinates_).begin(); coord != (*shellcoordinates_).end(); ++coord)
      {
        shellset.insert(*coord);
      }

      int rr = 0;
      for (shell = shellset.begin(); shell != shellset.end(); ++shell)
      {
        (*shellcoordinates_)[rr] = *shell;
        ++rr;
      }
    }
  }

  //----------------------------------------------------------------------
  // allocate arrays for sums of in plane mean values

  int size = shellcoordinates_->size();

  // arrays for point based averaging
  // --------------------------------

  // first order moments
  pointsumu_ = std::make_shared<std::vector<double>>();
  pointsumu_->resize(size, 0.0);

  pointsumv_ = std::make_shared<std::vector<double>>();
  pointsumv_->resize(size, 0.0);

  pointsumw_ = std::make_shared<std::vector<double>>();
  pointsumw_->resize(size, 0.0);

  pointsump_ = std::make_shared<std::vector<double>>();
  pointsump_->resize(size, 0.0);

  // now the second order moments
  pointsumuu_ = std::make_shared<std::vector<double>>();
  pointsumuu_->resize(size, 0.0);

  pointsumvv_ = std::make_shared<std::vector<double>>();
  pointsumvv_->resize(size, 0.0);

  pointsumww_ = std::make_shared<std::vector<double>>();
  pointsumww_->resize(size, 0.0);

  pointsumpp_ = std::make_shared<std::vector<double>>();
  pointsumpp_->resize(size, 0.0);

  pointsumuv_ = std::make_shared<std::vector<double>>();
  pointsumuv_->resize(size, 0.0);

  pointsumuw_ = std::make_shared<std::vector<double>>();
  pointsumuw_->resize(size, 0.0);

  pointsumvw_ = std::make_shared<std::vector<double>>();
  pointsumvw_->resize(size, 0.0);

  if (withscatra_)
  {
    pointsumc_ = std::make_shared<std::vector<double>>();
    pointsumc_->resize(size, 0.0);

    pointsumcc_ = std::make_shared<std::vector<double>>();
    pointsumcc_->resize(size, 0.0);

    // pointsumphi_/pointsumphiphi_ are allocated in ApplyScatraResults()
  }

  //----------------------------------------------------------------------
  // initialise output

  std::shared_ptr<std::ofstream> log;

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::string s(statistics_outfilename_);
    s.append(".flow_statistics");

    log = std::make_shared<std::ofstream>(s.c_str(), std::ios::out);
    (*log) << "# Statistics for turbulent incompressible flow in a rotating cylinder (first- and "
              "second-order moments)\n\n";

    log->flush();
  }

  // clear statistics
  this->clear_statistics();

  return;
}  // TurbulenceStatisticsCcy::TurbulenceStatisticsCcy


/*----------------------------------------------------------------------*

       Compute the in-plane mean values of first and second order
       moments for velocities, pressure and Cs are added to global
                            'sum' vectors.

 -----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsCcy::do_time_sample(Core::LinAlg::Vector<double>& velnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> scanp,
    std::shared_ptr<Core::LinAlg::Vector<double>> fullphinp)
{
  // we have an additional sample
  numsamp_++;

  // meanvelnp is a refcount copy of velnp
  meanvelnp_->update(1.0, velnp, 0.0);

  if (withscatra_)
  {
    if (scanp != nullptr)
      meanscanp_->update(1.0, *scanp, 0.0);
    else
      FOUR_C_THROW("Vector scanp is nullptr");

    if (fullphinp != nullptr)
    {
      meanfullphinp_->update(1.0, *fullphinp, 0.0);
    }
    else
      FOUR_C_THROW("Vector fullphinp is nullptr");
  }

  //----------------------------------------------------------------------
  // loop planes and calculate pointwise means in each plane
  this->evaluate_pointwise_mean_values_in_planes();

  return;
}  // TurbulenceStatisticsCcy::DoTimeSample


/*----------------------------------------------------------------------*

          Compute in plane means of u,u^2 etc. (nodal quantities)

  ----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsCcy::evaluate_pointwise_mean_values_in_planes()
{
  const int numsubdivisions = 5;

  //----------------------------------------------------------------------
  // sort shellcoordinates and nodeshells
  std::map<double, int, PlaneSortCriterion> countpoints;
  std::map<double, double, PlaneSortCriterion> meanu;
  std::map<double, double, PlaneSortCriterion> meanv;
  std::map<double, double, PlaneSortCriterion> meanw;
  std::map<double, double, PlaneSortCriterion> meanp;
  std::map<double, double, PlaneSortCriterion> meanuu;
  std::map<double, double, PlaneSortCriterion> meanvv;
  std::map<double, double, PlaneSortCriterion> meanww;
  std::map<double, double, PlaneSortCriterion> meanpp;
  std::map<double, double, PlaneSortCriterion> meanuv;
  std::map<double, double, PlaneSortCriterion> meanuw;
  std::map<double, double, PlaneSortCriterion> meanvw;

  std::map<double, double, PlaneSortCriterion> meanc;
  std::map<double, double, PlaneSortCriterion> meancc;

  std::vector<std::map<double, double, PlaneSortCriterion>> meanphi(numscatradofpernode_);
  std::vector<std::map<double, double, PlaneSortCriterion>> meanphiphi(numscatradofpernode_);

  for (std::vector<double>::iterator coord = (*shellcoordinates_).begin();
      coord != (*shellcoordinates_).end(); ++coord)
  {
    double r = *coord;

    meanu.insert(std::pair<double, double>(r, 0.0));
    meanv.insert(std::pair<double, double>(r, 0.0));
    meanw.insert(std::pair<double, double>(r, 0.0));
    meanp.insert(std::pair<double, double>(r, 0.0));

    meanuu.insert(std::pair<double, double>(r, 0.0));
    meanvv.insert(std::pair<double, double>(r, 0.0));
    meanww.insert(std::pair<double, double>(r, 0.0));
    meanpp.insert(std::pair<double, double>(r, 0.0));
    meanuv.insert(std::pair<double, double>(r, 0.0));
    meanuw.insert(std::pair<double, double>(r, 0.0));
    meanvw.insert(std::pair<double, double>(r, 0.0));

    countpoints.insert(std::pair<double, int>(r, 0));

    if (withscatra_)
    {
      meanc.insert(std::pair<double, double>(r, 0.0));
      meancc.insert(std::pair<double, double>(r, 0.0));

      for (int k = 0; k < numscatradofpernode_; ++k)
      {
        meanphi[k].insert(std::pair<double, double>(r, 0.0));
        meanphiphi[k].insert(std::pair<double, double>(r, 0.0));
      }
    }
  }

  // try to cast discretisation to nurbs variant
  // this tells you what kind of computation of
  // samples is required
  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*discret_));

  if (nurbsdis == nullptr) FOUR_C_THROW("Oops. Your discretization is not a NurbsDiscretization.");

  nurbsdis->set_state("velnp", *meanvelnp_);

  Core::FE::Nurbs::NurbsDiscretization* scatranurbsdis(nullptr);
  if (withscatra_)
  {
    nurbsdis->set_state("scanp", *meanscanp_);

    scatranurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*scatradis_));
    if (scatranurbsdis == nullptr)
      FOUR_C_THROW("Oops. Your discretization is not a NurbsDiscretization.");

    if (meanfullphinp_ == nullptr)
      FOUR_C_THROW("std::shared_ptr is nullptr");
    else
      scatranurbsdis->set_state("phinp_for_statistics", *meanfullphinp_);

    if (not(scatranurbsdis->dof_row_map())->same_as(meanfullphinp_->get_map()))
    {
      scatranurbsdis->dof_row_map()->print(std::cout);
      meanfullphinp_->get_map().print(std::cout);
      FOUR_C_THROW("Global dof numbering in maps does not match");
    }
  }

  // get nurbs dis' knotvector sizes
  std::vector<int> n_x_m_x_l(nurbsdis->return_n_x_m_x_l(0));

  // get nurbs dis' element numbers
  std::vector<int> nele_x_mele_x_lele(nurbsdis->return_nele_x_mele_x_lele(0));

  // get the knotvector itself
  std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

  // get element map
  const Core::LinAlg::Map* elementmap = nurbsdis->element_row_map();

  // loop all available elements
  for (int iele = 0; iele < elementmap->num_my_elements(); ++iele)
  {
    // get element pointer
    Core::Elements::Element* const actele = nurbsdis->g_element(elementmap->gid(iele));

    // want to loop all control points of the element,
    // so get the number of points
    const int numnp = actele->num_node();

    // get the elements control points/nodes
    Core::Nodes::Node** nodes = actele->nodes();

    // acquire weights from nodes
    Core::LinAlg::SerialDenseVector weights(numnp);

    for (int inode = 0; inode < numnp; ++inode)
    {
      Core::FE::Nurbs::ControlPoint* cp =
          dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

      weights(inode) = cp->w();
    }
    // get gid, location in the patch
    int gid = actele->id();

    int patchid = 0;

    std::vector<int> ele_cart_id(3);
    knots->convert_ele_gid_to_knot_ids(gid, patchid, ele_cart_id);

    // access elements knot span
    std::vector<Core::LinAlg::SerialDenseVector> knots(3);
    bool zero_size = (*((*nurbsdis).get_knot_vector())).get_ele_knots(knots, actele->id());

    // zero sized elements have to be skipped
    if (zero_size)
    {
      continue;
    }

    // get shapefunctions, compute all visualisation point positions
    Core::LinAlg::SerialDenseVector nurbs_shape_funct(numnp);

    // extract local values from the global vectors
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;

    actele->location_vector(*nurbsdis, lm, lmowner, lmstride);

    // extract local values from global vector
    std::vector<double> myvelnp = Core::FE::extract_values(*(nurbsdis->get_state("velnp")), lm);

    // create Matrix objects
    Core::LinAlg::Matrix<3, 27> evelnp;
    Core::LinAlg::Matrix<27, 1> eprenp;

    // insert velocity  into element array
    for (int i = 0; i < 27; ++i)
    {
      const int fi = 4 * i;

      evelnp(0, i) = myvelnp[fi];
      evelnp(1, i) = myvelnp[1 + fi];
      evelnp(2, i) = myvelnp[2 + fi];

      eprenp(i) = myvelnp[3 + fi];
    }

    Core::LinAlg::Matrix<1, 27> escanp(Core::LinAlg::Initialization::zero);

    //! scalar at t_(n+1) or t_(n+alpha_F)
    const int nen = 27;  // only quadratic nurbs elements are supported!!
    std::vector<Core::LinAlg::Matrix<nen, 1>> ephinp_(numscatradofpernode_);

    if (withscatra_)
    {
      // extract local values from global vector
      std::vector<double> myscanp = Core::FE::extract_values(*(nurbsdis->get_state("scanp")), lm);

      // insert data into element array (scalar field is stored at pressure dofs)
      for (int i = 0; i < 27; ++i)
      {
        const int fi = 4 * i;
        escanp(0, i) = myscanp[3 + fi];
      }

      // get pointer to corresponding scatra element with identical global id
      Core::Elements::Element* const actscatraele = scatranurbsdis->g_element(gid);
      if (actscatraele == nullptr)
        FOUR_C_THROW("could not access transport element with gid {}", gid);

      // extract local values from the global vectors
      std::vector<int> scatralm;
      std::vector<int> scatralmowner;
      std::vector<int> scatralmstride;

      actscatraele->location_vector(*scatranurbsdis, scatralm, scatralmowner, scatralmstride);

      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp =
          scatranurbsdis->get_state("phinp_for_statistics");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp' for statistics");
      std::vector<double> myphinp = Core::FE::extract_values(*phinp, scatralm);

      // fill all element arrays
      for (int i = 0; i < nen; ++i)
      {
        for (int k = 0; k < numscatradofpernode_; ++k)
        {
          // split for each transported scalar, insert into element arrays
          ephinp_[k](i, 0) = myphinp[k + (i * numscatradofpernode_)];
        }
      }  // for i
    }

    switch (actele->shape())
    {
      case Core::FE::CellType::nurbs27:
      {
        Core::LinAlg::Matrix<3, 1> vel;

        // element local point position
        Core::LinAlg::SerialDenseVector uv(3);

        {
          // standard

          //               v
          //              /
          //  w  7       /   8
          //  ^   +---------+
          //  |  /         /|
          //  | /         / |
          // 5|/        6/  |
          //  +---------+   |
          //  |         |   |
          //  |         |   +
          //  |         |  / 4
          //  |         | /
          //  |         |/
          //  +---------+ ----->u
          // 1           2
          // use r-coordinate of point 1 and 8
          // temporary x vector
          std::vector<double> x(3);

          // point 2
          uv(0) = 1.0;
          uv(1) = -1.0;
          uv(2) = -1.0;
          Core::FE::Nurbs::nurbs_get_3d_funct(
              nurbs_shape_funct, uv, knots, weights, actele->shape());
          for (int isd = 0; isd < 3; ++isd)
          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
            }
            x[isd] = val;
          }

          const double r = sqrt(x[0] * x[0] + x[1] * x[1]);

          {
            double val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += nurbs_shape_funct(inode) * evelnp(0, inode);
            }
            vel(0) = val;

            val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += nurbs_shape_funct(inode) * evelnp(1, inode);
            }
            vel(1) = val;

            val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += nurbs_shape_funct(inode) * evelnp(2, inode);
            }
            vel(2) = val;

            val = 0;
            for (int inode = 0; inode < numnp; ++inode)
            {
              val += nurbs_shape_funct(inode) * eprenp(inode);
            }
            meanp[r] += val;
            meanpp[r] += val * val;

            if (withscatra_)
            {
              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * escanp(inode);
              }
              meanc[r] += val;
              meancc[r] += val * val;

              for (int k = 0; k < numscatradofpernode_; ++k)
              {
                val = 0;
                for (int inode = 0; inode < numnp; ++inode)
                {
                  val += nurbs_shape_funct(inode) * ((ephinp_[k])(inode));
                }
                (meanphi[k])[r] += val;
                (meanphiphi[k])[r] += val * val;
              }
            }

            std::map<double, int, PlaneSortCriterion>::iterator shell = countpoints.find(r);
            if (shell == countpoints.end())
            {
              FOUR_C_THROW("radial coordinate {:12.5e} was not map\n", r);
            }
            else
            {
              shell->second += 1;
            }

            double uphi = 1.0 / r * (x[0] * vel(1) - x[1] * vel(0));
            double ur = 1.0 / r * (x[0] * vel(0) + x[1] * vel(1));

            meanu[r] += uphi;
            meanv[r] += ur;
            meanw[r] += vel(2);

            meanuu[r] += uphi * uphi;
            meanvv[r] += ur * ur;
            meanww[r] += vel(2) * vel(2);

            meanuv[r] += uphi * ur;
            meanuw[r] += uphi * vel(2);
            meanvw[r] += ur * vel(2);
          }

          for (int rr = 1; rr < numsubdivisions - 1; ++rr)
          {
            uv(1) += 2.0 / (numsubdivisions - 1);

            Core::FE::Nurbs::nurbs_get_3d_funct(
                nurbs_shape_funct, uv, knots, weights, actele->shape());
            for (int isd = 0; isd < 3; ++isd)
            {
              double val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
              }
              x[isd] = val;
            }

            const double r = sqrt(x[0] * x[0] + x[1] * x[1]);

            {
              double val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * evelnp(0, inode);
              }
              vel(0) = val;

              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * evelnp(1, inode);
              }
              vel(1) = val;

              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * evelnp(2, inode);
              }
              vel(2) = val;

              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * eprenp(inode);
              }
              meanp[r] += val;
              meanpp[r] += val * val;

              if (withscatra_)
              {
                val = 0;
                for (int inode = 0; inode < numnp; ++inode)
                {
                  val += nurbs_shape_funct(inode) * escanp(inode);
                }
                meanc[r] += val;
                meancc[r] += val * val;

                for (int k = 0; k < numscatradofpernode_; ++k)
                {
                  val = 0;
                  for (int inode = 0; inode < numnp; ++inode)
                  {
                    val += nurbs_shape_funct(inode) * ((ephinp_[k])(inode));
                  }
                  (meanphi[k])[r] += val;
                  (meanphiphi[k])[r] += val * val;
                }
              }

              std::map<double, int, PlaneSortCriterion>::iterator shell = countpoints.find(r);
              if (shell == countpoints.end())
              {
                FOUR_C_THROW("radial coordinate {:12.5e} was not map\n", r);
              }
              else
              {
                shell->second += 1;
              }

              double uphi = 1.0 / r * (x[0] * vel(1) - x[1] * vel(0));
              double ur = 1.0 / r * (x[0] * vel(0) + x[1] * vel(1));

              meanu[r] += uphi;
              meanv[r] += ur;
              meanw[r] += vel(2);

              meanuu[r] += uphi * uphi;
              meanvv[r] += ur * ur;
              meanww[r] += vel(2) * vel(2);

              meanuv[r] += uphi * ur;
              meanuw[r] += uphi * vel(2);
              meanvw[r] += ur * vel(2);
            }
          }

          // set upper point of element, too (only for last layer)
          if (ele_cart_id[1] + 1 == nele_x_mele_x_lele[1])
          {
            // point 4
            uv(0) = 1.0;
            uv(1) = 1.0;
            uv(2) = -1.0;
            Core::FE::Nurbs::nurbs_get_3d_funct(
                nurbs_shape_funct, uv, knots, weights, actele->shape());
            for (int isd = 0; isd < 3; ++isd)
            {
              double val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += (((nodes[inode])->x())[isd]) * nurbs_shape_funct(inode);
              }
              x[isd] = val;
            }


            const double r = sqrt(x[0] * x[0] + x[1] * x[1]);

            {
              double val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * evelnp(0, inode);
              }
              vel(0) = val;

              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * evelnp(1, inode);
              }
              vel(1) = val;

              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * evelnp(2, inode);
              }
              vel(2) = val;

              val = 0;
              for (int inode = 0; inode < numnp; ++inode)
              {
                val += nurbs_shape_funct(inode) * eprenp(inode);
              }
              meanp[r] += val;
              meanpp[r] += val * val;

              if (withscatra_)
              {
                val = 0;
                for (int inode = 0; inode < numnp; ++inode)
                {
                  val += nurbs_shape_funct(inode) * escanp(inode);
                }
                meanc[r] += val;
                meancc[r] += val * val;

                for (int k = 0; k < numscatradofpernode_; ++k)
                {
                  val = 0;
                  for (int inode = 0; inode < numnp; ++inode)
                  {
                    val += nurbs_shape_funct(inode) * ((ephinp_[k])(inode));
                  }
                  (meanphi[k])[r] += val;
                  (meanphiphi[k])[r] += val * val;
                }
              }

              std::map<double, int, PlaneSortCriterion>::iterator shell = countpoints.find(r);
              if (shell == countpoints.end())
              {
                FOUR_C_THROW("radial coordinate {:12.5e} was not map\n", r);
              }
              else
              {
                shell->second += 1;
              }

              double uphi = 1.0 / r * (x[0] * vel(1) - x[1] * vel(0));
              double ur = 1.0 / r * (x[0] * vel(0) + x[1] * vel(1));

              meanu[r] += uphi;
              meanv[r] += ur;
              meanw[r] += vel(2);

              meanuu[r] += uphi * uphi;
              meanvv[r] += ur * ur;
              meanww[r] += vel(2) * vel(2);

              meanuv[r] += uphi * ur;
              meanuw[r] += uphi * vel(2);
              meanvw[r] += ur * vel(2);
            }
          }
        }
        break;
      }
      default:
        FOUR_C_THROW(
            "Unknown element shape for a nurbs element or nurbs type not valid for turbulence "
            "calculation\n");
    }
  }  // end element loop

  // clean up
  nurbsdis->clear_state();
  if (scatranurbsdis != nullptr)
  {
    scatranurbsdis->clear_state();
  }

  // communicate results among processors
  int size = countpoints.size();
  int rr;

  // collect number of samples
  std::vector<int> lpointcount;
  std::vector<int> pointcount(size);

  for (std::map<double, int, PlaneSortCriterion>::iterator shell = countpoints.begin();
      shell != countpoints.end(); ++shell)
  {
    lpointcount.push_back(shell->second);
  }
  pointcount = Core::Communication::sum_all(lpointcount, discret_->get_comm());

  // collect number of samples
  std::vector<double> lmeanu;
  std::vector<double> lmeanv;
  std::vector<double> lmeanw;
  std::vector<double> lmeanp;

  std::vector<double> lmeanuu;
  std::vector<double> lmeanvv;
  std::vector<double> lmeanww;
  std::vector<double> lmeanpp;

  std::vector<double> lmeanuv;
  std::vector<double> lmeanuw;
  std::vector<double> lmeanvw;

  std::vector<double> lmeanc;
  std::vector<double> lmeancc;

  std::vector<double> lmeanphi;
  std::vector<double> lmeanphiphi;

  std::vector<double> gmeanu(size);
  std::vector<double> gmeanv(size);
  std::vector<double> gmeanw(size);
  std::vector<double> gmeanp(size);

  std::vector<double> gmeanuu(size);
  std::vector<double> gmeanvv(size);
  std::vector<double> gmeanww(size);
  std::vector<double> gmeanpp(size);

  std::vector<double> gmeanuv(size);
  std::vector<double> gmeanuw(size);
  std::vector<double> gmeanvw(size);

  std::vector<double> gmeanc(size);
  std::vector<double> gmeancc(size);

  std::vector<double> gmeanphi(size);
  std::vector<double> gmeanphiphi(size);

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanu.begin();
      shell != meanu.end(); ++shell)
  {
    if (fabs(pointcount[rr]) < 1e-6)
    {
      FOUR_C_THROW("zero pointcount during computation of averages, layer {}\n", rr);
    }

    lmeanu.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanu = Core::Communication::sum_all(lmeanu, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanv.begin();
      shell != meanv.end(); ++shell)
  {
    lmeanv.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanv = Core::Communication::sum_all(lmeanv, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanw.begin();
      shell != meanw.end(); ++shell)
  {
    lmeanw.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanw = Core::Communication::sum_all(lmeanw, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanp.begin();
      shell != meanp.end(); ++shell)
  {
    lmeanp.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanp = Core::Communication::sum_all(lmeanp, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanuu.begin();
      shell != meanuu.end(); ++shell)
  {
    lmeanuu.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanuu = Core::Communication::sum_all(lmeanuu, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanvv.begin();
      shell != meanvv.end(); ++shell)
  {
    lmeanvv.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanvv = Core::Communication::sum_all(lmeanvv, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanww.begin();
      shell != meanww.end(); ++shell)
  {
    lmeanww.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanww = Core::Communication::sum_all(lmeanww, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanpp.begin();
      shell != meanpp.end(); ++shell)
  {
    lmeanpp.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanpp = Core::Communication::sum_all(lmeanpp, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanuv.begin();
      shell != meanuv.end(); ++shell)
  {
    lmeanuv.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanuv = Core::Communication::sum_all(lmeanuv, discret_->get_comm());

  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanuw.begin();
      shell != meanuw.end(); ++shell)
  {
    lmeanuw.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanuw = Core::Communication::sum_all(lmeanuw, discret_->get_comm());


  rr = 0;
  for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanvw.begin();
      shell != meanvw.end(); ++shell)
  {
    lmeanvw.push_back(shell->second / pointcount[rr]);
    ++rr;
  }
  gmeanvw = Core::Communication::sum_all(lmeanvw, discret_->get_comm());

  for (int mm = 0; mm < size; ++mm)
  {
    (*pointsumu_)[mm] += gmeanu[mm];
    (*pointsumv_)[mm] += gmeanv[mm];
    (*pointsumw_)[mm] += gmeanw[mm];
    (*pointsump_)[mm] += gmeanp[mm];

    (*pointsumuu_)[mm] += gmeanuu[mm];
    (*pointsumvv_)[mm] += gmeanvv[mm];
    (*pointsumww_)[mm] += gmeanww[mm];
    (*pointsumpp_)[mm] += gmeanpp[mm];

    (*pointsumuv_)[mm] += gmeanuv[mm];
    (*pointsumuw_)[mm] += gmeanuw[mm];
    (*pointsumvw_)[mm] += gmeanvw[mm];
  }

  if (withscatra_)
  {
    rr = 0;
    for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanc.begin();
        shell != meanc.end(); ++shell)
    {
      lmeanc.push_back(shell->second / pointcount[rr]);
      ++rr;
    }
    gmeanc = Core::Communication::sum_all(lmeanc, discret_->get_comm());

    rr = 0;
    for (std::map<double, double, PlaneSortCriterion>::iterator shell = meancc.begin();
        shell != meancc.end(); ++shell)
    {
      lmeancc.push_back(shell->second / pointcount[rr]);
      ++rr;
    }
    gmeancc = Core::Communication::sum_all(lmeancc, discret_->get_comm());

    for (int mm = 0; mm < size; ++mm)
    {
      (*pointsumc_)[mm] += gmeanc[mm];
      (*pointsumcc_)[mm] += gmeancc[mm];
    }

    // safety checks
    if ((size != pointsumphi_->numRows()) or (size != pointsumphiphi_->numRows()))
      FOUR_C_THROW("Size mismatch: size = {} <-> M = {}", size, pointsumphi_->numRows());
    if ((numscatradofpernode_ != pointsumphi_->numCols()) or
        (numscatradofpernode_ != pointsumphiphi_->numCols()))
      FOUR_C_THROW(
          "Size mismatch: numdof = {} <-> N = {}", numscatradofpernode_, pointsumphi_->numCols());

    // loop all available scatra fields
    for (int k = 0; k < numscatradofpernode_; ++k)
    {
      // sequential reuse of this vector + push back actions below require this:
      lmeanphi.resize(0);
      lmeanphiphi.resize(0);

      rr = 0;
      for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanphi[k].begin();
          shell != meanphi[k].end(); ++shell)
      {
        lmeanphi.push_back(shell->second / pointcount[rr]);
        gmeanphi[rr] = 0.0;  // initialize due to sequential reuse of this vector
        ++rr;
      }
      gmeanphi = Core::Communication::sum_all(lmeanphi, discret_->get_comm());

      rr = 0;
      for (std::map<double, double, PlaneSortCriterion>::iterator shell = meanphiphi[k].begin();
          shell != meanphiphi[k].end(); ++shell)
      {
        lmeanphiphi.push_back(shell->second / pointcount[rr]);
        gmeanphiphi[rr] = 0.0;  // initialize due to sequential reuse of this vector
        ++rr;
      }
      gmeanphiphi = Core::Communication::sum_all(lmeanphiphi, discret_->get_comm());

      // insert values for scalar field k
      for (int mm = 0; mm < size; ++mm)
      {
        (*pointsumphi_)(mm, k) += gmeanphi[mm];
        (*pointsumphiphi_)(mm, k) += gmeanphiphi[mm];
      }

    }  // loop over scalar fields k


  }  // if(withscatra)

  return;
}  // TurbulenceStatisticsCcy::evaluate_pointwise_mean_values_in_planes()


/*----------------------------------------------------------------------*

       Compute a time average of the mean values over all steps
          since the last output. Dump the result to file.

  ----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsCcy::time_average_means_and_output_of_statistics(int step)
{
  if (numsamp_ == 0)
  {
    FOUR_C_THROW("No samples to do time average");
  }

  //----------------------------------------------------------------------
  // output to log-file
  std::shared_ptr<std::ofstream> log;
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::string s(statistics_outfilename_);
    s.append(".flow_statistics");

    log = std::make_shared<std::ofstream>(s.c_str(), std::ios::app);
    (*log) << "\n\n\n";
    (*log) << "# Statistics record " << countrecord_;
    (*log) << " (Steps " << step - numsamp_ + 1 << "--" << step << ")\n";

    (*log) << "#|-------------------";
    (*log) << "---------------------------------------------------";
    (*log) << "--point based (interpolated)-------------------------";
    (*log) << "----------------------------------------------------------|";
    (*log) << "\n";

    (*log) << "#       y        ";
    (*log) << "      u_theta     ";
    (*log) << "       u_r       ";
    (*log) << "       u_z       ";
    (*log) << "        p        ";
    (*log) << " u_theta*u_theta ";
    (*log) << "     u_r*u_r     ";
    (*log) << "     u_z*u_z     ";
    (*log) << "       p*p       ";
    (*log) << "   u_theta*u_r   ";
    (*log) << "   u_theta*u_z   ";
    (*log) << "     u_r*u_z     ";
    if (withscatra_)
    {
      (*log) << "          c          ";
      (*log) << "         c*c         ";

      for (int k = 0; k < numscatradofpernode_; k++)
      {
        (*log) << "          c" << k + 1 << "          ";
        (*log) << "       c" << k + 1 << "*c" << k + 1 << "        ";
      }
    }
    (*log) << "\n";
    (*log) << std::scientific;
    for (unsigned i = 0; i < shellcoordinates_->size(); ++i)
    {
      // y and y+
      (*log) << " " << std::setw(14) << std::setprecision(7) << (*shellcoordinates_)[i];

      // pointwise means
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumu_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumv_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumw_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsump_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumuu_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumvv_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumww_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumpp_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumuv_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumuw_)[i] / numsamp_;
      (*log) << "    " << std::setw(13) << std::setprecision(6) << (*pointsumvw_)[i] / numsamp_;
      if (withscatra_)
      {
        (*log) << "    " << std::setw(17) << std::setprecision(10) << (*pointsumc_)[i] / numsamp_;
        (*log) << "    " << std::setw(17) << std::setprecision(10) << (*pointsumcc_)[i] / numsamp_;

        for (int k = 0; k < numscatradofpernode_; k++)
        {
          (*log) << "    " << std::setw(17) << std::setprecision(10)
                 << ((*pointsumphi_)(i, k)) / numsamp_;
          (*log) << "    " << std::setw(17) << std::setprecision(10)
                 << ((*pointsumphiphi_)(i, k)) / numsamp_;
        }
      }
      (*log) << "\n";
    }
    log->flush();
  }  // end myrank 0


  // log was written, so increase counter for records
  countrecord_++;

  return;

}  // TurbulenceStatisticsCcy::time_average_means_and_output_of_statistics


/*----------------------------------------------------------------------*

                  Reset sums and number of samples to 0

  ----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsCcy::clear_statistics()
{
  // reset the number of samples
  numsamp_ = 0;

  // reset integral and pointwise averages
  for (unsigned i = 0; i < shellcoordinates_->size(); ++i)
  {
    (*pointsumu_)[i] = 0;
    (*pointsumv_)[i] = 0;
    (*pointsumw_)[i] = 0;
    (*pointsump_)[i] = 0;

    (*pointsumuu_)[i] = 0;
    (*pointsumvv_)[i] = 0;
    (*pointsumww_)[i] = 0;
    (*pointsumpp_)[i] = 0;

    (*pointsumuv_)[i] = 0;
    (*pointsumuw_)[i] = 0;
    (*pointsumvw_)[i] = 0;
  }

  meanvelnp_->put_scalar(0.0);

  if (withscatra_)
  {
    // reset integral and pointwise averages
    for (size_t i = 0; i < shellcoordinates_->size(); ++i)
    {
      (*pointsumc_)[i] = 0.0;
      (*pointsumcc_)[i] = 0.0;
    }

    if (meanscanp_ == nullptr)
      FOUR_C_THROW("meanscanp_ is nullptr");
    else
      meanscanp_->put_scalar(0.0);

    if (meanfullphinp_ != nullptr)
    {
      meanfullphinp_->put_scalar(0.0);

      // ToDo Is is a good way to initialize everything to zero??
      // Use Shape() instead???
      pointsumphi_->putScalar(0.0);
      pointsumphiphi_->putScalar(0.0);
    }
  }

  return;
}  // TurbulenceStatisticsCcy::ClearStatistics


/*----------------------------------------------------------------------

Add results from scalar transport fields to statistics

----------------------------------------------------------------------*/
void FLD::TurbulenceStatisticsCcy::add_scatra_results(
    std::shared_ptr<Core::FE::Discretization> scatradis, Core::LinAlg::Vector<double>& phinp)
{
  if (withscatra_)
  {
    if (scatradis == nullptr)
      FOUR_C_THROW("Halleluja.");
    else
      scatradis_ = scatradis;  // now we have access

    // we do not have to cast to a NURBSDiscretization here!
    meanfullphinp_ = Core::LinAlg::create_vector(*(scatradis_->dof_row_map()), true);
    numscatradofpernode_ = scatradis_->num_dof(scatradis_->l_row_node(0));

    // now we know about the number of scatra dofs and can allocate:
    int size = shellcoordinates_->size();
    pointsumphi_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>(size, numscatradofpernode_);
    pointsumphiphi_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>(size, numscatradofpernode_);

    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    {
      std::cout << std::endl
                << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
      std::cout << "TurbulenceStatisticsCcy:    added access to ScaTra results" << std::endl;
      std::cout << " | nodeshellsize       = " << nodeshells_->size() << std::endl
                << " | numshellcoordinates = " << size << " (4 subdivisions per element)"
                << std::endl
                << " | numscatradofpernode = " << numscatradofpernode_ << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl
                << std::endl;
    }
  }
  else
  {
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    {
      std::cout << "------------------------------------------------------------" << std::endl;
      std::cout << "TurbulenceStatisticsCcy: NO access to ScaTra results !" << std::endl;
      std::cout << "------------------------------------------------------------" << std::endl;
    }
  }

  return;
}  // FLD::TurbulenceStatisticsCcy::AddScaTraResults

FOUR_C_NAMESPACE_CLOSE
