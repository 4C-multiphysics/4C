// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMCONTACT_BEAM3CONTACT_HPP
#define FOUR_C_BEAMCONTACT_BEAM3CONTACT_HPP

#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beamcontact_beam3contactinterface.hpp"
#include "4C_beamcontact_beam3contactvariables.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_tangentsmoothing.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_contact_input.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*!
   \brief contact element for contact between two 3D beam elements

   Refer also to the Semesterarbeit of Matthias Mayr, 2010

   */

  template <const int numnodes, const int numnodalvalues>
  class Beam3contact : public Beam3contactinterface
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}

    //! @name Constructors and destructors and related methods
    /*!
    \brief Standard Constructor
    \param pdiscret (in): the problem discretization
    \param cdiscret (in): the beam contact discretization
    \param dofoffset (in): offset of dof between pdiscret and cdiscret
    \param element1 (in): first element of contact pair
    \param element2 (in): second element of contact pair
    \param ele1pos (in): nodal coordinates of first element
    \param ele2pos (in): nodal coordinates of second element
    */
    Beam3contact(const Core::FE::Discretization& pdiscret, const Core::FE::Discretization& cdiscret,
        const std::map<int, int>& dofoffsetmap, Core::Elements::Element* element1,
        Core::Elements::Element* element2, Teuchos::ParameterList& beamcontactparams);


    //@}


    //! @name Access methods
    /*!
    \brief Get problem discretization
    */
    inline const Core::FE::Discretization& problem_discret() const override { return pdiscret_; };

    /*!
    \brief Get beam contact discretization
    */
    inline const Core::FE::Discretization& contact_discret() const override { return cdiscret_; };

    /*!
    \brief Get offset of dofs between cdiscret and pdiscret
    */
    inline const std::map<int, int>& dof_offset() const override { return dofoffsetmap_; };

    /*!
    \brief Get first element
    */
    inline const Core::Elements::Element* element1() override { return element1_; };

    /*!
    \brief Get first element
    */
    inline const Core::Elements::Element* element2() override { return element2_; };

    /*!
    \brief Get number of standard large angle/small angle/endpoint contact points on this element
    pair
    */
    int get_num_cps() override { return cpvariables_.size(); };

    int get_num_gps() override { return gpvariables_.size(); };

    int get_num_eps() override { return epvariables_.size(); };

    /*!
    \brief Get vector of type declarations (0=closest point contact, 1=gauss point contact, 2= end
    point contact) of all contact pairs
    */
    std::vector<int> get_contact_type() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<int> types(size1 + size2 + size3, 0);

      for (int i = 0; i < size1; i++)
      {
        types[i] = 0;
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        types[i] = 1;
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        types[i] = 2;
      }

      return types;
    };

    /*!
    \brief Get vector of all gaps of this contact pair
    */
    std::vector<double> get_gap() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<double> gaps(size1 + size2 + size3, 0.0);

      for (int i = 0; i < size1; i++)
      {
        gaps[i] = Core::FADUtils::cast_to_double(cpvariables_[i]->get_gap());
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        gaps[i] = Core::FADUtils::cast_to_double(gpvariables_[i - size1]->get_gap());
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        gaps[i] = Core::FADUtils::cast_to_double(epvariables_[i - size1 - size2]->get_gap());
      }

      return gaps;
    };

    /*!
    \brief Get vector of all contact forces of this contact pair
    */
    std::vector<double> get_contact_force() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<double> f(size1 + size2 + size3, 0.0);

      for (int i = 0; i < size1; i++)
      {
        f[i] = Core::FADUtils::cast_to_double(
            cpvariables_[i]->getfp() * cpvariables_[i]->get_p_pfac());
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        f[i] = Core::FADUtils::cast_to_double(
            gpvariables_[i - size1]->getfp() * gpvariables_[i - size1]->get_p_pfac());
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        f[i] = Core::FADUtils::cast_to_double(epvariables_[i - size1 - size2]->getfp() *
                                              epvariables_[i - size1 - size2]->get_p_pfac());
      }

      return f;
    };

    /*!
    \brief Get vector of all contact angles of this contact pair
    */
    std::vector<double> get_contact_angle() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<double> angles(size1 + size2 + size3, 0.0);

      for (int i = 0; i < size1; i++)
      {
        angles[i] = Core::FADUtils::cast_to_double(cpvariables_[i]->get_angle());
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        angles[i] = Core::FADUtils::cast_to_double(gpvariables_[i - size1]->get_angle());
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        angles[i] = Core::FADUtils::cast_to_double(epvariables_[i - size1 - size2]->get_angle());
      }

      return angles;
    };

    /*!
    \brief Get vector of all closest points of this contact pair
    */
    std::vector<std::pair<double, double>> get_closest_point() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<std::pair<double, double>> cps(size1 + size2 + size3, std::make_pair(0.0, 0.0));

      for (int i = 0; i < size1; i++)
      {
        double xi = Core::FADUtils::cast_to_double(cpvariables_[i]->get_cp().first);
        double eta = Core::FADUtils::cast_to_double(cpvariables_[i]->get_cp().second);
        cps[i] = std::make_pair(xi, eta);
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        double xi = Core::FADUtils::cast_to_double(gpvariables_[i - size1]->get_cp().first);
        double eta = Core::FADUtils::cast_to_double(gpvariables_[i - size1]->get_cp().second);
        cps[i] = std::make_pair(xi, eta);
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        double xi = Core::FADUtils::cast_to_double(epvariables_[i - size1 - size2]->get_cp().first);
        double eta =
            Core::FADUtils::cast_to_double(epvariables_[i - size1 - size2]->get_cp().second);
        cps[i] = std::make_pair(xi, eta);
      }

      return cps;
    };

    /*!
    \brief Return number of individual contact segments on element pair
    */
    std::pair<int, int> get_num_segments() override { return std::make_pair(numseg1_, numseg2_); };

    /*!
    \brief Return ids of active segments
    */
    std::vector<std::pair<int, int>> get_segment_ids() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<std::pair<int, int>> ids(size1 + size2 + size3, std::make_pair(1, 1));

      for (int i = 0; i < size1; i++)
      {
        ids[i] = cpvariables_[i]->get_seg_ids();
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        ids[i] = gpvariables_[i - size1]->get_seg_ids();
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        ids[i] = epvariables_[i - size1 - size2]->get_seg_ids();
      }

      return ids;
    };

    /*!
    \brief Get flag indicating whether contact is active (true) or inactive (false)
    */
    bool get_contact_flag() override
    {
      // The element pair is assumed to be active when we have at least one active contact point
      return (cpvariables_.size() + gpvariables_.size() + epvariables_.size());
    };

    /*!
    \brief Get coordinates of contact point of element1
    */
    std::vector<Core::LinAlg::Matrix<3, 1>> get_x1() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<Core::LinAlg::Matrix<3, 1>> r1(
          size1 + size2 + size3, Core::LinAlg::Matrix<3, 1>(Core::LinAlg::Initialization::zero));

      for (int i = 0; i < size1; i++)
      {
        TYPE eta1 = cpvariables_[i]->get_cp().first;
        for (int j = 0; j < 3; j++)
          r1[i](j) = Core::FADUtils::cast_to_double(r(eta1, element1_)(j));
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        TYPE eta1 = gpvariables_[i - size1]->get_cp().first;
        for (int j = 0; j < 3; j++)
          r1[i](j) = Core::FADUtils::cast_to_double(r(eta1, element1_)(j));
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        TYPE eta1 = epvariables_[i - size1 - size2]->get_cp().first;
        for (int j = 0; j < 3; j++)
          r1[i](j) = Core::FADUtils::cast_to_double(r(eta1, element1_)(j));
      }

      return r1;
    };

    /*!
    \brief Get coordinates of contact point of element2
    */
    std::vector<Core::LinAlg::Matrix<3, 1>> get_x2() override
    {
      int size1 = cpvariables_.size();
      int size2 = gpvariables_.size();
      int size3 = epvariables_.size();
      std::vector<Core::LinAlg::Matrix<3, 1>> r2(
          size1 + size2 + size3, Core::LinAlg::Matrix<3, 1>(Core::LinAlg::Initialization::zero));

      for (int i = 0; i < size1; i++)
      {
        TYPE eta2 = cpvariables_[i]->get_cp().second;
        for (int j = 0; j < 3; j++)
          r2[i](j) = Core::FADUtils::cast_to_double(r(eta2, element2_)(j));
      }

      for (int i = size1; i < size2 + size1; i++)
      {
        TYPE eta2 = gpvariables_[i - size1]->get_cp().second;
        for (int j = 0; j < 3; j++)
          r2[i](j) = Core::FADUtils::cast_to_double(r(eta2, element2_)(j));
      }

      for (int i = size1 + size2; i < size1 + size2 + size3; i++)
      {
        TYPE eta2 = epvariables_[i - size1 - size2]->get_cp().second;
        for (int j = 0; j < 3; j++)
          r2[i](j) = Core::FADUtils::cast_to_double(r(eta2, element2_)(j));
      }

      return r2;
    };


    // TODO:
    /*!
    \brief Get normal vector
    */
    Core::LinAlg::SerialDenseVector get_normal() override
    {
      Core::LinAlg::SerialDenseVector normal(3);

      for (int i = 0; i < 3; i++) normal(i) = 0.0;

      return normal;
    }

    /*!
    \brief Get flag indicating whether the nodal values of one element had been shifted due to r1=r2
           Since this is only possible for beam3contactnew elements but not for beam3contact
    elements we always return false within this class.
    */
    bool get_shift_status() override { return false; };

    /*!
      \Check, if there is a difference between the result of the new and old gap definition, i.e. if
      the beams centerlines have already crossed or not. Since this is only possible for
      beam3contactnew elements but not for beam3contact elements we always return false within this
      class.
    */
    bool get_new_gap_status() override { return false; };
    //@}

    /*!
      \Get energy of penalty contact.
    */
    double get_energy() override
    {
      if (Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_lp and
          Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_qp and
          Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_lpqp)
        FOUR_C_THROW("Contact Energy calculation not implemented for the chosen penalty law!");


      double energy = 0.0;

      for (int i = 0; i < (int)cpvariables_.size(); i++)
      {
        double ppfac = Core::FADUtils::cast_to_double(cpvariables_[i]->get_p_pfac());
        double e = -cpvariables_[i]->get_integrated_energy();
        energy += ppfac * e;
      }

      for (int i = 0; i < (int)gpvariables_.size(); i++)
      {
        double ppfac = Core::FADUtils::cast_to_double(gpvariables_[i]->get_p_pfac());
        double e = -gpvariables_[i]->get_integrated_energy();
        energy += ppfac * e;
      }

      for (int i = 0; i < (int)epvariables_.size(); i++)
      {
        double ppfac = Core::FADUtils::cast_to_double(epvariables_[i]->get_p_pfac());
        double e = -epvariables_[i]->get_integrated_energy();
        energy += ppfac * e;
      }

      return energy;
    };

    /*!
      \Get energy of perp penalty contact without transition factor contribution.
    */
    double get_unscaled_perp_energy() override
    {
      if (Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_lp and
          Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_qp and
          Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_lpqp)
        FOUR_C_THROW("Contact Energy calculation not implemented for the chosen penalty law!");


      double energy = 0.0;

      for (int i = 0; i < (int)cpvariables_.size(); i++)
      {
        double e = -cpvariables_[i]->get_integrated_energy();
        energy += e;
      }

      return energy;
    };

    /*!
      \Get energy of parallel penalty contact without transition factor contribution.
    */
    double get_unscaled_parallel_energy() override
    {
      if (Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_lp and
          Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_qp and
          Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(bcparams_, "BEAMS_PENALTYLAW") !=
              BeamContact::pl_lpqp)
        FOUR_C_THROW("Contact Energy calculation not implemented for the chosen penalty law!");

      double energy = 0.0;

      for (int i = 0; i < (int)gpvariables_.size(); i++)
      {
        double e = -gpvariables_[i]->get_integrated_energy();
        energy += e;
      }

      return energy;
    };

    // TODO
    /*!
      \We don't need this method for beam3contact elements!
    */
    Core::LinAlg::Matrix<3, 1, TYPE>* get_normal_old() override { return nullptr; };

    // TODO
    /*!
      \We don't need this method for beam3contact elements!
    */
    bool first_time_step() override { return false; };
    //@}

    //! @name Public evaluation methods
    /*!
    \brief Evaluate this contact element pair
    */
    bool evaluate(Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector<double>& fint,
        const double& pp,
        std::map<std::pair<int, int>, std::shared_ptr<Beam3contactinterface>>& contactpairmap,
        Teuchos::ParameterList& timeintparams, bool fdcheck = false) override;

    /*!
    \brief Change the sign of the normal vector: This has to be done at the end of a time step when
    the remaining penetration is larger that the sum of the beam radii (R1+R2). Otherwise, the beams
    could cross in the next time step when the new gap function definition (ngf_=true) for slender
    beams is applied!
    */
    void invert_normal() override { FOUR_C_THROW("Function not implemented!"); };

    // TODO
    /*!
      \brief We don't need this method for beam3contact elements!
    */
    void update_class_variables_step() override {};

    /*
    \brief Update nodal coordinates of both elements at the beginning of a new time step!
    */
    void update_ele_pos(Core::LinAlg::SerialDenseMatrix& newele1pos,
        Core::LinAlg::SerialDenseMatrix& newele2pos) override;

    /*
    \brief Update interpolated nodal tangents for tangent smoothing
    */
    void update_ele_smooth_tangents(
        std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions) override;

    /** \brief print information about this beam contact element pair to screen
     *
     */
    void print() const override;

    //@}
   private:
    //! @name member variables

    //! reference to problem discretization
    const Core::FE::Discretization& pdiscret_;

    //! reference to beam contact discretization
    const Core::FE::Discretization& cdiscret_;

    //! dof offset between pdiscret and cdiscret
    const std::map<int, int>& dofoffsetmap_;

    //! first element of contact pair
    Core::Elements::Element* element1_;

    //! second element of contact pair
    Core::Elements::Element* element2_;

    //! beam contact parameter list
    Teuchos::ParameterList& bcparams_;

    //! current node coordinates of the two elements
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPE> ele1pos_;
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPE> ele2pos_;

    //! neighbor elements of element 1
    std::shared_ptr<BeamInteraction::B3CNeighbor> neighbors1_;

    //! neighbor elements of element 2
    std::shared_ptr<BeamInteraction::B3CNeighbor> neighbors2_;

    //! averaged nodal tangents, necessary for smoothed tangent fields of C^0 Reissner beams
    Core::LinAlg::Matrix<3 * numnodes, 1> nodaltangentssmooth1_;
    Core::LinAlg::Matrix<3 * numnodes, 1> nodaltangentssmooth2_;

    //! current Newton iteration
    int iter_;

    //! current time step
    int numstep_;

    //! cross section radius of first beam
    const double r1_;

    //! cross section radius of second beam
    const double r2_;

    //! Maximal gap at which a contact can become active
    const double maxactivegap_;

    //! Maximal distance between a real segment on beam element 1 and its straight approximation
    double maxsegdist1_;

    //! Maximal distance between a real segment on beam element 2 and its straight approximation
    double maxsegdist2_;

    //! Number of segments on element1
    int numseg1_;

    //! Number of segments on element2
    int numseg2_;

    //! bound for search of large angle contact segment pairs
    double deltalargeangle_;

    //! bound for search of small angle contact segment pairs
    double deltasmallangle_;

    //! Indicates if the left / right node of the slave element 1 coincides with the endpoint of the
    //! physical beam (true) or not (false)
    std::pair<bool, bool> boundarynode1_;

    //! Indicates if the left / right node of the master element 2 coincides with the endpoint of
    //! the physical beam (true) or not (false)
    std::pair<bool, bool> boundarynode2_;

    //! Variables stored at the closest points of the large-angle-contact algorithm
    std::vector<std::shared_ptr<Beam3contactvariables<numnodes, numnodalvalues>>> cpvariables_;

    //! Variables stored at the Gauss points of the small-angle-contact algorithm
    std::vector<std::shared_ptr<Beam3contactvariables<numnodes, numnodalvalues>>> gpvariables_;

    //! Variables stored at the end points of the endpoint-contact algorithm
    std::vector<std::shared_ptr<Beam3contactvariables<numnodes, numnodalvalues>>> epvariables_;

    //@}

    //! @name Private evaluation methods

    /*!
    \brief Get active large angle pairs
    */
    void get_active_large_angle_pairs(std::vector<Core::LinAlg::Matrix<3, 1, double>>& endpoints1,
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& endpoints2,
        std::map<std::pair<int, int>, Core::LinAlg::Matrix<3, 1, double>>& closelargeanglesegments,
        const double pp);

    /*!
    \brief Evaluate active large angle pairs
    */
    void evaluate_active_large_angle_pairs(
        Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector<double>& fint);

    /*!
    \brief Get active small angle pairs
    */
    void get_active_small_angle_pairs(
        std::map<std::pair<int, int>, Core::LinAlg::Matrix<3, 1, double>>& closesmallanglesegments,
        std::pair<int, int>* iminmax = nullptr,
        std::pair<bool, bool>* leftrightsolutionwithinsegment = nullptr,
        std::pair<double, double>* eta1_leftrightboundary = nullptr);

    /*!
    \brief Evaluate active small angle pairs
    */
    void evaluate_active_small_angle_pairs(Core::LinAlg::SparseMatrix& stiffmatrix,
        Core::LinAlg::Vector<double>& fint, std::pair<int, int>* iminmax = nullptr,
        std::pair<bool, bool>* leftrightsolutionwithinsegment = nullptr,
        std::pair<double, double>* eta1_leftrightboundary = nullptr);


    /*!
    \brief Get active endpoint pairs
    */
    void get_active_end_point_pairs(
        std::vector<std::pair<int, int>>& closeendpointsegments, const double pp);

    /*!
    \brief Evaluate active endpoint pairs
    */
    void evaluate_active_end_point_pairs(
        Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector<double>& fint);

    /*!
    \brief Find segments close to each other
    */
    void get_close_segments(const std::vector<Core::LinAlg::Matrix<3, 1, double>>& endpoints1,
        const std::vector<Core::LinAlg::Matrix<3, 1, double>>& endpoints2,
        std::map<std::pair<int, int>, Core::LinAlg::Matrix<3, 1, double>>& closesmallanglesegments,
        std::map<std::pair<int, int>, Core::LinAlg::Matrix<3, 1, double>>& closelargeanglesegments,
        std::vector<std::pair<int, int>>& closeendpointsegments, double maxactivedist);

    /*!
    \brief Find contact point via closest point projection
    */
    bool closest_point_projection(double& eta_left1, double& eta_left2, double& l1, double& l2,
        Core::LinAlg::Matrix<3, 1, double>& segmentdata, std::pair<TYPE, TYPE>& solutionpoints,
        int segid1, int segid2);

    /*!
    \brief Find closest point eta2_master on a line for a given slave point eta1_slave
    */
    bool point_to_line_projection(double& eta1_slave, double& eta_left2, double& l2,
        double& eta2_master, double& gap, double& alpha, bool& pairactive, bool smallanglepair,
        bool invertpairs = false, bool orthogonalprojection = false);

    /*!
    \brief Determine minimal distance and contact angle for unconverged segment pair
    */
    void check_unconverged_segment_pair(double& eta_left1, double& eta_left2, double& l1,
        double& l2, double& eta1_min, double& eta2_min, double& g_min, double& alpha_g_min,
        bool& pointtolinesolfound);

    /*!
    \brief Subdivide elements into segments for CPP
    */
    double create_segments(Core::Elements::Element* ele,
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& endpoints_final, int& numsegment, int i);

    /*!
    \brief Get maximal gap at which a contact can become active
    */
    double get_max_active_dist();

    /*!
    \brief Check, if segments are fine enough
    */
    bool check_segment(Core::LinAlg::Matrix<3, 1, double>& r1,
        Core::LinAlg::Matrix<3, 1, double>& t1, Core::LinAlg::Matrix<3, 1, double>& r2,
        Core::LinAlg::Matrix<3, 1, double>& t2, Core::LinAlg::Matrix<3, 1, double>& rm,
        double& segdist);

    /*!
    \brief Calculate scalar contact force
    */
    void calc_penalty_law(Beam3contactvariables<numnodes, numnodalvalues>& variables);

    /*!
    \brief Calculate angle-dependent penalty scale factor for small-angle-contact
    */
    void calc_perp_penalty_scale_fac(Beam3contactvariables<numnodes, numnodalvalues>& cpvariables,
        Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi, Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const double shiftangle1, const double shiftangle2);

    /*!
    \brief Calculate angle-dependent penalty scale factor for large-angle-contact
    */
    void calc_par_penalty_scale_fac(Beam3contactvariables<numnodes, numnodalvalues>& gpvariables,
        Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi, Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const double shiftangle1, const double shiftangle2);

    /*!
     \brief Compute contact forces
     */
    void evaluate_fc_contact(Core::LinAlg::Vector<double>* fint,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1, const Core::LinAlg::Matrix<3, 1, TYPE>& r2,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
        Beam3contactvariables<numnodes, numnodalvalues>& variables, const double& intfac, bool cpp,
        bool gp, bool fixedendpointxi, bool fixedendpointeta,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPE>* fc1_FAD = nullptr,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPE>* fc2_FAD = nullptr);

    /*!
    \brief Evaluate contact stiffness
    */
    void evaluate_stiffc_contact(Core::LinAlg::SparseMatrix& stiffmatrix,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1, const Core::LinAlg::Matrix<3, 1, TYPE>& r2,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi,
        Beam3contactvariables<numnodes, numnodalvalues>& variables, const double& intfac, bool cpp,
        bool gp, bool fixedendpointxi, bool fixedendpointeta);

    /*!
    \brief FAD-based Evaluation of contact stiffness in case of ENDPOINTSEGMENTATION
    */
    void evaluate_stiffc_contact_int_seg(Core::LinAlg::SparseMatrix& stiffmatrix,
        const Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi_bound,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1, const Core::LinAlg::Matrix<3, 1, TYPE>& r2,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
        Beam3contactvariables<numnodes, numnodalvalues>& cpvariables, const double& intfac,
        const double& d_xi_ele_d_xi_bound, TYPE signed_jacobi_interval);

    /*!
    \brief Linearizations of contact point
    */
    void compute_lin_xi_and_lin_eta(
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi);

    /*!
    \brief Lin. of contact point coordinate eta with fixed xi
    */
    void compute_lin_eta_fix_xi(
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi);

    /*!
    \brief Lin. of contact point coordinate xi with fixed eta
    */
    void compute_lin_xi_fix_eta(
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi);

    /*!
    \brief Compute linearization of integration interval bounds (necessary in case of
    ENDPOINTSEGMENTATION)
    */
    void compute_lin_xi_bound(
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi_bound,
        TYPE& eta1_bound, TYPE eta2);

    /*!
    \brief Compute linearization of gap
    */
    void compute_lin_gap(
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_gap,
        const Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
        const Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r, const TYPE& norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2);

    /*!
    \brief Compute linearization of cosine of contact angle
    */
    void compute_lin_cos_contact_angle(
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_coscontactangle,
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
        Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi);

    /*!
    \brief Compute linearization of normal vector
    */
    void compute_lin_normal(
        Core::LinAlg::Matrix<3, 2 * 3 * numnodes * numnodalvalues, TYPE>& delta_normal,
        const Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_xi,
        const Core::LinAlg::Matrix<2 * 3 * numnodes * numnodalvalues, 1, TYPE>& delta_eta,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2);

    /*!
    \brief Calculate shape function values for given parameter values
    */
    void get_shape_functions(Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi, const TYPE& eta1,
        const TYPE& eta2);

    /*!
    \brief Calculate one specified shape function value / derivative for given parameter value and
    element
    */
    void get_shape_functions(Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N,
        const TYPE& eta, int deriv, Core::Elements::Element* ele);

    /*!
    \brief Assemble the shape functions into corresponding matrices
    */
    void assemble_shapefunctions(
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i,
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i_xi,
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i_xixi,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N_xi,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N_xixi);

    /*!
    \brief Assemble shape functions for one given matrix
    */
    void assemble_shapefunctions(
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N_i,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N);

    /*!
    \brief compute coordinate at given curve point
    */
    Core::LinAlg::Matrix<3, 1, TYPE> r(const TYPE& eta, Core::Elements::Element* ele);

    /*!
    \brief compute derivative at given curve point
    */
    Core::LinAlg::Matrix<3, 1, TYPE> r_xi(const TYPE& eta, Core::Elements::Element* ele);

    /*!
    \brief Compute coordinates and their derivatives from the discretization
    */
    void compute_coords_and_derivs(Core::LinAlg::Matrix<3, 1, TYPE>& r1,
        Core::LinAlg::Matrix<3, 1, TYPE>& r2, Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi, Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xixi);

    /*!
    \brief Compute coordinates of contact points of last time step from the discretization
    */
    void compute_old_coords_and_derivs(Core::LinAlg::Matrix<3, 1, TYPE>& r1_old,
        Core::LinAlg::Matrix<3, 1, TYPE>& r2_old, Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi_old,
        Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi_old,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi);

    /*!
    \brief Utility method for CPP (evaluate nonlinear function f)
    */
    void evaluate_orthogonality_condition(Core::LinAlg::Matrix<2, 1, TYPE>& f,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi, const Core::LinAlg::Matrix<3, 1, TYPE>& t1,
        const Core::LinAlg::Matrix<3, 1, TYPE>& t2);

    /*!
    \brief Utility method for CPP (evaluate Jacobian of nonlinear function f)
    */
    void evaluate_lin_orthogonality_condition(Core::LinAlg::Matrix<2, 2, TYPE>& df,
        Core::LinAlg::Matrix<2, 2, TYPE>& dfinv, const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const double norm_delta_r, const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi, const Core::LinAlg::Matrix<3, 1, TYPE>& t1,
        const Core::LinAlg::Matrix<3, 1, TYPE>& t2, const Core::LinAlg::Matrix<3, 1, TYPE>& t1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& t2_xi, bool& elementscolinear);

    /*!
    \brief Evaluate orthogonality cond. of point to line projection
    */
    void evaluate_ptl_orthogonality_condition(TYPE& f,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi, bool orthogonalprojection);

    /*!
    \brief Evaluate Jacobian df of PTLOrthogonalityCondition
    */
    bool evaluate_lin_ptl_orthogonality_condition(TYPE& df,
        const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi, bool orthogonalprojection);

    /*!
    \brief Compute normal vector and gap function at contact point
    */
    void compute_normal(Core::LinAlg::Matrix<3, 1, TYPE>& r1, Core::LinAlg::Matrix<3, 1, TYPE>& r2,
        Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi, Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        std::shared_ptr<Beam3contactvariables<numnodes, numnodalvalues>> variables,
        int contacttype);

    /*!
    \brief Check, if we have contact or not (e.g. gap < gmax [e.g. gmax=0]?)
    */
    bool check_contact_status(const double& gap);

    /*!
    \brief Check, if we have contact or not (e.g. gap < gdmax?)
    */
    bool check_damping_status(const double& gap);

    /*!
    \brief Get global dofs of a node

    Internally this method first extracts the dofs of the given node
    in the beam contact discretization (which has its own dofs) and
    then transfers these dofs to their actual GIDs in the underlying
    problem discretization by applying the pre-computed dofoffset_.
    */
    std::vector<int> get_global_dofs(const Core::Nodes::Node* node);

    /*!
      \brief Get jacobi factor of beam element
    */
    double get_jacobi(Core::Elements::Element* element1);

    /** \brief get Jacobi factor of beam element at xi \in [-1;1]
     *
     */
    inline double get_jacobi_at_xi(Core::Elements::Element* element1, const double& xi)
    {
      const Discret::Elements::Beam3Base* ele =
          dynamic_cast<const Discret::Elements::Beam3Base*>(element1);

      if (ele == nullptr) FOUR_C_THROW("Dynamic cast to Beam3Base failed");

      return ele->get_jacobi_fac_at_xi(xi);
    }

    /*!
      \brief Set class variables at the beginning of a Newton step
    */
    void set_class_variables(Teuchos::ParameterList& timeintparams);

    /*!
      \brief Linearization-check of coordinates xi and eta via FAD
    */
    void fad_check_lin_xi_and_lin_eta(const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xixi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N1_xi,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPE>& N2_xi);

    /*!
      \brief Linearization-check for local Newton in CPP via FAD
    */
    void fad_check_lin_orthogonality_condition(const Core::LinAlg::Matrix<3, 1, TYPE>& delta_r,
        const double& norm_delta_r, const Core::LinAlg::Matrix<3, 1, TYPE>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPE>& r2_xi, const Core::LinAlg::Matrix<3, 1, TYPE>& t1,
        const Core::LinAlg::Matrix<3, 1, TYPE>& t2);

    /*!
      \brief FD-Check of stiffness matrix
    */
    void fd_check(Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector<double>& fint,
        const double& pp,
        std::map<std::pair<int, int>, std::shared_ptr<Beam3contactinterface>>& contactpairmap,
        Teuchos::ParameterList& timeintparams, bool fdcheck);

    //@}

  };  // class Beam3contact
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
