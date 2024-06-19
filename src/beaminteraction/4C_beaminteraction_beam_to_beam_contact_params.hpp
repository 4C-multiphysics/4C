/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief data container holding all beam to beam contact
       input parameters

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/


#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_CONTACT_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_CONTACT_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_inpar_beamcontact.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BEAMINTERACTION
{
  class BeamToBeamContactParams
  {
   public:
    //! constructor
    BeamToBeamContactParams();

    //! destructor
    virtual ~BeamToBeamContactParams() = default;

    //! initialize with the stuff coming from input file
    void Init();

    //! setup member variables
    void setup();

    //! returns the isinit_ flag
    inline const bool& is_init() const { return isinit_; };

    //! returns the issetup_ flag
    inline const bool& is_setup() const { return issetup_; };

    //! Checks the init and setup status
    inline void check_init_setup() const
    {
      if (!is_init() or !is_setup()) FOUR_C_THROW("Call Init() and setup() first!");
    }

    //! Checks the init status
    inline void check_init() const
    {
      if (!is_init()) FOUR_C_THROW("Init() has not been called yet!");
    }

    inline enum Inpar::BEAMCONTACT::Strategy Strategy() const { return strategy_; }

    inline enum Inpar::BEAMCONTACT::PenaltyLaw PenaltyLaw() const { return penalty_law_; }

    inline double beam_to_beam_penalty_law_regularization_g0() const
    {
      return btb_penalty_law_regularization_g0_;
    }

    inline double beam_to_beam_penalty_law_regularization_f0() const
    {
      return btb_penalty_law_regularization_f0_;
    }

    inline double beam_to_beam_penalty_law_regularization_c0() const
    {
      return btb_penalty_law_regularization_c0_;
    }

    inline double GapShift() const { return gap_shift_; }

    inline double beam_to_beam_point_penalty_param() const { return btb_point_penalty_param_; }

    inline double beam_to_beam_line_penalty_param() const { return btb_line_penalty_param_; }

    inline double beam_to_beam_perp_shifting_angle1() const { return btb_perp_shifting_angle1_; }

    inline double beam_to_beam_perp_shifting_angle2() const { return btb_perp_shifting_angle2_; }

    inline double beam_to_beam_parallel_shifting_angle1() const
    {
      return btb_parallel_shifting_angle1_;
    }

    inline double beam_to_beam_parallel_shifting_angle2() const
    {
      return btb_parallel_shifting_angle2_;
    }

    inline double SegmentationAngle() const { return segangle_; }

    inline int num_integration_intervals() const { return num_integration_intervals_; }

    inline double BasicStiffGap() const { return btb_basicstiff_gap_; }

    inline bool EndPointPenalty() const { return btb_endpoint_penalty_; }

   private:
    bool isinit_;

    bool issetup_;

    //! strategy
    enum Inpar::BEAMCONTACT::Strategy strategy_;

    //! penalty law
    enum Inpar::BEAMCONTACT::PenaltyLaw penalty_law_;

    //! regularization parameters for penalty law
    double btb_penalty_law_regularization_g0_;
    double btb_penalty_law_regularization_f0_;
    double btb_penalty_law_regularization_c0_;

    //! Todo understand and check usage of this parameter
    double gap_shift_;

    //! beam-to-beam point penalty parameter
    double btb_point_penalty_param_;

    //! beam-to-beam line penalty parameter
    double btb_line_penalty_param_;

    //! shifting angles [radians] for point contact (near perpendicular configurations) fade
    double btb_perp_shifting_angle1_;
    double btb_perp_shifting_angle2_;

    //! shifting angles [radians] for line contact (near parallel configurations) fade
    double btb_parallel_shifting_angle1_;
    double btb_parallel_shifting_angle2_;

    //! maximum difference in tangent orientation between the endpoints of one created segment
    //  if the angle is larger => subdivide and create more segments
    double segangle_;

    //! number of integration intervals
    int num_integration_intervals_;

    //! gap value, from which on only the basic part of the stiffness contribution is applied
    //  this should accelerate convergence
    double btb_basicstiff_gap_;

    //! flag indicating if the integration should take special care of physical
    //  end points of beams in order to avoid strong discontinuities
    bool btb_endpoint_penalty_;
  };

}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
