/*----------------------------------------------------------------------*/
/*! \file
\brief
\level 2

*/


/*----------------------------------------------------------------------*
 | headers                                                  seitz 07/13 |
 *----------------------------------------------------------------------*/
#include "baci_so3_plast_ssn.hpp"

#include "baci_comm_utils_factory.hpp"
#include "baci_global_data.hpp"
#include "baci_inpar_tsi.hpp"
#include "baci_io_linedefinition.hpp"
#include "baci_linalg_serialdensevector.hpp"
#include "baci_mat_plasticelasthyper.hpp"
#include "baci_so3_line.hpp"
#include "baci_so3_surface.hpp"
#include "baci_thermo_ele_impl_utils.hpp"
#include "baci_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | ctor (public)                                            seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
DRT::ELEMENTS::So3Plast<distype>::So3Plast(int id, int owner)
    : SoBase(id, owner),
      fbar_(false),
      KbbInv_(std::vector<CORE::LINALG::SerialDenseMatrix>(0)),
      Kbd_(std::vector<CORE::LINALG::SerialDenseMatrix>(0)),
      fbeta_(std::vector<CORE::LINALG::SerialDenseVector>(0)),
      dDp_last_iter_(std::vector<CORE::LINALG::SerialDenseVector>(0)),
      dDp_inc_(std::vector<CORE::LINALG::SerialDenseVector>(0)),
      plspintype_(plspin),
      KaaInv_(Teuchos::null),
      Kad_(Teuchos::null),
      KaT_(Teuchos::null),
      KdT_eas_(Teuchos::null),
      feas_(Teuchos::null),
      Kba_(Teuchos::null),
      alpha_eas_(Teuchos::null),
      alpha_eas_last_timestep_(Teuchos::null),
      alpha_eas_delta_over_last_timestep_(Teuchos::null),
      alpha_eas_inc_(Teuchos::null),
      eastype_(soh8p_easnone),
      neas_(0),
      tsi_(false),
      is_nitsche_contact_(false)
{
  if (distype == CORE::FE::CellType::nurbs27)
    SetNurbsElement() = true;
  else
    SetNurbsElement() = false;
  return;
}


/*----------------------------------------------------------------------*
 | copy-ctor (public)                                       seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
DRT::ELEMENTS::So3Plast<distype>::So3Plast(const DRT::ELEMENTS::So3Plast<distype>& old)
    : SoBase(old)
{
  if (distype == CORE::FE::CellType::nurbs27)
    SetNurbsElement() = true;
  else
    SetNurbsElement() = false;
  return;
}


/*----------------------------------------------------------------------*
 | deep copy this instance of Solid3 and return pointer to  seitz 07/13 |
 | it (public)                                                          |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
DRT::Element* DRT::ELEMENTS::So3Plast<distype>::Clone() const
{
  auto* newelement = new DRT::ELEMENTS::So3Plast<distype>(*this);

  return newelement;
}


template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nen_, 1>>
    DRT::ELEMENTS::So3Plast<distype>::shapefunct_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nen_>>
    DRT::ELEMENTS::So3Plast<distype>::deriv_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::invJ_;
template <CORE::FE::CellType distype>
std::pair<bool, double> DRT::ELEMENTS::So3Plast<distype>::detJ_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nen_>>
    DRT::ELEMENTS::So3Plast<distype>::N_XYZ_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::defgrd_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::defgrd_mod_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::rcg_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::delta_Lp_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::numstr_,
                    DRT::ELEMENTS::So3Plast<distype>::numdofperelement_>>
    DRT::ELEMENTS::So3Plast<distype>::bop_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::numstr_, 1>>
    DRT::ELEMENTS::So3Plast<distype>::pk2_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::numstr_,
                    DRT::ELEMENTS::So3Plast<distype>::numstr_>>
    DRT::ELEMENTS::So3Plast<distype>::cmat_;

template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nen_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::xrefe_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nen_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::xcurr_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nen_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::xcurr_rate_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nen_, 1>>
    DRT::ELEMENTS::So3Plast<distype>::etemp_;

template <CORE::FE::CellType distype>
std::pair<bool, double> DRT::ELEMENTS::So3Plast<distype>::detF_;
template <CORE::FE::CellType distype>
std::pair<bool, double> DRT::ELEMENTS::So3Plast<distype>::detF_0_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::inv_defgrd_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::inv_defgrd_0_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nen_>>
    DRT::ELEMENTS::So3Plast<distype>::N_XYZ_0_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::numstr_, 1>>
    DRT::ELEMENTS::So3Plast<distype>::rcg_vec_;
template <CORE::FE::CellType distype>
std::pair<bool, double> DRT::ELEMENTS::So3Plast<distype>::f_bar_fac_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::numdofperelement_, 1>>
    DRT::ELEMENTS::So3Plast<distype>::htensor_;

template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::numstr_,
                    DRT::ELEMENTS::So3Plast<distype>::numstr_>>
    DRT::ELEMENTS::So3Plast<distype>::T0invT_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nsd_,
                    DRT::ELEMENTS::So3Plast<distype>::nsd_>>
    DRT::ELEMENTS::So3Plast<distype>::jac_0_;
template <CORE::FE::CellType distype>
std::pair<bool, double> DRT::ELEMENTS::So3Plast<distype>::det_jac_0_;
template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::SerialDenseMatrix> DRT::ELEMENTS::So3Plast<distype>::M_eas_;

template <CORE::FE::CellType distype>
std::pair<bool, CORE::LINALG::Matrix<DRT::ELEMENTS::So3Plast<distype>::nen_, 1>>
    DRT::ELEMENTS::So3Plast<distype>::weights_;
template <CORE::FE::CellType distype>
std::pair<bool, std::vector<CORE::LINALG::SerialDenseVector>>
    DRT::ELEMENTS::So3Plast<distype>::knots_;

/*----------------------------------------------------------------------*
 |                                                          seitz 05/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
int DRT::ELEMENTS::So3Plast<distype>::NumVolume() const
{
  switch (distype)
  {
    case CORE::FE::CellType::tet4:
    case CORE::FE::CellType::hex8:
    case CORE::FE::CellType::hex18:
    case CORE::FE::CellType::hex27:
    case CORE::FE::CellType::nurbs27:
      return 0;
      break;
    default:
      FOUR_C_THROW("unknown distpye for So3Plast");
      break;
      return 0;
  }
}

/*----------------------------------------------------------------------*
 |                                                          seitz 05/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
int DRT::ELEMENTS::So3Plast<distype>::NumSurface() const
{
  switch (distype)
  {
    case CORE::FE::CellType::hex8:
    case CORE::FE::CellType::hex18:
    case CORE::FE::CellType::hex27:
    case CORE::FE::CellType::nurbs27:
      return 6;
      break;
    case CORE::FE::CellType::tet4:
      return 4;
      break;
    default:
      FOUR_C_THROW("unknown distpye for So3Plast");
      break;
      return 0;
  }
}

/*----------------------------------------------------------------------*
 |                                                          seitz 05/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
int DRT::ELEMENTS::So3Plast<distype>::NumLine() const
{
  switch (distype)
  {
    case CORE::FE::CellType::hex8:
    case CORE::FE::CellType::hex18:
    case CORE::FE::CellType::hex27:
    case CORE::FE::CellType::nurbs27:
      return 12;
      break;
    case CORE::FE::CellType::tet4:
      return 6;
      break;
    default:
      FOUR_C_THROW("unknown distpye for So3Plast");
      break;
      return 0;
  }
}

/*----------------------------------------------------------------------*
 |                                                          seitz 05/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::So3Plast<distype>::Lines()
{
  return CORE::COMM::ElementBoundaryFactory<StructuralLine, DRT::Element>(
      CORE::COMM::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |                                                          seitz 05/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::So3Plast<distype>::Surfaces()
{
  return CORE::COMM::ElementBoundaryFactory<StructuralSurface, DRT::Element>(
      CORE::COMM::buildSurfaces, *this);
}

/*----------------------------------------------------------------------*
 | pack data (public)                                       seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::So3Plast<distype>::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // add base class Element
  SoBase::Pack(data);

  // Gauss points and weights
  const auto size2 = (int)xsi_.size();
  AddtoPack(data, size2);
  for (int i = 0; i < size2; ++i) AddtoPack(data, xsi_[i]);
  AddtoPack(data, wgt_);

  // parameters
  AddtoPack(data, (int)fbar_);

  // plastic spin type
  AddtoPack(data, (int)plspintype_);

  // tsi
  AddtoPack(data, (int)tsi_);
  if (tsi_)
  {
    AddtoPack(data, (int)KbT_->size());
    for (unsigned i = 0; i < KbT_->size(); i++)
    {
      AddtoPack(data, (*dFintdT_)[i]);
      AddtoPack(data, (*KbT_)[i]);
      AddtoPack(data, (*temp_last_)[i]);
    }
  }

  // EAS element technology
  AddtoPack(data, (int)eastype_);
  AddtoPack(data, neas_);
  if (eastype_ != soh8p_easnone)
  {
    AddtoPack(data, (*alpha_eas_));
    AddtoPack(data, (*alpha_eas_last_timestep_));
    AddtoPack(data, (*alpha_eas_delta_over_last_timestep_));
  }

  // history at each Gauss point
  int histsize = dDp_last_iter_.size();
  AddtoPack(data, histsize);
  if (histsize != 0)
    for (int i = 0; i < histsize; i++) AddtoPack(data, dDp_last_iter_[i]);

  // nitsche contact
  AddtoPack(data, (int)is_nitsche_contact_);
  if (is_nitsche_contact_)
  {
    AddtoPack(data, cauchy_);
    AddtoPack(data, cauchy_deriv_);
    if (tsi_) AddtoPack(data, cauchy_deriv_T_);
  }

  return;
}  // Pack()


/*----------------------------------------------------------------------*
 | unpack data (public)                                     seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::So3Plast<distype>::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  SoBase::Unpack(basedata);

  // Gauss points and weights
  int size2 = ExtractInt(position, data);
  xsi_.resize(size2, CORE::LINALG::Matrix<nsd_, 1>(true));
  for (int i = 0; i < size2; ++i) ExtractfromPack(position, data, xsi_[i]);
  ExtractfromPack(position, data, wgt_);
  numgpt_ = wgt_.size();

  // paramters
  fbar_ = (bool)ExtractInt(position, data);

  // plastic spin type
  plspintype_ = static_cast<PlSpinType>(ExtractInt(position, data));

  // tsi
  tsi_ = (bool)ExtractInt(position, data);
  if (tsi_)
  {
    dFintdT_ = Teuchos::rcp(new std::vector<CORE::LINALG::Matrix<numdofperelement_, 1>>(numgpt_));
    KbT_ = Teuchos::rcp(new std::vector<CORE::LINALG::SerialDenseVector>(
        numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true)));
    temp_last_ = Teuchos::rcp(new std::vector<double>(numgpt_));
    int size = ExtractInt(position, data);
    for (int i = 0; i < size; i++)
    {
      ExtractfromPack(position, data, (*dFintdT_)[i]);
      ExtractfromPack(position, data, (*KbT_)[i]);
      ExtractfromPack(position, data, (*temp_last_)[i]);
    }
  }

  // EAS element technology
  eastype_ = static_cast<DRT::ELEMENTS::So3PlastEasType>(ExtractInt(position, data));
  ExtractfromPack(position, data, neas_);

  // no EAS
  if (eastype_ == soh8p_easnone)
  {
    KaaInv_ = Teuchos::null;
    Kad_ = Teuchos::null;
    KaT_ = Teuchos::null;
    KdT_eas_ = Teuchos::null;
    feas_ = Teuchos::null;
    Kba_ = Teuchos::null;
    alpha_eas_ = Teuchos::null;
    alpha_eas_last_timestep_ = Teuchos::null;
    alpha_eas_delta_over_last_timestep_ = Teuchos::null;
    alpha_eas_inc_ = Teuchos::null;
  }
  else
  {
    KaaInv_ = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(neas_, neas_, true));
    Kad_ = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(neas_, numdofperelement_, true));
    if (tsi_)
    {
      KaT_ = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(neas_, nen_, true));
      KdT_eas_ = Teuchos::rcp(new CORE::LINALG::Matrix<numdofperelement_, nen_>);
    }
    feas_ = Teuchos::rcp(new CORE::LINALG::SerialDenseVector(neas_, true));
    Kba_ = Teuchos::rcp(new std::vector<CORE::LINALG::SerialDenseMatrix>(
        numgpt_, CORE::LINALG::SerialDenseMatrix(plspintype_, neas_, true)));
    alpha_eas_ = Teuchos::rcp(new CORE::LINALG::SerialDenseVector(neas_, true));
    alpha_eas_last_timestep_ = Teuchos::rcp(new CORE::LINALG::SerialDenseVector(neas_, true));
    alpha_eas_delta_over_last_timestep_ =
        Teuchos::rcp(new CORE::LINALG::SerialDenseVector(neas_, true));
    alpha_eas_inc_ = Teuchos::rcp(new CORE::LINALG::SerialDenseVector(neas_, true));
  }

  KbbInv_.resize(numgpt_, CORE::LINALG::SerialDenseMatrix(plspintype_, plspintype_, true));
  Kbd_.resize(numgpt_, CORE::LINALG::SerialDenseMatrix(plspintype_, numdofperelement_, true));
  fbeta_.resize(numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true));
  dDp_last_iter_.resize(numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true));
  dDp_inc_.resize(numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true));

  if (eastype_ != soh8p_easnone)
  {
    ExtractfromPack(position, data, (*alpha_eas_));
    ExtractfromPack(position, data, (*alpha_eas_last_timestep_));
    ExtractfromPack(position, data, (*alpha_eas_delta_over_last_timestep_));
  }

  int size = ExtractInt(position, data);
  for (int i = 0; i < size; i++) ExtractfromPack(position, data, dDp_last_iter_[i]);

  // Nitsche contact stuff
  is_nitsche_contact_ = (bool)ExtractInt(position, data);
  if (is_nitsche_contact_)
  {
    ExtractfromPack(position, data, cauchy_);
    ExtractfromPack(position, data, cauchy_deriv_);
    if (tsi_)
      ExtractfromPack(position, data, cauchy_deriv_T_);
    else
      cauchy_deriv_T_.resize(0);
  }
  else
  {
    cauchy_.resize(0);
    cauchy_deriv_.resize(0);
    cauchy_deriv_T_.resize(0);
  }

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;

}  // Unpack()


/*----------------------------------------------------------------------*
 | print this element (public)                              seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::So3Plast<distype>::Print(std::ostream& os) const
{
  os << "So3Plast ";
  return;
}


/*----------------------------------------------------------------------*
 | read this element, get the material (public)             seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
bool DRT::ELEMENTS::So3Plast<distype>::ReadElement(
    const std::string& eletype, const std::string& eledistype, INPUT::LineDefinition* linedef)
{
  std::string buffer;
  linedef->ExtractString("KINEM", buffer);

  // geometrically linear
  if (buffer == "linear")
  {
    FOUR_C_THROW("no linear kinematics");
  }
  // geometrically non-linear with Total Lagrangean approach
  else if (buffer == "nonlinear")
  {
    kintype_ = INPAR::STR::KinemType::nonlinearTotLag;
    // everything ok
  }
  else
    FOUR_C_THROW("Reading of SO3_PLAST element failed! KINEM unknown");

  // fbar
  if (linedef->HaveNamed("FBAR"))
  {
    std::string fb;
    linedef->ExtractString("FBAR", fb);
    if (fb == "yes")
      fbar_ = true;
    else if (fb == "no")
      fbar_ = false;
    else
      FOUR_C_THROW("unknown fbar option (valid: yes/no)");
  }

  // quadrature
  if (linedef->HaveNamed("NUMGP"))
  {
    if (distype != CORE::FE::CellType::hex8)
      FOUR_C_THROW("You may only choose the Gauss point number for SOLIDH8PLAST");
    if (GLOBAL::Problem::Instance()->GetProblemType() == GLOBAL::ProblemType::tsi)
      FOUR_C_THROW("You may not choose the Gauss point number in TSI problems");

    int ngp = 0;
    linedef->ExtractInt("NUMGP", ngp);

    switch (ngp)
    {
      case 8:
      {
        CORE::FE::IntPointsAndWeights<nsd_> intpoints(CORE::FE::GaussRule3D::hex_8point);
        numgpt_ = intpoints.IP().nquad;
        xsi_.resize(numgpt_);
        wgt_.resize(numgpt_);
        for (int gp = 0; gp < numgpt_; ++gp)
        {
          wgt_[gp] = (intpoints.IP().qwgt)[gp];
          const double* gpcoord = (intpoints.IP().qxg)[gp];
          for (int idim = 0; idim < nsd_; idim++) xsi_[gp](idim) = gpcoord[idim];
        }
        break;
      }
      case 9:
      {
        CORE::FE::GaussIntegration ip(distype, 3);
        numgpt_ = ip.NumPoints() + 1;
        xsi_.resize(numgpt_);
        wgt_.resize(numgpt_);
        for (int gp = 0; gp < numgpt_ - 1; ++gp)
        {
          wgt_[gp] = 5. / 9.;
          const double* gpcoord = ip.Point(gp);
          for (int idim = 0; idim < nsd_; idim++) xsi_[gp](idim) = gpcoord[idim];
        }
        // 9th quadrature point at element center
        xsi_[numgpt_ - 1](0) = 0.;
        xsi_[numgpt_ - 1](1) = 0.;
        xsi_[numgpt_ - 1](2) = 0.;
        wgt_[numgpt_ - 1] = 32. / 9.;
        break;
      }
      case 27:
      {
        CORE::FE::IntPointsAndWeights<nsd_> intpoints(CORE::FE::GaussRule3D::hex_27point);
        numgpt_ = intpoints.IP().nquad;
        xsi_.resize(numgpt_);
        wgt_.resize(numgpt_);
        for (int gp = 0; gp < numgpt_; ++gp)
        {
          wgt_[gp] = (intpoints.IP().qwgt)[gp];
          const double* gpcoord = (intpoints.IP().qxg)[gp];
          for (int idim = 0; idim < nsd_; idim++) xsi_[gp](idim) = gpcoord[idim];
        }
        break;
      }
      default:
        FOUR_C_THROW("so3_plast doesn't know what to do with %i Gauss points", ngp);
        break;
    }
  }
  else  // default integration
  {
    CORE::FE::IntPointsAndWeights<nsd_> intpoints(THR::DisTypeToOptGaussRule<distype>::rule);
    numgpt_ = intpoints.IP().nquad;
    xsi_.resize(numgpt_);
    wgt_.resize(numgpt_);
    for (int gp = 0; gp < numgpt_; ++gp)
    {
      wgt_[gp] = (intpoints.IP().qwgt)[gp];
      const double* gpcoord = (intpoints.IP().qxg)[gp];
      for (int idim = 0; idim < nsd_; idim++) xsi_[gp](idim) = gpcoord[idim];
    }
  }

  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);

  SetMaterial(material);

  Teuchos::RCP<MAT::So3Material> so3mat = SolidMaterial();
  so3mat->Setup(numgpt_, linedef);
  so3mat->ValidKinematics(INPAR::STR::KinemType::nonlinearTotLag);


  // Validate that materials doesn't use extended update call.
  if (SolidMaterial()->UsesExtendedUpdate())
    FOUR_C_THROW("This element currently does not support the extended update call.");

  if (so3mat->MaterialType() != INPAR::MAT::m_plelasthyper)
    std::cout << "*** warning *** so3plast used w/o PlasticElastHyper material. Better use "
                 "standard solid element!\n";
  if (HavePlasticSpin())
    plspintype_ = plspin;
  else
    plspintype_ = zerospin;

  // EAS
  if (linedef->HaveNamed("EAS"))
  {
    if (distype != CORE::FE::CellType::hex8)
      FOUR_C_THROW("EAS in so3 plast currently only for HEX8 elements");

    linedef->ExtractString("EAS", buffer);

    if (buffer == "none")
      eastype_ = soh8p_easnone;
    else if (buffer == "mild")
      eastype_ = soh8p_easmild;
    else if (buffer == "full")
      eastype_ = soh8p_easfull;
    else
      FOUR_C_THROW("unknown EAS type for so3_plast");

    if (fbar_ && eastype_ != soh8p_easnone) FOUR_C_THROW("no combination of Fbar and EAS");
  }
  else
    eastype_ = soh8p_easnone;

  // initialize EAS data
  EasInit();

  // plasticity related stuff
  KbbInv_.resize(numgpt_, CORE::LINALG::SerialDenseMatrix(plspintype_, plspintype_, true));
  Kbd_.resize(numgpt_, CORE::LINALG::SerialDenseMatrix(plspintype_, numdofperelement_, true));
  fbeta_.resize(numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true));
  dDp_last_iter_.resize(numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true));
  dDp_inc_.resize(numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true));

  Teuchos::ParameterList plparams = GLOBAL::Problem::Instance()->SemiSmoothPlastParams();
  CORE::UTILS::AddEnumClassToParameterList(
      "GLOBAL::ProblemType", GLOBAL::Problem::Instance()->GetProblemType(), plparams);
  ReadParameterList(Teuchos::rcpFromRef<Teuchos::ParameterList>(plparams));


  return true;

}  // ReadElement()

/*----------------------------------------------------------------------*
 | get the nodes from so3 (public)                          seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
int DRT::ELEMENTS::So3Plast<distype>::UniqueParObjectId() const
{
  switch (distype)
  {
    case CORE::FE::CellType::hex8:
    {
      return SoHex8PlastType::Instance().UniqueParObjectId();
      break;
    }  // hex8
    case CORE::FE::CellType::hex27:
      return SoHex27PlastType::Instance().UniqueParObjectId();
      break;
    case CORE::FE::CellType::tet4:
      return SoTet4PlastType::Instance().UniqueParObjectId();
      break;
    case CORE::FE::CellType::nurbs27:
      return SoNurbs27PlastType::Instance().UniqueParObjectId();
      break;
    default:
      FOUR_C_THROW("unknown element type!");
      break;
  }
  // Intel compiler needs a return
  return -1;

}  // UniqueParObjectId()


/*----------------------------------------------------------------------*
 | get the nodes from so3 (public)                          seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
DRT::ElementType& DRT::ELEMENTS::So3Plast<distype>::ElementType() const
{
  switch (distype)
  {
    case CORE::FE::CellType::hex8:
    {
      return SoHex8PlastType::Instance();
      break;
    }
    case CORE::FE::CellType::hex27:
      return SoHex27PlastType::Instance();
      break;
    case CORE::FE::CellType::tet4:
      return SoTet4PlastType::Instance();
      break;
    case CORE::FE::CellType::nurbs27:
      return SoNurbs27PlastType::Instance();
      break;
    default:
      FOUR_C_THROW("unknown element type!");
      break;
  }
  // Intel compiler needs a return
  return SoHex8PlastType::Instance();

};  // ElementType()


/*----------------------------------------------------------------------*
 | return names of visualization data (public)              seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::So3Plast<distype>::VisNames(std::map<std::string, int>& names)
{
  DRT::Element::VisNames(names);
  SolidMaterial()->VisNames(names);

  return;
}  // VisNames()

/*----------------------------------------------------------------------*
 | return visualization data (public)                       seitz 07/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
bool DRT::ELEMENTS::So3Plast<distype>::VisData(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (DRT::Element::VisData(name, data)) return true;

  return SolidMaterial()->VisData(name, data, numgpt_, Id());

}  // VisData()

/*----------------------------------------------------------------------*
 | read relevant parameters from paramter list              seitz 01/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::So3Plast<distype>::ReadParameterList(
    Teuchos::RCP<Teuchos::ParameterList> plparams)
{
  double cpl = plparams->get<double>("SEMI_SMOOTH_CPL");
  double s = plparams->get<double>("STABILIZATION_S");
  if (Material()->MaterialType() == INPAR::MAT::m_plelasthyper)
    static_cast<MAT::PlasticElastHyper*>(Material().get())->GetParams(s, cpl);

  GLOBAL::ProblemType probtype =
      Teuchos::getIntegralValue<GLOBAL::ProblemType>(*plparams, "GLOBAL::ProblemType");
  if (probtype == GLOBAL::ProblemType::tsi)
    tsi_ = true;
  else
    tsi_ = false;
  if (tsi_)
  {
    // get plastic hyperelastic material
    MAT::PlasticElastHyper* plmat = nullptr;
    if (Material()->MaterialType() == INPAR::MAT::m_plelasthyper)
      plmat = static_cast<MAT::PlasticElastHyper*>(Material().get());
    else
      FOUR_C_THROW("so3_ssn_plast elements only with PlasticElastHyper material");

    // get dissipation mode
    auto mode =
        CORE::UTILS::IntegralValue<INPAR::TSI::DissipationMode>(*plparams, "DISSIPATION_MODE");

    // prepare material for tsi
    plmat->SetupTSI(numgpt_, numdofperelement_, (eastype_ != soh8p_easnone), mode);

    // setup element data
    dFintdT_ = Teuchos::rcp(new std::vector<CORE::LINALG::Matrix<numdofperelement_, 1>>(numgpt_));
    temp_last_ = Teuchos::rcp(new std::vector<double>(numgpt_, plmat->InitTemp()));
    KbT_ = Teuchos::rcp(new std::vector<CORE::LINALG::SerialDenseVector>(
        numgpt_, CORE::LINALG::SerialDenseVector(plspintype_, true)));

    if (eastype_ != soh8p_easnone)
    {
      KaT_ = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(neas_, nen_, true));
      KdT_eas_ = Teuchos::rcp(new CORE::LINALG::Matrix<numdofperelement_, nen_>);
    }
    else
    {
      KaT_ = Teuchos::null;
      KdT_eas_ = Teuchos::null;
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
template <unsigned int num_cols>
void DRT::ELEMENTS::So3Plast<distype>::soh8_expol(
    CORE::LINALG::Matrix<numgpt_post, num_cols>& data, Epetra_MultiVector& expolData)
{
  if (distype != CORE::FE::CellType::hex8) FOUR_C_THROW("soh8_expol called from non-hex8 element");

  // static variables, that are the same for every element
  static CORE::LINALG::Matrix<nen_, numgpt_post> expolOperator;
  static bool isfilled;

  if (isfilled == false)
  {
    double sq3 = sqrt(3.0);

    expolOperator(0, 0) = 1.25 + 0.75 * sq3;
    expolOperator(0, 1) = -0.25 - 0.25 * sq3;
    expolOperator(0, 2) = -0.25 + 0.25 * sq3;
    expolOperator(0, 3) = -0.25 - 0.25 * sq3;
    expolOperator(0, 4) = -0.25 - 0.25 * sq3;
    expolOperator(0, 5) = -0.25 + 0.25 * sq3;
    expolOperator(0, 6) = 1.25 - 0.75 * sq3;
    expolOperator(0, 7) = -0.25 + 0.25 * sq3;
    expolOperator(1, 1) = 1.25 + 0.75 * sq3;
    expolOperator(1, 2) = -0.25 - 0.25 * sq3;
    expolOperator(1, 3) = -0.25 + 0.25 * sq3;
    expolOperator(1, 4) = -0.25 + 0.25 * sq3;
    expolOperator(1, 5) = -0.25 - 0.25 * sq3;
    expolOperator(1, 6) = -0.25 + 0.25 * sq3;
    expolOperator(1, 7) = 1.25 - 0.75 * sq3;
    expolOperator(2, 2) = 1.25 + 0.75 * sq3;
    expolOperator(2, 3) = -0.25 - 0.25 * sq3;
    expolOperator(2, 4) = 1.25 - 0.75 * sq3;
    expolOperator(2, 5) = -0.25 + 0.25 * sq3;
    expolOperator(2, 6) = -0.25 - 0.25 * sq3;
    expolOperator(2, 7) = -0.25 + 0.25 * sq3;
    expolOperator(3, 3) = 1.25 + 0.75 * sq3;
    expolOperator(3, 4) = -0.25 + 0.25 * sq3;
    expolOperator(3, 5) = 1.25 - 0.75 * sq3;
    expolOperator(3, 6) = -0.25 + 0.25 * sq3;
    expolOperator(3, 7) = -0.25 - 0.25 * sq3;
    expolOperator(4, 4) = 1.25 + 0.75 * sq3;
    expolOperator(4, 5) = -0.25 - 0.25 * sq3;
    expolOperator(4, 6) = -0.25 + 0.25 * sq3;
    expolOperator(4, 7) = -0.25 - 0.25 * sq3;
    expolOperator(5, 5) = 1.25 + 0.75 * sq3;
    expolOperator(5, 6) = -0.25 - 0.25 * sq3;
    expolOperator(5, 7) = -0.25 + 0.25 * sq3;
    expolOperator(6, 6) = 1.25 + 0.75 * sq3;
    expolOperator(6, 7) = -0.25 - 0.25 * sq3;
    expolOperator(7, 7) = 1.25 + 0.75 * sq3;

    for (int i = 0; i < NUMNOD_SOH8; ++i)
    {
      for (int j = 0; j < i; ++j)
      {
        expolOperator(i, j) = expolOperator(j, i);
      }
    }

    isfilled = true;
  }

  CORE::LINALG::Matrix<nen_, num_cols> nodalData;
  nodalData.Multiply(expolOperator, data);

  // "assembly" of extrapolated nodal data
  for (int i = 0; i < nen_; ++i)
  {
    const int lid = expolData.Map().LID(NodeIds()[i]);
    if (lid >= 0)  // rownode
    {
      const double invmyadjele = 1.0 / Nodes()[i]->NumElement();
      for (unsigned int j = 0; j < num_cols; ++j)
        (*(expolData(j)))[lid] += nodalData(i, j) * invmyadjele;
    }
  }
  return;
}

template void DRT::ELEMENTS::So3Plast<CORE::FE::CellType::hex8>::soh8_expol(
    CORE::LINALG::Matrix<numgpt_post, 1>&, Epetra_MultiVector&);
template void DRT::ELEMENTS::So3Plast<CORE::FE::CellType::hex8>::soh8_expol(
    CORE::LINALG::Matrix<numgpt_post, numstr_>&, Epetra_MultiVector&);
template void DRT::ELEMENTS::So3Plast<CORE::FE::CellType::hex8>::soh8_expol(
    CORE::LINALG::Matrix<numgpt_post, 9>&, Epetra_MultiVector&);

/*----------------------------------------------------------------------*
 | Have plastic spin                                        seitz 05/14 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
bool DRT::ELEMENTS::So3Plast<distype>::HavePlasticSpin()
{
  // get plastic hyperelastic material
  MAT::PlasticElastHyper* plmat = nullptr;
  if (Material()->MaterialType() == INPAR::MAT::m_plelasthyper)
    plmat = static_cast<MAT::PlasticElastHyper*>(Material().get());

  if (plmat != nullptr) return plmat->HavePlasticSpin();

  return false;
}

int DRT::ELEMENTS::PlastEasTypeToNumEasV(DRT::ELEMENTS::So3PlastEasType et)
{
  switch (et)
  {
    case soh8p_easnone:
      return PlastEasTypeToNumEas<soh8p_easnone>::neas;
      break;
    case soh8p_easmild:
      return PlastEasTypeToNumEas<soh8p_easmild>::neas;
      break;
    case soh8p_easfull:
      return PlastEasTypeToNumEas<soh8p_easfull>::neas;
      break;
    case soh8p_eassosh8:
      return PlastEasTypeToNumEas<soh8p_eassosh8>::neas;
      break;
    case soh18p_eassosh18:
      return PlastEasTypeToNumEas<soh18p_eassosh18>::neas;
      break;
    default:
      FOUR_C_THROW("EAS type not implemented");
  }
  return -1;
}

FOUR_C_NAMESPACE_CLOSE

#include "baci_so3_ssn_plast_fwd.hpp"
