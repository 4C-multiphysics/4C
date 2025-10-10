// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_constraintmixture.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"  // for debug plotting with gmsh
#include "4C_fem_general_utils_integration.hpp"         // for debug plotting with gmsh
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"  // for pstime
#include "4C_io_control.hpp"       // for debug plotting with gmsh
#include "4C_io_gmsh.hpp"          // for debug plotting with gmsh
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_constraintmixture_history.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
Mat::PAR::ConstraintMixture::ConstraintMixture(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      density_(matdata.parameters.get<double>("DENS")),
      mue_(matdata.parameters.get<double>("MUE")),
      nue_(matdata.parameters.get<double>("NUE")),
      phielastin_(matdata.parameters.get<double>("PHIE")),
      prestretchelastin_(matdata.parameters.get<double>("PREELA")),
      k1_(matdata.parameters.get<double>("K1")),
      k2_(matdata.parameters.get<double>("K2")),
      numhom_(matdata.parameters.get<int>("NUMHOM")),
      prestretchcollagen_((matdata.parameters.get<std::vector<double>>("PRECOLL"))),
      damagestretch_(matdata.parameters.get<double>("DAMAGE")),
      k1muscle_(matdata.parameters.get<double>("K1M")),
      k2muscle_(matdata.parameters.get<double>("K2M")),
      phimuscle_(matdata.parameters.get<double>("PHIM")),
      prestretchmuscle_(matdata.parameters.get<double>("PREMUS")),
      Smax_(matdata.parameters.get<double>("SMAX")),
      kappa_(matdata.parameters.get<double>("KAPPA")),
      lifetime_(matdata.parameters.get<double>("LIFETIME")),
      homstress_((matdata.parameters.get<std::vector<double>>("HOMSTR"))),
      sheargrowthfactor_(matdata.parameters.get<double>("SHEARGROWTHFAC")),
      homradius_(matdata.parameters.get<double>("HOMRAD")),
      starttime_(matdata.parameters.get<double>("STARTTIME")),
      integration_(matdata.parameters.get<std::string>("INTEGRATION")),
      abstol_(matdata.parameters.get<double>("TOL")),
      growthforce_(matdata.parameters.get<std::string>("GROWTHFORCE")),
      elastindegrad_(matdata.parameters.get<std::string>("ELASTINDEGRAD")),
      massprodfunc_(matdata.parameters.get<std::string>("MASSPROD")),
      initstretch_(matdata.parameters.get<std::string>("INITSTRETCH")),
      timecurve_(matdata.parameters.get<int>("CURVE")),
      degoption_((matdata.parameters.get<std::string>("DEGOPTION"))),
      maxmassprodfac_(matdata.parameters.get<double>("MAXMASSPRODFAC")),
      storehistory_(matdata.parameters.get<bool>("STOREHISTORY")),
      degtol_(1.0e-6)
{
  Core::LinAlg::Map dummy_map(
      1, 1, 0, Global::Problem::instance()->get_communicators().local_comm());
  for (int i = first; i <= last; i++)
  {
    matparams_.push_back(std::make_shared<Core::LinAlg::Vector<double>>(dummy_map, true));
  }
  matparams_.at(growthfactor)->put_scalar(matdata.parameters.get<double>("GROWTHFAC"));
  matparams_.at(elastin_survival)->put_scalar(matdata.parameters.get<double>("ELASTINFAC"));
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::ConstraintMixture::create_material()
{
  return std::make_shared<Mat::ConstraintMixture>(this);
}

Mat::ConstraintMixtureType Mat::ConstraintMixtureType::instance_;

Core::Communication::ParObject* Mat::ConstraintMixtureType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ConstraintMixture* comix = new Mat::ConstraintMixture();
  comix->unpack(buffer);
  return comix;
}

/*----------------------------------------------------------------------*
 |  Constructor                                   (public)         12/10|
 *----------------------------------------------------------------------*/
Mat::ConstraintMixture::ConstraintMixture() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 |  Copy-Constructor                             (public)          12/10|
 *----------------------------------------------------------------------*/
Mat::ConstraintMixture::ConstraintMixture(Mat::PAR::ConstraintMixture* params) : params_(params) {}

/*----------------------------------------------------------------------*
 |  Pack                                          (public)         12/10|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  int numgp;
  if (!isinit_)
  {
    numgp = 0;  // not initialized -> nothing to pack
  }
  else
  {
    numgp = a1_->size();  // size is number of gausspoints
  }
  add_to_pack(data, numgp);
  // Pack internal variables
  for (int gp = 0; gp < numgp; gp++)
  {
    add_to_pack(data, a1_->at(gp));
    add_to_pack(data, a2_->at(gp));
    add_to_pack(data, a3_->at(gp));
    add_to_pack(data, a4_->at(gp));
    add_to_pack(data, vismassstress_->at(gp));
    add_to_pack(data, refmassdens_->at(gp));
    add_to_pack(data, visrefmassdens_->at(gp));
    add_to_pack(data, localprestretch_->at(gp));
    add_to_pack(data, localhomstress_->at(gp));
  }
  if (numgp > 0)
  {
    add_to_pack(data, massprodbasal_);
    add_to_pack(data, homradius_);

    int delsize = deletemass_->size();
    add_to_pack(data, delsize);
    for (int iddel = 0; iddel < delsize; iddel++) add_to_pack(data, deletemass_->at(iddel));

    // Pack history
    int minindex = minindex_;
    add_to_pack(data, minindex);
    int sizehistory = history_->size();
    add_to_pack(data, sizehistory);
    for (int idpast = 0; idpast < sizehistory; idpast++) history_->at(idpast).pack(data);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack                                        (public)         12/10|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::unpack(Core::Communication::UnpackBuffer& buffer)
{
  isinit_ = true;


  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::ConstraintMixture*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  int numgp;
  extract_from_pack(buffer, numgp);
  if (numgp == 0)
  {  // no history data to unpack
    isinit_ = false;

    return;
  }

  // unpack fiber internal variables
  a1_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  a2_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  a3_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  a4_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  vismassstress_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  refmassdens_ = std::make_shared<std::vector<double>>(numgp);
  visrefmassdens_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  localprestretch_ = std::make_shared<std::vector<Core::LinAlg::Matrix<4, 1>>>(numgp);
  localhomstress_ = std::make_shared<std::vector<Core::LinAlg::Matrix<4, 1>>>(numgp);

  for (int gp = 0; gp < numgp; gp++)
  {
    Core::LinAlg::Matrix<3, 1> alin;
    extract_from_pack(buffer, alin);
    a1_->at(gp) = alin;
    extract_from_pack(buffer, alin);
    a2_->at(gp) = alin;
    extract_from_pack(buffer, alin);
    a3_->at(gp) = alin;
    extract_from_pack(buffer, alin);
    a4_->at(gp) = alin;
    extract_from_pack(buffer, alin);
    vismassstress_->at(gp) = alin;
    double a;
    extract_from_pack(buffer, a);
    refmassdens_->at(gp) = a;
    extract_from_pack(buffer, alin);
    visrefmassdens_->at(gp) = alin;
    Core::LinAlg::Matrix<4, 1> pre;
    extract_from_pack(buffer, pre);
    localprestretch_->at(gp) = pre;
    extract_from_pack(buffer, pre);
    localhomstress_->at(gp) = pre;
  }
  double basal;
  extract_from_pack(buffer, basal);
  massprodbasal_ = basal;
  extract_from_pack(buffer, basal);
  homradius_ = basal;

  int delsize = 0;
  extract_from_pack(buffer, delsize);
  deletemass_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(delsize);
  for (int iddel = 0; iddel < delsize; iddel++)
  {
    Core::LinAlg::Matrix<3, 1> deldata;
    extract_from_pack(buffer, deldata);
    deletemass_->at(iddel) = deldata;
  }

  // unpack history
  extract_from_pack(buffer, delsize);
  minindex_ = delsize;
  int sizehistory;
  extract_from_pack(buffer, sizehistory);
  history_ = std::make_shared<std::vector<ConstraintMixtureHistory>>(sizehistory);
  for (int idpast = 0; idpast < sizehistory; idpast++)
  {
    history_->at(idpast).unpack(buffer);
  }



  /*
  double oldesttime = 0.0;
  double acttime = 0.0;
  double tempdt = 0.0;
  history_->begin()->get_time(&oldesttime, &tempdt);
  history_->at(sizehistory-2).get_time(&acttime, &tempdt);
  double intdegr = 0.0;
  double degrtime = 0.0;
  double degrdt = 0.0;
  for (int idpast = 0; idpast < sizehistory -1; idpast++)
  {
    double degr = 0.0;
    history_->at(idpast).get_time(&degrtime, &degrdt);
    double timeloc = 0.0;
    double dtloc = 0.0;
    history_->at(idpast+1).get_time(&timeloc, &dtloc);
    degrdt = dtloc;
    Degradation(acttime-degrtime, degr);
    intdegr += degr * degrdt;
  }

  for (int idpast = 0; idpast < sizehistory-1; idpast++)
  {
    double temptime = 0.0;
    history_->at(idpast).get_time(&temptime, &tempdt);
    for (int idgauss = 0; idgauss < numgp; idgauss++)
    {
      Core::LinAlg::Matrix<4,1> stretchtemp(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<4,1> stretchact(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<4,1> stretchold(Core::LinAlg::Initialization::zero);
      history_->at(sizehistory-2).get_stretches(idgauss, &stretchact);
      history_->begin()->get_stretches(idgauss, &stretchold);
      stretchtemp.update(stretchact);
      // linear interpolated stretch
      //double scalar = (temptime - acttime) / (oldesttime - acttime);
      //stretchtemp.update(scalar,stretchold,1.0);
      //stretchtemp.update(-scalar,stretchact,1.0);
      // modify stretch
      //history_->at(idpast).set_stretches(idgauss,stretchtemp);
      // distribute mass equally
      double massprodbasal = (refmassdens_->at(idgauss) - (params_->phimuscle_ +
  params_->phielastin_) * params_->density_) / 4.0 / intdegr; Core::LinAlg::Matrix<4,1>
  masstemp(true); masstemp.put_scalar(massprodbasal);
  history_->at(idpast).set_mass(idgauss,masstemp);
    }
  }
  std::cout << "Unpack called, history of mass/stretch is lost" << std::endl;
  */

  return;
}

/*----------------------------------------------------------------------*
 |  Setup                                         (public)         12/10|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  if (params_->integration_ != "Implicit" && params_->integration_ != "Explicit")
    FOUR_C_THROW("unknown option for integration");
  if (params_->growthforce_ != "Single" && params_->growthforce_ != "All" &&
      params_->growthforce_ != "ElaCol")
    FOUR_C_THROW("unknown driving force for growth");
  if (params_->elastindegrad_ != "None" && params_->elastindegrad_ != "Rectangle" &&
      params_->elastindegrad_ != "Time" && params_->elastindegrad_ != "RectanglePlate" &&
      params_->elastindegrad_ != "Wedge" && params_->elastindegrad_ != "Circles" &&
      params_->elastindegrad_ != "InvEla")
    FOUR_C_THROW("unknown option for elastin degradation");
  if (params_->massprodfunc_ != "Lin" && params_->massprodfunc_ != "CosCos")
    FOUR_C_THROW("unknown option for mass production function");

  // visualization
  vismassstress_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  refmassdens_ = std::make_shared<std::vector<double>>(numgp);
  visrefmassdens_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  // homeostatic prestretch of collagen fibers
  localprestretch_ = std::make_shared<std::vector<Core::LinAlg::Matrix<4, 1>>>(numgp);
  localhomstress_ = std::make_shared<std::vector<Core::LinAlg::Matrix<4, 1>>>(numgp);

  for (int gp = 0; gp < numgp; gp++)
  {
    vismassstress_->at(gp)(0) = 0.0;
    vismassstress_->at(gp)(1) = 0.0;
    vismassstress_->at(gp)(2) = 0.0;
    refmassdens_->at(gp) = params_->density_;
    visrefmassdens_->at(gp)(0) =
        params_->density_ * (1.0 - params_->phielastin_ - params_->phimuscle_) / 4.0;
    visrefmassdens_->at(gp)(1) =
        params_->density_ * (1.0 - params_->phielastin_ - params_->phimuscle_) / 4.0;
    visrefmassdens_->at(gp)(2) =
        params_->density_ * (1.0 - params_->phielastin_ - params_->phimuscle_) / 4.0;
    //    visrefmassdens_->at(gp)(0) = params_->density_*(1.0 - params_->phielastin_ -
    //    params_->phimuscle_)/10.0; visrefmassdens_->at(gp)(1) = params_->density_*(1.0 -
    //    params_->phielastin_ - params_->phimuscle_)/10.0; visrefmassdens_->at(gp)(2) =
    //    params_->density_*(1.0 - params_->phielastin_ - params_->phimuscle_)/5.0*2.0;
  }

  deletemass_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(0);

  // history
  reset_all(numgp);

  // fiber vectors
  a1_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  a2_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  a3_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);
  a4_ = std::make_shared<std::vector<Core::LinAlg::Matrix<3, 1>>>(numgp);

  // read local (cylindrical) cosy-directions at current element
  FOUR_C_ASSERT_ALWAYS(coord_system, "Require coordinate system parameters (RAD, AXI, CIR).");

  Core::LinAlg::Matrix<3, 3> locsys;
  // basis is local cosy with third vec e3 = circumferential dir and e2 = axial dir
  double radnorm = Core::LinAlg::norm2(coord_system.value().element_system[0]);
  double axinorm = Core::LinAlg::norm2(coord_system.value().element_system[1]);
  double cirnorm = Core::LinAlg::norm2(coord_system.value().element_system[2]);

  radnorm = sqrt(radnorm);
  axinorm = sqrt(axinorm);
  cirnorm = sqrt(cirnorm);
  for (int i = 0; i < 3; i++)
  {
    locsys(i, 0) = coord_system.value().element_system[0](i) / radnorm;
    locsys(i, 1) = coord_system.value().element_system[1](i) / axinorm;
    locsys(i, 2) = coord_system.value().element_system[2](i) / cirnorm;
  }

  const double gamma = (45.0 * std::numbers::pi) / 180.;  // angle for diagonal fibers

  for (int gp = 0; gp < numgp; gp++)
  {
    for (int i = 0; i < 3; i++)
    {
      // a1 = e3, circumferential direction
      a1_->at(gp)(i) = locsys(i, 2);
      // a2 = e2, axial direction
      a2_->at(gp)(i) = locsys(i, 1);
      // a3 = cos gamma e3 + sin gamma e2
      a3_->at(gp)(i) = cos(gamma) * locsys(i, 2) + sin(gamma) * locsys(i, 1);
      // a4 = cos gamma e3 - sin gamma e2
      a4_->at(gp)(i) = cos(gamma) * locsys(i, 2) - sin(gamma) * locsys(i, 1);
    }
  }

  isinit_ = true;
}

/*----------------------------------------------------------------------*
 |  ResetAll                                      (public)         03/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::reset_all(const int numgp)
{
  // homeostatic variables
  for (int gp = 0; gp < numgp; gp++)
  {
    if (params_->numhom_ == 1)
    {
      localprestretch_->at(gp).put_scalar(params_->prestretchcollagen_[0]);
      localhomstress_->at(gp).put_scalar(params_->homstress_[0]);
    }
    else if (params_->numhom_ == 3)
    {
      localprestretch_->at(gp)(0) = params_->prestretchcollagen_[0];
      localprestretch_->at(gp)(1) = params_->prestretchcollagen_[1];
      localprestretch_->at(gp)(2) = params_->prestretchcollagen_[2];
      localprestretch_->at(gp)(3) = params_->prestretchcollagen_[2];
      localhomstress_->at(gp)(0) = params_->homstress_[0];
      localhomstress_->at(gp)(1) = params_->homstress_[1];
      localhomstress_->at(gp)(2) = params_->homstress_[2];
      localhomstress_->at(gp)(3) = params_->homstress_[2];
    }
    else
      FOUR_C_THROW("wrong number of homeostatic variables");
  }
  homradius_ = params_->homradius_;

  // history
  const Teuchos::ParameterList& timeintegr =
      Global::Problem::instance()->structural_dynamic_params();
  double dt = timeintegr.get<double>("TIMESTEP");
  int firstiter = 0;
  if (params_->integration_ == "Explicit") firstiter = 1;
  int numpast = 0;
  if (abs(round(params_->lifetime_ / dt) - params_->lifetime_ / dt) < 1.0e-8)
  {
    numpast = static_cast<int>(round(params_->lifetime_ / dt)) + firstiter;
  }
  else
  {
    numpast = static_cast<int>(ceil(params_->lifetime_ / dt)) + firstiter;
  }
  if (params_->degoption_ == "Exp" || params_->degoption_ == "ExpVar")
  {
    double taumax = -log(params_->degtol_) / log(2.0) * params_->lifetime_;
    numpast = static_cast<int>(round(taumax / dt)) + firstiter;
  }
  minindex_ = 0;

  {
    const Inpar::Solid::PreStress pstype = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
        Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
    const double pstime =
        Global::Problem::instance()->structural_dynamic_params().get<double>("PRESTRESSTIME");

    const double currentTime = params_->starttime_ + dt;
    // prestress time
    if (pstype == Inpar::Solid::PreStress::mulf && currentTime <= pstime + 1.0e-15)
    {
      FOUR_C_THROW("MULF is only working for PRESTRESSTIME smaller than STARTTIME!");
    }
  }

  // basal mass production rate determined by DENS, PHIE and degradation function
  double intdegr = 0.0;
  for (int idpast = 0; idpast < numpast - firstiter; idpast++)
  {
    double degr = 0.0;
    degradation((numpast - 1 - idpast) * dt, degr);
    intdegr += degr * dt;
  }
  massprodbasal_ =
      (1.0 - params_->phimuscle_ - params_->phielastin_) * params_->density_ / 4.0 / intdegr;
  //  massprodbasal_ = (1.0 - params_->phimuscle_ - params_->phielastin_) * params_->density_ / 10.0
  //  / intdegr;

  // history
  history_ = std::make_shared<std::vector<ConstraintMixtureHistory>>(numpast);
  bool expvar = false;
  if (params_->degoption_ == "ExpVar") expvar = true;
  for (int idpast = 0; idpast < numpast; idpast++)
  {
    history_->at(idpast).setup(numgp, massprodbasal_, expvar);
    history_->at(idpast).set_time(dt - (numpast - 1 - idpast) * dt, dt);
  }
}

/*----------------------------------------------------------------------*
 |  Update internal variables                     (public)         12/10|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::update()
{
  // set mass of damaged collagen to zero
  // has to be done before history vector is changed
  // do not erase as they are needed for computation of massprodbasal
  int delsize = deletemass_->size();
  for (int iddel = 0; iddel < delsize; iddel++)
  {
    int gpdel = deletemass_->at(iddel)(0);
    int idpastdel = deletemass_->at(iddel)(1);
    int idfiberdel = deletemass_->at(iddel)(2);
    history_->at(idpastdel).set_mass(gpdel, 0.0, idfiberdel);
  }

  // erase oldest collagen
  int numgp = history_->at(0).num_gp();
  int sizehistory = history_->size();
  double deptime = 0.0;
  double depdt = 0.0;
  history_->back().get_time(&deptime, &depdt);
  bool expvar = false;
  if (params_->degoption_ == "ExpVar") expvar = true;

  // just do update in case of growth
  if (deptime > params_->starttime_ + 1.0e-11)
  {
    if (!params_->storehistory_)
    {
      // delete just the steps that surely won't be needed, especially with a smaller timestep later
      // thus reference time is deposition time of last collagen fibers
      double acttime = deptime;
      int eraseiter = 0;
      history_->at(eraseiter).get_time(&deptime, &depdt);
      double degrad = 0.0;
      degradation(acttime - deptime, degrad);
      while (degrad < params_->degtol_ && eraseiter < sizehistory)
      {
        eraseiter += 1;
        history_->at(eraseiter).get_time(&deptime, &depdt);
        degradation(acttime - deptime, degrad);
      }
      if (eraseiter > 0)
      {
        history_->erase(history_->begin(), history_->begin() + eraseiter);
        // std::cout << "erased " << eraseiter << " history variables" << std::endl;
      }
    }

    // append new collagen
    ConstraintMixtureHistory newhis;
    newhis.setup(numgp, massprodbasal_, expvar);
    // it is very important to set time and dt to 0.0
    // this makes it clear that this step was created in update and has no reliable content
    // they are not known here either
    // in evaluate_stress this is important, if called after Update
    newhis.set_time(0.0, 0.0);
    history_->push_back(newhis);
  }
  else  // time < starttime, adapt deposition time, set history if wanted
  {
    // SetHistory
    if (params_->initstretch_ == "SetConstantHistory")
    {
      for (int igp = 0; igp < numgp; igp++)
      {
        // set stretch
        Core::LinAlg::Matrix<4, 1> actstretch(Core::LinAlg::Initialization::zero);
        history_->back().get_stretches(igp, &actstretch);
        // special treatment for first timestep
        if (deptime == depdt)
        {
          Core::LinAlg::Matrix<4, 1> ones(Core::LinAlg::Initialization::zero);
          ones.put_scalar(1.0);
          history_->back().set_stretches(igp, ones);
        }
        for (int istep = 0; istep < sizehistory; istep++)
        {
          Core::LinAlg::Matrix<4, 1> oldstretch(Core::LinAlg::Initialization::zero);
          history_->at(istep).get_stretches(igp, &oldstretch);
          oldstretch.elementwise_divide(actstretch);
          history_->at(istep).set_stretches(igp, oldstretch);
        }

        // set mass
        Core::LinAlg::Matrix<4, 1> actmass(Core::LinAlg::Initialization::zero);
        history_->back().get_mass(igp, &actmass);
        if (abs(actmass(0) - massprodbasal_) > 1.0e-8 ||
            abs(actmass(1) - massprodbasal_) > 1.0e-8 ||
            abs(actmass(2) - massprodbasal_) > 1.0e-8 || abs(actmass(3) - massprodbasal_) > 1.0e-8)
        {
          Core::LinAlg::Matrix<4, 1> ones(Core::LinAlg::Initialization::zero);
          ones.put_scalar(1.0);
          Core::LinAlg::Matrix<1, 1> summass(Core::LinAlg::Initialization::zero);
          summass.multiply_tn(ones, actmass);
          Core::LinAlg::Matrix<4, 1> newmass(Core::LinAlg::Initialization::zero);
          newmass.put_scalar(4.0 * massprodbasal_);
          if (summass(0) > 0.0 + 1.0e-12)
          {
            actmass.scale(1.0 / summass(0));
            newmass.elementwise_multiply(actmass);
          }
          for (int istep = 0; istep < sizehistory; istep++)
          {
            history_->at(istep).set_mass(igp, newmass);
          }
          actmass.put_scalar(massprodbasal_);
          history_->back().set_mass(igp, actmass);
        }
      }
    }

    if (params_->initstretch_ == "SetLinearHistory")
    {
      for (int igp = 0; igp < numgp; igp++)
      {
        Core::LinAlg::Matrix<4, 1> actstretch(Core::LinAlg::Initialization::zero);
        history_->back().get_stretches(igp, &actstretch);
        Core::LinAlg::Matrix<4, 1> actmass(Core::LinAlg::Initialization::zero);
        history_->back().get_mass(igp, &actmass);
        if (abs(actmass(0) - massprodbasal_) < 1.0e-8 &&
            abs(actmass(1) - massprodbasal_) < 1.0e-8 &&
            abs(actmass(2) - massprodbasal_) < 1.0e-8 && abs(actmass(3) - massprodbasal_) < 1.0e-8)
        {
          if (abs(actstretch(0) - 1.0) > 1.0e-8 || abs(actstretch(1) - 1.0) > 1.0e-8 ||
              abs(actstretch(2) - 1.0) > 1.0e-8 || abs(actstretch(3) - 1.0) > 1.0e-8)
          {
            // set stretch
            for (int istep = 0; istep < sizehistory - 1; istep++)
            {
              Core::LinAlg::Matrix<4, 1> oldstretch(Core::LinAlg::Initialization::zero);
              for (int idfiber = 0; idfiber < 4; idfiber++)
                oldstretch(idfiber) =
                    (actstretch(idfiber) - 1.0) * (1.0 - istep / (sizehistory - 1.0)) + 1.0;
              Core::LinAlg::Matrix<4, 1> ones(Core::LinAlg::Initialization::zero);
              ones.put_scalar(1.0);
              ones.elementwise_divide(oldstretch);
              history_->at(istep).set_stretches(igp, ones);
            }
            actstretch.put_scalar(1.0);
            history_->back().set_stretches(igp, actstretch);
          }
        }
        else
        {
          // set mass
          Core::LinAlg::Matrix<4, 1> ones(Core::LinAlg::Initialization::zero);
          ones.put_scalar(1.0);
          Core::LinAlg::Matrix<1, 1> summass(Core::LinAlg::Initialization::zero);
          summass.multiply_tn(ones, actmass);
          Core::LinAlg::Matrix<4, 1> newmass(Core::LinAlg::Initialization::zero);
          newmass.put_scalar(4.0 * massprodbasal_);
          if (summass(0) > 0.0 + 1.0e-12)
          {
            actmass.scale(1.0 / summass(0));
            newmass.elementwise_multiply(actmass);
          }
          for (int istep = 0; istep < sizehistory; istep++)
          {
            history_->at(istep).set_mass(igp, newmass);
          }
          actmass.put_scalar(massprodbasal_);
          history_->back().set_mass(igp, actmass);
        }
      }
    }

    // just adopt deposition time, the rest stays the same
    double newtime = 0.0;
    double newdt = 0.0;
    history_->at(0).get_time(&newtime, &newdt);
    double degrad = 0.0;
    degradation(deptime - newtime, degrad);
    if (degrad < params_->degtol_)
    {
      for (int iter = 0; iter < sizehistory - 1; iter++)
      {
        history_->at(iter + 1).get_time(&newtime, &newdt);
        history_->at(iter).set_time(newtime, newdt);
      }
      history_->back().set_time(0.0, 0.0);
    }
    else if (deptime == depdt || (deptime == 2 * depdt && params_->integration_ == "Implicit"))
    {
      // special case of first time step
      ConstraintMixtureHistory newhis;
      newhis.setup(numgp, massprodbasal_, expvar);
      newhis.set_time(0.0, 0.0);
      history_->push_back(newhis);
    }
    else
    {
      FOUR_C_THROW(
          "You should not change your timestep size in the case time < starttime! {}", deptime);
    }
  }  // time < starttime
}

/*----------------------------------------------------------------------*
 |  Reset internal variables                      (public)         01/12|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::reset_step()
{
  history_->back().set_time(0.0, 0.0);
  if (params_->degoption_ == "ExpVar")
    FOUR_C_THROW("variable degradation not combinable with adaptive time stepping");
}

/*----------------------------------------------------------------------*
 |  Evaluate                                      (public)         12/10|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);
  const Core::LinAlg::Matrix<6, 1> glstrain_mat =
      Core::LinAlg::make_strain_like_voigt_matrix(glstrain);

  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

  // map in GetParameter can now calculate LID, so we do not need it here   05/2017 birzle
  // get element lID incase we have element specific material parameters
  // int eleID = Global::Problem::instance()->GetDis("structure")->ElementColMap()->LID(eleGID);
  const double growthfactor = params_->get_parameter(params_->growthfactor, eleGID);

  // get variables from params
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double dt = *context.time_step_size;
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  double time = *context.total_time;
  std::string action = Mat::get_or<std::string>(params, "action", "none");
  bool output = false;
  if (action == "calc_struct_stress") output = true;
  // int eleid = params.get<int>("eleID",0);
  double inner_radius = 0.0;
  if (params_->sheargrowthfactor_ > 0.0) inner_radius = params.get<double>("inner radius");
  double eps = 1.0e-11;

  // elastin degradation
  double elastin_survival = 1.0;
  if (time > params_->starttime_ + eps)
  {
    if (params_->elastindegrad_ == "Rectangle" || params_->elastindegrad_ == "RectanglePlate" ||
        params_->elastindegrad_ == "Wedge" || params_->elastindegrad_ == "Circles")
    {
      const auto& ele_center_reference_coordinates =
          params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");

      elastin_degradation(ele_center_reference_coordinates, elastin_survival);
    }
    else if (params_->elastindegrad_ == "Time")
    {
      elastin_survival = exp(-(time - params_->starttime_) * log(2.0) / 14600.0);
    }
  }
  else if (params_->initstretch_ == "SetHomeo" || params_->initstretch_ == "SetLinearHistory" ||
           params_->initstretch_ == "SetConstantHistory")
  {
    if (params_->elastindegrad_ == "Rectangle" || params_->elastindegrad_ == "RectanglePlate" ||
        params_->elastindegrad_ == "Wedge" || params_->elastindegrad_ == "Circles")
    {
      FOUR_C_ASSERT(context.ref_coords, "Reference coordinates not given in evaluation context.");
      const auto& gp_reference_coordinates = *context.ref_coords;

      elastin_degradation(gp_reference_coordinates, elastin_survival);
    }
  }
  if (params_->elastindegrad_ == "InvEla")
    elastin_survival = params_->get_parameter(params_->elastin_survival, eleGID);

  // stuff for collagen damage
  deletemass_->resize(0);

  // differentiate between explicit and implicit description
  int firstiter = 0;
  if (params_->integration_ == "Explicit") firstiter = 1;

  // determine minimal index
  if (params_->storehistory_)
  {
    double acttime = 0.0;
    minindex_ = 0;
    int sizehistory = history_->size();
    double temptime = 0.0;
    double tempdt = 0.0;
    history_->at(minindex_).get_time(&temptime, &tempdt);
    history_->back().get_time(&acttime, &tempdt);
    if (acttime == 0.0 || time <= acttime) acttime = time;
    double degrad = 0.0;
    degradation(acttime - temptime, degrad);
    while (degrad < params_->degtol_ && minindex_ < sizehistory - 1)
    {
      minindex_ += 1;
      history_->at(minindex_).get_time(&temptime, &tempdt);
      degradation(acttime - temptime, degrad);
    }
  }

  if (!output)
  {
    // set actual time as it might have changed after an restart etc. but just once
    double temptime = 0.0;
    double tempdt = 0.0;
    history_->back().get_time(&temptime, &tempdt);
    if (temptime == 0.0 && tempdt == 0.0)
    {
      int sizehistory = history_->size();
      history_->at(sizehistory - 2).get_time(&temptime, &tempdt);
      // for restart the function apply_force_internal calls the material with the old time
      // (i.e. time = temptime) thus make sure not to store it
      if (time > temptime + 1.0e-11)
      {
        history_->back().set_time(time, dt);
        temptime = time;
        // if you change your time step size the basal mass production rate changes
        // basal mass production rate determined by DENS, PHIE and degradation function
        double intdegr = 0.0;
        double degrtime = 0.0;
        double degrdt = 0.0;
        for (int idpast = minindex_; idpast < sizehistory - firstiter; idpast++)
        {
          double degr = 0.0;
          history_->at(idpast).get_time(&degrtime, &degrdt);
          if (firstiter == 1)
          {
            double timeloc = 0.0;
            double dtloc = 0.0;
            history_->at(idpast + 1).get_time(&timeloc, &dtloc);
            degrdt = dtloc;
          }
          degradation(time - degrtime, degr);
          intdegr += degr * degrdt;
        }
        massprodbasal_ =
            (1.0 - params_->phimuscle_ - params_->phielastin_) * params_->density_ / 4.0 / intdegr;
        //        massprodbasal_ = (1.0 - params_->phimuscle_ - params_->phielastin_) *
        //        params_->density_ / 10.0 / intdegr;

        // update degradation values (not possible in Update, as dt not known there)
        if (params_->degoption_ == "ExpVar")
        {
          if (params_->integration_ == "Implicit")
            FOUR_C_THROW("ExpVar not implemented for implicit time integration");
          int sizehistory = history_->size();
          // stretch of previous time step
          Core::LinAlg::Matrix<4, 1> actstretch(Core::LinAlg::Initialization::zero);
          history_->at(sizehistory - 2).get_stretches(gp, &actstretch);
          for (int idfiber = 0; idfiber < 4; idfiber++)
          {
            double homstrain = 0.0;
            double fac_cmat = 0.0;
            evaluate_single_fiber_scalars(
                localprestretch_->at(gp)(idfiber) * localprestretch_->at(gp)(idfiber), fac_cmat,
                homstrain);
            homstrain = homstrain * localprestretch_->at(gp)(idfiber);
            for (int idtime = minindex_; idtime < sizehistory - 2; idtime++)
            {
              Core::LinAlg::Matrix<4, 1> depstretch(Core::LinAlg::Initialization::zero);
              history_->at(idtime).get_stretches(gp, &depstretch);
              double strain = 0.0;
              double fac_cmat = 0.0;
              double stretch =
                  localprestretch_->at(gp)(idfiber) * actstretch(idfiber) / depstretch(idfiber);
              double I4 = stretch * stretch;
              evaluate_single_fiber_scalars(I4, fac_cmat, strain);
              strain = strain * stretch;
              double olddegrad = 0.0;
              history_->at(idtime).get_var_degrad(gp, idfiber, &olddegrad);
              double newdegrad = exp(-(strain / homstrain - 1.0) * (strain / homstrain - 1.0) * dt *
                                     log(2.0) / params_->lifetime_);
              newdegrad = newdegrad * olddegrad;
              history_->at(idtime).set_var_degrad(gp, idfiber, newdegrad);
            }
          }
        }  // expvar
      }
    }
    else if (time > temptime + 1.0e-11)
    {
      // in remodeling time might be wrong depending on the time integration used
      // correct this for the computation but do not store it
      time = temptime;
    }

    //--------------------------------------------------------------------------------------
    // build identity tensor I
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) Id(i) = 1.0;
    // right Cauchy-Green Tensor  C = 2 * E + I
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(glstrain_mat);
    C.scale(2.0);
    C += Id;

    //--------------------------------------------------------------------------------------
    // compute actual collagen stretches
    Core::LinAlg::Matrix<4, 1> actstretch(Core::LinAlg::Initialization::zero);
    double actcollagenstretch =
        a1_->at(gp)(0) * a1_->at(gp)(0) * C(0) + a1_->at(gp)(1) * a1_->at(gp)(1) * C(1) +
        a1_->at(gp)(2) * a1_->at(gp)(2) * C(2) + a1_->at(gp)(0) * a1_->at(gp)(1) * C(3) +
        a1_->at(gp)(1) * a1_->at(gp)(2) * C(4) +
        a1_->at(gp)(0) * a1_->at(gp)(2) * C(5);  // = trace(A1:C)
    actcollagenstretch = sqrt(actcollagenstretch);
    actstretch(0) = actcollagenstretch;
    actcollagenstretch =
        a2_->at(gp)(0) * a2_->at(gp)(0) * C(0) + a2_->at(gp)(1) * a2_->at(gp)(1) * C(1) +
        a2_->at(gp)(2) * a2_->at(gp)(2) * C(2) + a2_->at(gp)(0) * a2_->at(gp)(1) * C(3) +
        a2_->at(gp)(1) * a2_->at(gp)(2) * C(4) +
        a2_->at(gp)(0) * a2_->at(gp)(2) * C(5);  // = trace(A2:C)
    actcollagenstretch = sqrt(actcollagenstretch);
    actstretch(1) = actcollagenstretch;
    actcollagenstretch =
        a3_->at(gp)(0) * a3_->at(gp)(0) * C(0) + a3_->at(gp)(1) * a3_->at(gp)(1) * C(1) +
        a3_->at(gp)(2) * a3_->at(gp)(2) * C(2) + a3_->at(gp)(0) * a3_->at(gp)(1) * C(3) +
        a3_->at(gp)(1) * a3_->at(gp)(2) * C(4) +
        a3_->at(gp)(0) * a3_->at(gp)(2) * C(5);  // = trace(A3:C)
    actcollagenstretch = sqrt(actcollagenstretch);
    actstretch(2) = actcollagenstretch;
    actcollagenstretch =
        a4_->at(gp)(0) * a4_->at(gp)(0) * C(0) + a4_->at(gp)(1) * a4_->at(gp)(1) * C(1) +
        a4_->at(gp)(2) * a4_->at(gp)(2) * C(2) + a4_->at(gp)(0) * a4_->at(gp)(1) * C(3) +
        a4_->at(gp)(1) * a4_->at(gp)(2) * C(4) +
        a4_->at(gp)(0) * a4_->at(gp)(2) * C(5);  // = trace(A4:C)
    actcollagenstretch = sqrt(actcollagenstretch);
    actstretch(3) = actcollagenstretch;

    // store them
    // time <= starttime needs further considerations
    if (time > params_->starttime_ + eps)
    {
      history_->back().set_stretches(gp, actstretch);
    }
    else if (params_->initstretch_ == "Homeo")
    {  // this is not working for all material parameters
      int numsteps = history_->size();
      for (int i = 0; i < numsteps; i++) history_->at(i).set_stretches(gp, actstretch);
    }
    else if (params_->initstretch_ == "SetConstantHistory" &&
             time > (0.9 * params_->starttime_ - dt + 1.0e-12) &&
             time <= (0.9 * params_->starttime_ + 1.0e-12))
    {
      Core::LinAlg::Matrix<4, 1> tempstretch(actstretch);
      for (int i = 0; i < 4; i++)
      {
        if (tempstretch(i) < 1.0) tempstretch(i) = 1.0;
      }
      history_->back().set_stretches(gp, tempstretch);
    }
    else if (params_->initstretch_ == "SetLinearHistory" &&
             time <= (0.9 * params_->starttime_ + 1.0e-12) &&
             time > (0.9 * params_->starttime_ - dt + 1.0e-12))
    {
      Core::LinAlg::Matrix<4, 1> tempstretch(actstretch);
      history_->back().set_stretches(gp, tempstretch);
    }

    // set prestretch according to time curve or adapt prestretch
    if (time < params_->starttime_ - eps && params_->timecurve_ != 0)
    {
      // increase prestretch according to time curve
      int curvenum = params_->timecurve_;
      double curvefac = 1.0;
      // numbering starts from zero here, thus use curvenum-1
      if (curvenum)
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curvenum)
                       .evaluate(time);
      if (curvefac > (1.0 + eps) || curvefac < (0.0 - eps))
        FOUR_C_THROW("correct your time curve for prestretch, just values in [0,1] are allowed {}",
            curvefac);
      if (params_->numhom_ == 1)
      {
        double prestretch = 1.0 + (params_->prestretchcollagen_[0] - 1.0) * curvefac;
        localprestretch_->at(gp).put_scalar(prestretch);
      }
      else
      {
        double prestretch = 1.0 + (params_->prestretchcollagen_[0] - 1.0) * curvefac;
        localprestretch_->at(gp)(0) = prestretch;
        prestretch = 1.0 + (params_->prestretchcollagen_[1] - 1.0) * curvefac;
        localprestretch_->at(gp)(1) = prestretch;
        prestretch = 1.0 + (params_->prestretchcollagen_[2] - 1.0) * curvefac;
        localprestretch_->at(gp)(2) = prestretch;
        localprestretch_->at(gp)(3) = prestretch;
      }
    }
    else if (abs(time - params_->starttime_) < eps && params_->initstretch_ == "UpdatePrestretch")
    {
      // use current stretch as prestretch
      if (params_->numhom_ == 1)
        localprestretch_->at(gp).update(params_->prestretchcollagen_[0], actstretch);
      else
      {
        localprestretch_->at(gp)(0) = params_->prestretchcollagen_[0] * actstretch(0);
        localprestretch_->at(gp)(1) = params_->prestretchcollagen_[1] * actstretch(1);
        localprestretch_->at(gp)(2) = params_->prestretchcollagen_[2] * actstretch(2);
        localprestretch_->at(gp)(3) = params_->prestretchcollagen_[2] * actstretch(3);
      }
      // adopt deposition stretch
      int numsteps = history_->size();
      for (int i = 0; i < numsteps; i++) history_->at(i).set_stretches(gp, actstretch);
      homradius_ = inner_radius;
    }

    // start in every iteration from the original value, this is important for implicit only
    Core::LinAlg::Matrix<4, 1> massprodstart(Core::LinAlg::Initialization::zero);
    for (int id = 0; id < 4; id++) massprodstart(id) = massprodbasal_;
    //    massprodstart(2) = massprodstart(2)*4;
    //    massprodstart(3) = massprodstart(3)*4;
    history_->back().set_mass(gp, massprodstart);

    evaluate_stress(&glstrain_mat, gp, &cmat_view, &stress_view, firstiter, time, elastin_survival);

    //--------------------------------------------------------------------------------------
    // compute new deposition rates
    // either for future use or just for visualization
    Core::LinAlg::Matrix<4, 1> massstress(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<4, 1> massprodcomp(Core::LinAlg::Initialization::zero);
    if (params_->growthforce_ == "All")
    {
      mass_production(gp, Core::LinAlg::make_matrix_view(*defgrad),
          Core::LinAlg::make_stress_like_voigt_view(stress), &massstress, inner_radius,
          &massprodcomp, growthfactor);
    }
    else if (params_->growthforce_ == "ElaCol")
    {
      double masstemp = 0.0;
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stresstemp(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmattemp(
          Core::LinAlg::Initialization::zero);

      // 1st step: elastin
      //==========================
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Siso(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatiso(
          Core::LinAlg::Initialization::zero);
      evaluate_elastin(C, &cmatiso, &Siso, time, &masstemp, elastin_survival);
      stresstemp = Siso;
      cmattemp = cmatiso;

      // 2nd step: collagen
      //==========================
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a1_->at(gp), &masstemp, firstiter, time, 0);
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a2_->at(gp), &masstemp, firstiter, time, 1);
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a3_->at(gp), &masstemp, firstiter, time, 2);
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a4_->at(gp), &masstemp, firstiter, time, 3);

      mass_production(gp, Core::LinAlg::make_matrix_view(*defgrad), stresstemp, &massstress,
          inner_radius, &massprodcomp, growthfactor);
    }
    else
    {
      double massstresstemp = 0.0;
      double massprodtemp = 0.0;
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stresstemp(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmattemp(
          Core::LinAlg::Initialization::zero);
      double masstemp = 0.0;
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a1_->at(gp), &masstemp, firstiter, time, 0);
      mass_production_single_fiber(gp, Core::LinAlg::make_matrix_view(*defgrad), stresstemp,
          &massstresstemp, inner_radius, &massprodtemp, a1_->at(gp), 0, growthfactor);
      massstress(0) = massstresstemp;
      massprodcomp(0) = massprodtemp;
      stresstemp.put_scalar(0.0);
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a2_->at(gp), &masstemp, firstiter, time, 1);
      mass_production_single_fiber(gp, Core::LinAlg::make_matrix_view(*defgrad), stresstemp,
          &massstresstemp, inner_radius, &massprodtemp, a2_->at(gp), 1, growthfactor);
      massstress(1) = massstresstemp;
      massprodcomp(1) = massprodtemp;
      stresstemp.put_scalar(0.0);
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a3_->at(gp), &masstemp, firstiter, time, 2);
      mass_production_single_fiber(gp, Core::LinAlg::make_matrix_view(*defgrad), stresstemp,
          &massstresstemp, inner_radius, &massprodtemp, a3_->at(gp), 2, growthfactor);
      massstress(2) = massstresstemp;
      massprodcomp(2) = massprodtemp;
      stresstemp.put_scalar(0.0);
      evaluate_fiber_family(
          C, gp, &cmattemp, &stresstemp, a4_->at(gp), &masstemp, firstiter, time, 3);
      mass_production_single_fiber(gp, Core::LinAlg::make_matrix_view(*defgrad), stresstemp,
          &massstresstemp, inner_radius, &massprodtemp, a4_->at(gp), 3, growthfactor);
      massstress(3) = massstresstemp;
      massprodcomp(3) = massprodtemp;
    }

    // set new homstress
    if (abs(time - params_->starttime_) < eps && params_->initstretch_ == "UpdatePrestretch")
    {
      localhomstress_->at(gp).update(massstress);
      // perhaps add time < params_->starttime_? correction factor actstretch needed?
    }

    if (time > params_->starttime_ + eps &&
        (growthfactor != 0.0 || params_->sheargrowthfactor_ != 0.0))
    {
      // start values for local Newton iteration are computed
      // distinguish between explicit and implicit integration
      if (params_->integration_ == "Explicit")
      {
        history_->back().set_mass(gp, massprodcomp);
        vismassstress_->at(gp)(0) = massstress(0);
        vismassstress_->at(gp)(1) = massstress(1);
        vismassstress_->at(gp)(2) = massstress(2);
      }
      else
      {
        if (params_->massprodfunc_ != "Lin" || params_->initstretch_ == "SetConstantHistory" ||
            params_->initstretch_ == "SetLinearHistory")
          FOUR_C_THROW(
              "Your desired option of elastin degradation, mass production function or "
              "initstretch\n is not implemented in implicit time integration");
        if (params_->growthforce_ == "All")
        {
          evaluate_implicit_all(Core::LinAlg::make_matrix_view(*defgrad), &glstrain_mat, gp,
              &cmat_view, &stress_view, dt, time, massprodcomp, massstress, elastin_survival,
              growthfactor);
        }
        else if (params_->growthforce_ == "Single")
        {
          evaluate_implicit_single(Core::LinAlg::make_matrix_view(*defgrad), &glstrain_mat, gp,
              &cmat_view, &stress_view, dt, time, elastin_survival, growthfactor);
        }
        else if (params_->growthforce_ == "ElaCol")
        {
          FOUR_C_THROW("GROWTHFORCE ElaCol not implemented for implicit integration");
        }
      }
    }
    else
    {
      // visualization of massstresss for the other cases
      vismassstress_->at(gp)(0) = massstress(0);
      vismassstress_->at(gp)(1) = massstress(1);
      vismassstress_->at(gp)(2) = massstress(2);
      if ((params_->initstretch_ == "SetConstantHistory" ||
              params_->initstretch_ == "SetLinearHistory") &&
          time > (0.6 * params_->starttime_ + 1.0e-12) &&
          time <= (0.9 * params_->starttime_ - dt + 1.0e-12))
        history_->back().set_mass(gp, massprodcomp);
    }
  }
  else
  {
    // in case of output everything is fully converged, we just have to evaluate stress etc.
    // should be independent of order of update and output, as new steps are set with dt = 0.0
    // and oldest fibers are carefully erased
    double temptime = 0.0;
    double tempdt = 0.0;
    history_->back().get_time(&temptime, &tempdt);
    if (time != temptime)
    {
      if (temptime == 0.0)
      {
        int size = history_->size();
        history_->at(size - 2).get_time(&temptime, &tempdt);
        evaluate_stress(
            &glstrain_mat, gp, &cmat_view, &stress_view, firstiter, temptime, elastin_survival);
        FOUR_C_THROW("has to be checked, update called before output");
      }
      else
        FOUR_C_THROW(
            "times do not match: {} actual time, {} deposition time of last fiber", time, temptime);
    }
    else
    {
      evaluate_stress(
          &glstrain_mat, gp, &cmat_view, &stress_view, firstiter, time, elastin_survival);
    }
  }
}  // Evaluate

/*----------------------------------------------------------------------*
 |  evaluate_stress                                (private)        01/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::evaluate_stress(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,
    const int gp, Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, const int firstiter, double time,
    double elastin_survival)
{
  //--------------------------------------------------------------------------------------
  // some variables
  double density = params_->density_;
  double currmassdens = 0.0;

  //--------------------------------------------------------------------------------------
  // build identity tensor I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) Id(i) = 1.0;
  // right Cauchy-Green Tensor  C = 2 * E + I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(*glstrain);
  C.scale(2.0);
  C += Id;

  //--------------------------------------------------------------------------------------
  // calculate stress and elasticity matrix

  // 1st step: elastin
  //==========================
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Siso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatiso(Core::LinAlg::Initialization::zero);
  evaluate_elastin(C, &cmatiso, &Siso, time, &currmassdens, elastin_survival);
  (*stress) = Siso;
  (*cmat) = cmatiso;

  // 2nd step: collagen
  //==========================
  evaluate_fiber_family(C, gp, cmat, stress, a1_->at(gp), &currmassdens, firstiter, time, 0);

  evaluate_fiber_family(C, gp, cmat, stress, a2_->at(gp), &currmassdens, firstiter, time, 1);

  evaluate_fiber_family(C, gp, cmat, stress, a3_->at(gp), &currmassdens, firstiter, time, 2);

  evaluate_fiber_family(C, gp, cmat, stress, a4_->at(gp), &currmassdens, firstiter, time, 3);

  // 3rd step: smooth muscle
  //==========================
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Smus(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatmus(Core::LinAlg::Initialization::zero);
  evaluate_muscle(C, &cmatmus, &Smus, gp, &currmassdens);
  (*stress) += Smus;
  (*cmat) += cmatmus;

  // 4th step: volumetric part
  //==========================
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Svol(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatvol(Core::LinAlg::Initialization::zero);
  evaluate_volumetric(C, &cmatvol, &Svol, currmassdens, density);
  (*stress) += Svol;
  (*cmat) += cmatvol;

  // set actual mass density
  refmassdens_->at(gp) = currmassdens;
}

/*----------------------------------------------------------------------*
 |  evaluate_fiber_family                           (private)        05/11|
 *----------------------------------------------------------------------*
 strain energy function

 W^k = M^k(0)/rho*Q^k(0)*Psi^k + \int_0^t m^k(tau)/rho*q^k(t-tau)*Psi^k dtau

 */
void Mat::ConstraintMixture::evaluate_fiber_family(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,
    const int gp, Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, Core::LinAlg::Matrix<3, 1> a,
    double* currmassdens, const int firstiter, double time, const int idfiber)
{
  //--------------------------------------------------------------------------------------
  // some variables
  double prestretchcollagen = localprestretch_->at(gp)(idfiber);
  double density = params_->density_;
  int sizehistory = history_->size();
  double eps = 1.0e-11;
  if (idfiber != 3) visrefmassdens_->at(gp)(idfiber) = 0.0;

  //--------------------------------------------------------------------------------------
  // structural tensors in voigt notation
  // A = a x a
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> A;
  for (int i = 0; i < 3; i++) A(i) = a(i) * a(i);

  A(3) = a(0) * a(1);
  A(4) = a(1) * a(2);
  A(5) = a(0) * a(2);

  double I4 = A(0) * C(0) + A(1) * C(1) + A(2) * C(2) +
              1. * (A(3) * C(3) + A(4) * C(4) + A(5) * C(5));  // I4 = trace(A C)

  // variables for stress and cmat
  double fac_cmat = 0.0;
  double fac_stress = 0.0;

  //--------------------------------------------------------------------------------------
  // calculate stress and elasticity matrix
  for (int idpast = minindex_; idpast < sizehistory - firstiter; idpast++)
  {
    double deptime = 0.0;
    double depdt = 0.0;
    history_->at(idpast).get_time(&deptime, &depdt);
    // right dt for explicit integration & adaptation for special implicit step
    if (firstiter == 1)
    {
      double timeloc = 0.0;
      double dtloc = 0.0;
      history_->at(idpast + 1).get_time(&timeloc, &dtloc);
      depdt = dtloc;
    }
    else if (deptime >= (time - params_->lifetime_ - eps) &&
             (deptime - depdt) < (time - params_->lifetime_ - eps) &&
             (params_->degoption_ != "Exp" && params_->degoption_ != "ExpVar"))
    {
      depdt = deptime - time + params_->lifetime_;
    }

    Core::LinAlg::Matrix<4, 1> collstretch(Core::LinAlg::Initialization::zero);
    history_->at(idpast).get_stretches(gp, &collstretch);
    double stretch = prestretchcollagen / collstretch(idfiber);
    // prestretch of collagen fibers is not applied, might be reasonable combined with prestress
    if (params_->initstretch_ == "experimental" && deptime <= params_->starttime_ + eps)
      stretch = 1.0 / collstretch(idfiber);

    double I4_loc = I4;

    // linear distribution of stretch
    if (params_->initstretch_ == "SetLinearHistory" &&
        time <= (0.9 * params_->starttime_ + 1.0e-12))
    {
      I4_loc = ((sqrt(I4) - 1.0) * (1.0 - idpast / (sizehistory - 1.0)) + 1.0) *
               ((sqrt(I4) - 1.0) * (1.0 - idpast / (sizehistory - 1.0)) + 1.0);
      Core::LinAlg::Matrix<4, 1> tempstretch(Core::LinAlg::Initialization::zero);
      history_->at(idpast).get_stretches(gp, &tempstretch);
      if (abs(tempstretch(idfiber) - 1.0) > 1.0e-12)
        FOUR_C_THROW("linear stretch when stretch history has been modified");
    }

    I4_loc = I4_loc * stretch * stretch;  // account for prestretch and stretch at deposition time
    if (sqrt(I4_loc) > params_->damagestretch_)
    {
      Core::LinAlg::Matrix<3, 1> deldata(Core::LinAlg::Initialization::zero);
      deldata(0) = gp;
      deldata(1) = idpast;
      deldata(2) = idfiber;
      deletemass_->push_back(deldata);
    }

    double fac_cmat_loc = 0.0;
    double fac_stress_loc = 0.0;
    evaluate_single_fiber_scalars(I4_loc, fac_cmat_loc, fac_stress_loc);

    if (params_->initstretch_ == "SetLinearHistory" &&
        time <= (0.9 * params_->starttime_ + 1.0e-12))
      fac_cmat_loc = fac_cmat_loc *
                     ((sqrt(I4) - 1.0) * (1.0 - idpast / (sizehistory - 1.0)) + 1.0) *
                     (1.0 - idpast / (sizehistory - 1.0)) / sqrt(I4);

    double qdegrad = 0.0;
    degradation(time - deptime, qdegrad);
    if (params_->degoption_ == "ExpVar")
    {
      double vardegrad;
      history_->at(idpast).get_var_degrad(gp, idfiber, &vardegrad);
      qdegrad = qdegrad * vardegrad;
    }
    Core::LinAlg::Matrix<4, 1> collmass(Core::LinAlg::Initialization::zero);
    history_->at(idpast).get_mass(gp, &collmass);
    fac_stress +=
        fac_stress_loc * stretch * stretch * qdegrad * collmass(idfiber) / density * depdt;

    if (params_->initstretch_ == "Homeo" ||
        (idpast == sizehistory - 1 && time > params_->starttime_ + 1.0e-11))
    {
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Saniso_loc(A);
      Saniso_loc.scale(
          fac_stress_loc * stretch * stretch * qdegrad * collmass(idfiber) / density * depdt);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatanisoadd(
          Core::LinAlg::Initialization::zero);
      cmatanisoadd.multiply_nt(A, Saniso_loc);
      cmatanisoadd.scale(-2.0 / (collstretch(idfiber) * collstretch(idfiber)));
      (*cmat) += cmatanisoadd;
    }
    else
    {
      fac_cmat += fac_cmat_loc * stretch * stretch * stretch * stretch * qdegrad *
                  collmass(idfiber) / density * depdt;
    }

    (*currmassdens) += qdegrad * collmass(idfiber) * depdt;
    if (idfiber != 3) visrefmassdens_->at(gp)(idfiber) += qdegrad * collmass(idfiber) * depdt;
  }

  // matrices for stress and cmat
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Saniso(A);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmataniso(Core::LinAlg::Initialization::zero);
  cmataniso.multiply_nt(A, A);
  Saniso.scale(fac_stress);
  cmataniso.scale(fac_cmat);
  (*stress) += Saniso;
  (*cmat) += cmataniso;
}

/*----------------------------------------------------------------------*
 |  evaluate_single_fiber_scalars                    (private)        02/12|
 *----------------------------------------------------------------------*
 strain energy function

 Psi    = k1/(2.0*k2)*(exp(k2*(I_4 - 1.0)^2)-1.0)

 I_4 .. invariant accounting for the fiber direction
 */
void Mat::ConstraintMixture::evaluate_single_fiber_scalars(
    double I4, double& fac_cmat, double& fac_stress)
{
  const double k1 = params_->k1_;
  const double k2 = params_->k2_;

  //--------------------------------------------------------------------------------------
  // fibers can only stretch/compress down to a minimal value
  // look this up, I am not sure. There is nothing stated explicitly.
  // take material parameters of muscle cells
  double fib1_tension = 1.0;
  if (I4 < 1.0)
  {
    I4 = 1.0;
    fib1_tension = 0.0;
  }

  // stress
  const double exp1 = std::exp(k2 * (I4 - 1.) * (I4 - 1.));
  if (std::isinf(exp1)) FOUR_C_THROW("stretch in fiber direction is too high {}", sqrt(I4));
  fac_stress = 2. * (k1 * (I4 - 1.) * exp1);  // 2 dW/dI4

  // cmat
  fac_cmat = fib1_tension * 4.0 *
             (k1 * exp1 + 2.0 * k1 * k2 * (I4 - 1.0) * (I4 - 1.0) * exp1);  // 4 d^2Wf/dI4dI4
}

/*----------------------------------------------------------------------*
 |  evaluate_elastin                               (private)        12/10|
 *----------------------------------------------------------------------*
 strain energy function

 W    = 1/2 mue (I1-3) + mue / (2 beta) (I3^-beta -1)

*/
void Mat::ConstraintMixture::evaluate_elastin(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, double time, double* currmassdens,
    double elastin_survival)
{
  double density = params_->density_;
  double prestretchelastin = params_->prestretchelastin_;
  double refmassdenselastin = params_->phielastin_ * density;
  if (refmassdenselastin < 0.0 || refmassdenselastin > density)
    FOUR_C_THROW("mass fraction of elastin not in [0;1]");

  if (time < params_->starttime_ - 1.0e-11 && params_->timecurve_ != 0)
  {
    // increase prestretch according to time curve
    int curvenum = params_->timecurve_;
    double curvefac = 1.0;
    // numbering starts from zero here, thus use curvenum-1
    if (curvenum)
      curvefac = Global::Problem::instance()
                     ->function_by_id<Core::Utils::FunctionOfTime>(curvenum)
                     .evaluate(time);
    if (curvefac > 1.0 || curvefac < 0.0)
      FOUR_C_THROW(
          "correct your time curve for prestretch, just values in [0,1] are allowed {}", curvefac);
    prestretchelastin = 1.0 + (params_->prestretchelastin_ - 1.0) * curvefac;
  }
  // account for isotropic prestretch of elastin
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Ciso(C);
  Ciso.scale(prestretchelastin * prestretchelastin);

  double mue = params_->mue_;
  double nue = params_->nue_;
  double beta = nue / (1.0 - 2.0 * nue);

  // isotropic invariant
  const double I3 = Ciso(0) * Ciso(1) * Ciso(2) + 0.25 * Ciso(3) * Ciso(4) * Ciso(5) -
                    0.25 * Ciso(1) * Ciso(5) * Ciso(5) - 0.25 * Ciso(2) * Ciso(3) * Ciso(3) -
                    0.25 * Ciso(0) * Ciso(4) * Ciso(4);  // 3rd invariant, determinant

  //--------------------------------------------------------------------------------------
  // invert Ciso
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cisoinv(Core::LinAlg::Initialization::zero);
  Cisoinv(0) = Ciso(1) * Ciso(2) - 0.25 * Ciso(4) * Ciso(4);
  Cisoinv(1) = Ciso(0) * Ciso(2) - 0.25 * Ciso(5) * Ciso(5);
  Cisoinv(2) = Ciso(0) * Ciso(1) - 0.25 * Ciso(3) * Ciso(3);
  Cisoinv(3) = 0.25 * Ciso(5) * Ciso(4) - 0.5 * Ciso(3) * Ciso(2);
  Cisoinv(4) = 0.25 * Ciso(3) * Ciso(5) - 0.5 * Ciso(0) * Ciso(4);
  Cisoinv(5) = 0.25 * Ciso(3) * Ciso(4) - 0.5 * Ciso(5) * Ciso(1);
  Cisoinv.scale(1.0 / I3);

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Siso(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) Siso(i) = mue;

  double gamma2 = -mue * pow(I3, -beta);
  Siso.update(gamma2, Cisoinv, 1.0);
  Siso.scale(
      refmassdenselastin / density * prestretchelastin * prestretchelastin * elastin_survival);
  *stress = Siso;

  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatiso(Core::LinAlg::Initialization::zero);
  double delta6 = 2. * beta * mue * pow(I3, -beta);
  cmatiso.multiply_nt(delta6, Cisoinv, Cisoinv);
  double delta7 = 2. * mue * pow(I3, -beta);
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmatiso, Cisoinv, delta7);
  cmatiso.scale(refmassdenselastin / density * prestretchelastin * prestretchelastin *
                prestretchelastin * prestretchelastin * elastin_survival);
  *cmat = cmatiso;

  (*currmassdens) += refmassdenselastin;
}

/*----------------------------------------------------------------------*
 |  evaluate_muscle                                (private)        12/11|
 *----------------------------------------------------------------------*
 strain energy function

 Psi    = k1^m/(2.0*k2^m)*(exp(k2^m*(I_4 - 1.0)^2)-1.0)

        + S_{max} * (\lambda + 1/3*(\lambda_M-\lambda)^3/(\lambda_M-\lambda_0)^2

 I_4 .. invariant accounting for the fiber direction

*/
void Mat::ConstraintMixture::evaluate_muscle(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, const int gp, double* currmassdens)
{
  const double k1 = params_->k1muscle_;                        // 14175.0;
  const double k2 = params_->k2muscle_;                        // 8.5;
  const double prestretchmuscle = params_->prestretchmuscle_;  // 1.2;
  const double massfrac = params_->phimuscle_;                 // 0.2;

  //--------------------------------------------------------------------------------------
  // structural tensors in voigt notation
  // A = a x a
  Core::LinAlg::Matrix<3, 1> a = a1_->at(gp);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> A;
  for (int i = 0; i < 3; i++) A(i) = a(i) * a(i);

  A(3) = a(0) * a(1);
  A(4) = a(1) * a(2);
  A(5) = a(0) * a(2);

  double I4 = A(0) * C(0) + A(1) * C(1) + A(2) * C(2) +
              1. * (A(3) * C(3) + A(4) * C(4) + A(5) * C(5));  // I4 = trace(A C)

  if (massfrac > 0.0 + 1.0e-12 && k1 > 0.0 + 1.0e-12)
  {
    //++++++++++++++++++++++ passive part ++++++++++++++++++
    //--- determine 2nd Piola Kirchhoff stresses S -----------------------------------------
    double preI4 = I4 * prestretchmuscle * prestretchmuscle;
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Saniso(A);  // first compute S = 2 dW/dI4 A
    const double exp1 = std::exp(k2 * (preI4 - 1.) * (preI4 - 1.));
    if (std::isinf(exp1)) FOUR_C_THROW("stretch in fiber direction is too high");
    const double fib1 = 2. * (k1 * (preI4 - 1.) * exp1);  // 2 dW/dI4
    Saniso.scale(fib1);                                   // S

    // consider mass fraction and prestretch
    Saniso.scale(massfrac * prestretchmuscle * prestretchmuscle);

    *stress = Saniso;

    //--- do elasticity matrix -------------------------------------------------------------
    const double delta7 =
        4.0 * (k1 * exp1 + 2.0 * k1 * k2 * (preI4 - 1.0) * (preI4 - 1.0) * exp1);  // 4 d^2Wf/dI4dI4
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmataniso;
    for (int i = 0; i < 6; ++i)
    {
      for (int j = 0; j < 6; ++j)
      {
        cmataniso(i, j) = delta7 * A(i) * A(j);  // delta7 A x A
      }
    }
    // consider mass fraction and prestretch
    cmataniso.scale(
        massfrac * prestretchmuscle * prestretchmuscle * prestretchmuscle * prestretchmuscle);

    *cmat = cmataniso;
  }

  //++++++++++++++++++++++ active part ++++++++++++++++++
  // does not depend on mass fraction
  double lambda = sqrt(I4);
  double Smax = params_->Smax_;  // 50000;
  double lambda_M = 1.2;         // 1.2,1.4;
  double lambda_0 = 0.7;         // 0.7,0.8;
  // perhaps check if lambda is between lambda_M and lambda_0

  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Saniso(A);
  double facS = Smax / lambda *
                (1.0 - ((lambda_M - lambda) * (lambda_M - lambda) / (lambda_M - lambda_0) /
                           (lambda_M - lambda_0)));
  Saniso.scale(facS);
  *stress += Saniso;

  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmataniso(Core::LinAlg::Initialization::zero);
  double faccmat = -Smax / (lambda * lambda) *
                   ((1.0 - ((lambda_M - lambda) * (lambda_M - lambda) / (lambda_M - lambda_0) /
                               (lambda_M - lambda_0))) /
                           lambda -
                       2.0 * (lambda_M - lambda) / ((lambda_M - lambda_0) * (lambda_M - lambda_0)));
  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 6; ++j)
    {
      cmataniso(i, j) = faccmat * A(i) * A(j);
    }
  }
  *cmat += cmataniso;

  *currmassdens += massfrac * params_->density_;
}

/*----------------------------------------------------------------------*
 |  evaluate_volumetric                            (private)        12/10|
 *----------------------------------------------------------------------*
 strain energy function

 W    = 1/2 kappa (J-M(t)/M(0))^2

*/
void Mat::ConstraintMixture::evaluate_volumetric(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, double currMassDens, double refMassDens)
{
  double kappa = params_->kappa_;

  // isotropic invariant
  const double I3 = C(0) * C(1) * C(2) + 0.25 * C(3) * C(4) * C(5) - 0.25 * C(1) * C(5) * C(5) -
                    0.25 * C(2) * C(3) * C(3) -
                    0.25 * C(0) * C(4) * C(4);  // 3rd invariant, determinant
  if (I3 < 0.0) FOUR_C_THROW("fatal failure in constraint mixture artery wall material");

  const double J = sqrt(I3);                                  // determinant of F
  const double p = kappa * (J - currMassDens / refMassDens);  // dW_vol/dJ

  //--------------------------------------------------------------------------------------
  // invert C
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv(Core::LinAlg::Initialization::zero);
  Cinv(0) = C(1) * C(2) - 0.25 * C(4) * C(4);
  Cinv(1) = C(0) * C(2) - 0.25 * C(5) * C(5);
  Cinv(2) = C(0) * C(1) - 0.25 * C(3) * C(3);
  Cinv(3) = 0.25 * C(5) * C(4) - 0.5 * C(3) * C(2);
  Cinv(4) = 0.25 * C(3) * C(5) - 0.5 * C(0) * C(4);
  Cinv(5) = 0.25 * C(3) * C(4) - 0.5 * C(5) * C(1);
  Cinv.scale(1.0 / I3);

  //--- determine 2nd Piola Kirchhoff stresses S -----------------------------------------
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Svol(Core::LinAlg::Initialization::zero);
  // volumetric part J*kappa*(J-M(t)/M(0))*Cinv
  for (int i = 0; i < 6; i++) Svol(i) = J * p * Cinv(i);

  *stress = Svol;

  //--- do elasticity matrix -------------------------------------------------------------
  // cmatvol = J(p + J dp/dJ) Cinv x Cinv  -  2 J p Cinv o Cinv
  Core::LinAlg::FourTensorOperations::add_holzapfel_product((*cmat), Cinv, (-2 * J * p));
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j < 6; j++)
    {
      (*cmat)(i, j) += J * (p + J * kappa) * Cinv(i) * Cinv(j);
    }
  }
}

/*----------------------------------------------------------------------*
 |  mass_production                                (private)        01/11|
 *----------------------------------------------------------------------*
compute new deposition rates for all fiber families
driving force depends on S

m^k = mbasal * (1 + K * ( sqrt(a_i^T*C*S*C*S*C*a_i/detC)/lambda_k(t) /homeo - 1))

therefore we need C and S as matrices
*/
void Mat::ConstraintMixture::mass_production(const int gp, Core::LinAlg::Matrix<3, 3> defgrd,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S, Core::LinAlg::Matrix<4, 1>* massstress,
    double inner_radius, Core::LinAlg::Matrix<4, 1>* massprodcomp, double growthfactor)
{
  Core::LinAlg::Matrix<3, 3> Smatrix(Core::LinAlg::Initialization::zero);
  Smatrix(0, 0) = S(0);
  Smatrix(0, 1) = S(3);
  Smatrix(0, 2) = S(5);
  Smatrix(1, 0) = Smatrix(0, 1);
  Smatrix(1, 1) = S(1);
  Smatrix(1, 2) = S(4);
  Smatrix(2, 0) = Smatrix(0, 2);
  Smatrix(2, 1) = Smatrix(1, 2);
  Smatrix(2, 2) = S(2);
  // one has to use the defgrd here, as with EAS C != F^T F
  Core::LinAlg::Matrix<3, 3> Cmatrix(Core::LinAlg::Initialization::zero);
  Cmatrix.multiply_tn(defgrd, defgrd);
  double detC = Cmatrix.determinant();

  Core::LinAlg::Matrix<3, 3> temp1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp2(Core::LinAlg::Initialization::zero);
  temp1.multiply(Smatrix, Cmatrix);
  temp2.multiply(Cmatrix, temp1);
  temp1.put_scalar(0.0);
  temp1.multiply(Smatrix, temp2);
  temp2.put_scalar(0.0);
  temp2.multiply(1.0 / detC, Cmatrix, temp1);

  double sheardiff = 0.0;
  double sheargrowthfactor = params_->sheargrowthfactor_;
  if (sheargrowthfactor > 0.0)
    sheardiff =
        1.0 - homradius_ * homradius_ * homradius_ / (inner_radius * inner_radius * inner_radius);
  double maxmassprodfac = params_->maxmassprodfac_;

  // Fiber1
  Core::LinAlg::Matrix<3, 1> temp_vector(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1> temp_scalar(Core::LinAlg::Initialization::zero);
  temp_vector.multiply(Cmatrix, a1_->at(gp));
  temp_scalar.multiply_tn(a1_->at(gp), temp_vector);
  double currentstretch = sqrt(temp_scalar(0));
  temp_vector.multiply(temp2, a1_->at(gp));
  temp_scalar.multiply_tn(a1_->at(gp), temp_vector);
  double massstress1 = sqrt(temp_scalar(0)) / currentstretch;
  if (params_->massprodfunc_ == "Lin")
  {
    if (localhomstress_->at(gp)(0) != 0.0)
      (*massprodcomp)(0) =
          massprodbasal_ * (1.0 + growthfactor * (massstress1 / localhomstress_->at(gp)(0) - 1.0) +
                               sheargrowthfactor * sheardiff);
    else
      (*massprodcomp)(0) =
          massprodbasal_ * (1.0 + growthfactor * massstress1 + sheargrowthfactor * sheardiff);
    if ((*massprodcomp)(0) > (maxmassprodfac * massprodbasal_))
      (*massprodcomp)(0) = maxmassprodfac * massprodbasal_;
  }
  else if (params_->massprodfunc_ == "CosCos")
  {
    if (localhomstress_->at(gp)(0) != 0.0)
    {
      double facstress = 0.0;
      double deltastress = massstress1 / localhomstress_->at(gp)(0) - 1.0;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(0) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
    else
    {
      double facstress = 0.0;
      double deltastress = massstress1;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(0) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
  }
  if ((*massprodcomp)(0) < 0.0)
  {
    (*massprodcomp)(0) = 0.0;
    // std::cout << "warning negative massproduction rate" << std::endl;
  }

  // Fiber2
  temp_vector.multiply(Cmatrix, a2_->at(gp));
  temp_scalar.multiply_tn(a2_->at(gp), temp_vector);
  currentstretch = sqrt(temp_scalar(0));
  temp_vector.multiply(temp2, a2_->at(gp));
  temp_scalar.multiply_tn(a2_->at(gp), temp_vector);
  double massstress2 = sqrt(temp_scalar(0)) / currentstretch;
  if (params_->massprodfunc_ == "Lin")
  {
    if (localhomstress_->at(gp)(1) != 0.0)
      (*massprodcomp)(1) =
          massprodbasal_ * (1.0 + growthfactor * (massstress2 / localhomstress_->at(gp)(1) - 1.0) +
                               sheargrowthfactor * sheardiff);
    else
      (*massprodcomp)(1) =
          massprodbasal_ * (1.0 + growthfactor * massstress2 + sheargrowthfactor * sheardiff);
    if ((*massprodcomp)(1) > (maxmassprodfac * massprodbasal_))
      (*massprodcomp)(1) = maxmassprodfac * massprodbasal_;
  }
  else if (params_->massprodfunc_ == "CosCos")
  {
    if (localhomstress_->at(gp)(1) != 0.0)
    {
      double facstress = 0.0;
      double deltastress = massstress2 / localhomstress_->at(gp)(1) - 1.0;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(1) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
    else
    {
      double facstress = 0.0;
      double deltastress = massstress2;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(1) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
  }
  if ((*massprodcomp)(1) < 0.0)
  {
    (*massprodcomp)(1) = 0.0;
    // std::cout << "warning negative massproduction rate" << std::endl;
  }

  // Fiber3
  temp_vector.multiply(Cmatrix, a3_->at(gp));
  temp_scalar.multiply_tn(a3_->at(gp), temp_vector);
  currentstretch = sqrt(temp_scalar(0));
  temp_vector.multiply(temp2, a3_->at(gp));
  temp_scalar.multiply_tn(a3_->at(gp), temp_vector);
  double massstress3 = sqrt(temp_scalar(0)) / currentstretch;
  if (params_->massprodfunc_ == "Lin")
  {
    if (localhomstress_->at(gp)(2) != 0.0)
      (*massprodcomp)(2) =
          massprodbasal_ * (1.0 + growthfactor * (massstress3 / localhomstress_->at(gp)(2) - 1.0) +
                               sheargrowthfactor * sheardiff);  // /4.0 massstress3
    else
      (*massprodcomp)(2) =
          massprodbasal_ * (1.0 + growthfactor * massstress3 + sheargrowthfactor * sheardiff);
    if ((*massprodcomp)(2) > (maxmassprodfac * massprodbasal_))
      (*massprodcomp)(2) = maxmassprodfac * massprodbasal_;
  }
  else if (params_->massprodfunc_ == "CosCos")
  {
    if (localhomstress_->at(gp)(2) != 0.0)
    {
      double facstress = 0.0;
      double deltastress = massstress3 / localhomstress_->at(gp)(2) - 1.0;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(2) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
    else
    {
      double facstress = 0.0;
      double deltastress = massstress3;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(2) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
  }
  //  (*massprodcomp)(2) = 4. * (*massprodcomp)(2);
  if ((*massprodcomp)(2) < 0.0)
  {
    (*massprodcomp)(2) = 0.0;
    // std::cout << "warning negative massproduction rate" << std::endl;
  }

  // Fiber4
  temp_vector.multiply(Cmatrix, a4_->at(gp));
  temp_scalar.multiply_tn(a4_->at(gp), temp_vector);
  currentstretch = sqrt(temp_scalar(0));
  temp_vector.multiply(temp2, a4_->at(gp));
  temp_scalar.multiply_tn(a4_->at(gp), temp_vector);
  double massstress4 = sqrt(temp_scalar(0)) / currentstretch;
  if (params_->massprodfunc_ == "Lin")
  {
    if (localhomstress_->at(gp)(3) != 0.0)
      (*massprodcomp)(3) =
          massprodbasal_ * (1.0 + growthfactor * (massstress4 / localhomstress_->at(gp)(3) - 1.0) +
                               sheargrowthfactor * sheardiff);  // /4.0 massstress4
    else
      (*massprodcomp)(3) =
          massprodbasal_ * (1.0 + growthfactor * massstress4 + sheargrowthfactor * sheardiff);
    if ((*massprodcomp)(3) > (maxmassprodfac * massprodbasal_))
      (*massprodcomp)(3) = maxmassprodfac * massprodbasal_;
  }
  else if (params_->massprodfunc_ == "CosCos")
  {
    if (localhomstress_->at(gp)(3) != 0.0)
    {
      double facstress = 0.0;
      double deltastress = massstress4 / localhomstress_->at(gp)(3) - 1.0;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(3) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
    else
    {
      double facstress = 0.0;
      double deltastress = massstress4;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp)(3) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
  }
  //  (*massprodcomp)(3) = 4. * (*massprodcomp)(3);
  if ((*massprodcomp)(3) < 0.0)
  {
    (*massprodcomp)(3) = 0.0;
    // std::cout << "warning negative massproduction rate" << std::endl;
  }

  (*massstress)(0) = massstress1;
  (*massstress)(1) = massstress2;
  (*massstress)(2) = massstress3;
  (*massstress)(3) = massstress4;
}

/*----------------------------------------------------------------------*
 |  mass_production_single_fiber                     (private)        05/11|
 *----------------------------------------------------------------------*
compute new deposition rate for one fiber family
driving force depends on S^k

m^k = mbasal * (1 + K * ( sqrt(a_i^T*C*S^k*C*S^k*C*a_i/detC)/lambda_k(t) /homeo - 1))

therefore we need C and S as matrices
*/
void Mat::ConstraintMixture::mass_production_single_fiber(const int gp,
    Core::LinAlg::Matrix<3, 3> defgrd, Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S, double* massstress,
    double inner_radius, double* massprodcomp, Core::LinAlg::Matrix<3, 1> a, const int idfiber,
    double growthfactor)
{
  Core::LinAlg::Matrix<3, 3> Smatrix(Core::LinAlg::Initialization::zero);
  Smatrix(0, 0) = S(0);
  Smatrix(0, 1) = S(3);
  Smatrix(0, 2) = S(5);
  Smatrix(1, 0) = Smatrix(0, 1);
  Smatrix(1, 1) = S(1);
  Smatrix(1, 2) = S(4);
  Smatrix(2, 0) = Smatrix(0, 2);
  Smatrix(2, 1) = Smatrix(1, 2);
  Smatrix(2, 2) = S(2);
  // one has to use the defgrd here, as with EAS C != F^T F
  Core::LinAlg::Matrix<3, 3> Cmatrix(Core::LinAlg::Initialization::zero);
  Cmatrix.multiply_tn(defgrd, defgrd);
  double detC = Cmatrix.determinant();

  double sheardiff = 0.0;
  double sheargrowthfactor = params_->sheargrowthfactor_;
  if (sheargrowthfactor > 0.0)
    sheardiff =
        1.0 - homradius_ * homradius_ * homradius_ / (inner_radius * inner_radius * inner_radius);
  double maxmassprodfac = params_->maxmassprodfac_;

  Core::LinAlg::Matrix<3, 1> CSCa(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> SCa(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1> temp(Core::LinAlg::Initialization::zero);
  CSCa.multiply(Cmatrix, a);
  SCa.multiply(Smatrix, CSCa);
  CSCa.multiply(Cmatrix, SCa);
  temp.multiply_tn(1.0 / detC, SCa, CSCa);
  (*massstress) = temp(0);
  Core::LinAlg::Matrix<3, 1> temp_vector(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1> temp_scalar(Core::LinAlg::Initialization::zero);
  temp_vector.multiply(Cmatrix, a);
  temp_scalar.multiply_tn(a, temp_vector);
  double currentstretch = sqrt(temp_scalar(0));
  (*massstress) = sqrt(*massstress) / currentstretch;
  double homstress = localhomstress_->at(gp)(idfiber);
  //  if (idfiber == 2 || idfiber == 3)
  //    homstress = 4. * homstress;
  if (params_->massprodfunc_ == "Lin")
  {
    if (homstress != 0.0)
      (*massprodcomp) = massprodbasal_ * (1.0 + growthfactor * ((*massstress) / homstress - 1.0) +
                                             sheargrowthfactor * sheardiff);
    else
      (*massprodcomp) =
          massprodbasal_ * (1.0 + growthfactor * (*massstress) + sheargrowthfactor * sheardiff);
    if (*massprodcomp > (maxmassprodfac * massprodbasal_))
      *massprodcomp = maxmassprodfac * massprodbasal_;
  }
  else if (params_->massprodfunc_ == "CosCos")
  {
    if (homstress != 0.0)
    {
      double facstress = 0.0;
      double deltastress = (*massstress) / homstress - 1.0;
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
    else
    {
      double facstress = 0.0;
      double deltastress = (*massstress);
      mass_function(growthfactor, deltastress, maxmassprodfac, facstress);
      double facshear = 0.0;
      mass_function(sheargrowthfactor, sheardiff, maxmassprodfac, facshear);
      (*massprodcomp) = massprodbasal_ * 0.5 * (facstress + facshear);
    }
  }
  //  if (idfiber == 2 || idfiber == 3)
  //    (*massprodcomp) = 4. * (*massprodcomp);
  if ((*massprodcomp) < 0.0)
  {
    (*massprodcomp) = 0.0;
    // std::cout << "warning negative massproduction rate" << std::endl;
  }
}

/*----------------------------------------------------------------------*
 |  mass_function                                  (private)        06/13|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::mass_function(
    double growthfac, double delta, double mmax, double& massfac)
{
  if (delta < -std::numbers::pi / (2.0 * growthfac))
  {
    massfac = 0.0;
  }
  else if (delta < 0.0)  // && - PI / (2.0 * growthfac) <= delta
  {
    massfac = 0.5 * (1.0 + cos(2.0 * growthfac * delta));
  }
  else if (delta < (mmax - 1.0) * std::numbers::pi / (2.0 * growthfac))  // && 0.0 < delta
  {
    massfac = 0.5 * (mmax + 1.0 - (mmax - 1.0) * cos(2.0 * growthfac * delta / (mmax - 1.0)));
  }
  else  // (mmax -1.0) * PI / (2.0*growthfac) <= delta
  {
    massfac = mmax;
  }
}

/*----------------------------------------------------------------------*
 |  Degradation                                   (private)        10/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::degradation(double t, double& degr)
{
  if (params_->degoption_ == "Lin")  // heaviside step function
  {
    if (t <= params_->lifetime_ + 1.0e-11)
    {
      degr = 1.0;
    }
    else
    {
      degr = 0.0;
    }
  }
  else if (params_->degoption_ == "Exp" || params_->degoption_ == "ExpVar")  // exponential decrease
  {
    degr = exp(-t / params_->lifetime_ * log(2.0));
  }
  else if (params_->degoption_ == "Cos")  // transition zone with cos shape
  {
    if (t < 0.2 * params_->lifetime_ - 1.0e-11)
    {
      degr = 1.0;
    }
    else if (t < params_->lifetime_ - 1.0e-11)
    {
      degr = 0.5 *
             (cos(std::numbers::pi * (t - 0.2 * params_->lifetime_) / (0.8 * params_->lifetime_)) +
                 1.0);
    }
    else
    {
      degr = 0.0;
    }
  }
  else
    FOUR_C_THROW(
        "Degradation option not implemented! Valid options are Lin, Exp, ExpVar and Cos !");
}

/*----------------------------------------------------------------------*
 |  elastin_degradation                            (private)        05/13|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::elastin_degradation(
    Core::LinAlg::Tensor<double, 3> coord, double& elastin_survival) const
{
  if (params_->elastindegrad_ == "Rectangle")
  {
    double funcz = 0.0;
    double z1 = -16.0;
    double z2 = -12.0;
    double z3 = 12.0;
    double z4 = 16.0;
    if (z1 < coord(2) && coord(2) < z2)
    {
      funcz = 0.5 * (1.0 - cos((coord(2) - z1) / (z2 - z1) * std::numbers::pi));
    }
    else if (z3 < coord(2) && coord(2) < z4)
    {
      funcz = 0.5 * (1.0 + cos((coord(2) - z3) / (z4 - z3) * std::numbers::pi));
    }
    else if (z2 <= coord(2) && coord(2) <= z3)
    {
      funcz = 1.0;
    }

    double funcphi = 0.0;
    double phi = atan2(coord(1), coord(0));
    double phi1 = -0.55 * std::numbers::pi;
    double phi2 = -0.5 * std::numbers::pi;
    double phi3 = -0.25 * std::numbers::pi;
    double phi4 = -0.2 * std::numbers::pi;
    if (phi1 < phi && phi < phi2)
    {
      funcphi = 0.5 * (1.0 - cos((phi - phi1) / (phi2 - phi1) * std::numbers::pi));
    }
    else if (phi3 < phi && phi < phi4)
    {
      funcphi = 0.5 * (1.0 + cos((phi - phi3) / (phi4 - phi3) * std::numbers::pi));
    }
    else if (phi2 <= phi && phi <= phi3)
    {
      funcphi = 1.0;
    }

    elastin_survival = 1.0 - funcz * funcphi;
  }
  else if (params_->elastindegrad_ == "RectanglePlate")
  {
    double funcz = 0.0;
    double z1 = -0.5;
    double z2 = 0.0;
    double z3 = 2.0;
    double z4 = 2.5;
    if (z1 < coord(2) && coord(2) < z2)
    {
      funcz = 0.5 * (1.0 - cos((coord(2) - z1) / (z2 - z1) * std::numbers::pi));
    }
    else if (z3 < coord(2) && coord(2) < z4)
    {
      funcz = 0.5 * (1.0 + cos((coord(2) - z3) / (z4 - z3) * std::numbers::pi));
    }
    else if (z2 <= coord(2) && coord(2) <= z3)
    {
      funcz = 1.0;
    }

    double funcx = 0.0;
    double x1 = -2.5;
    double x2 = -2.0;
    double x3 = 2.0;
    double x4 = 2.5;
    if (x1 < coord(0) && coord(0) < x2)
    {
      funcx = 0.5 * (1.0 - cos((coord(0) - x1) / (x2 - x1) * std::numbers::pi));
    }
    else if (x3 < coord(0) && coord(0) < x4)
    {
      funcx = 0.5 * (1.0 + cos((coord(0) - x3) / (x4 - x3) * std::numbers::pi));
    }
    else if (x2 <= coord(0) && coord(0) <= x3)
    {
      funcx = 1.0;
    }

    elastin_survival = 1.0 - funcz * funcx;
  }
  else if (params_->elastindegrad_ == "Wedge")
  {
    double funcz = 0.0;
    double z1 = -16.0;  //-22.0; //-16.0;
    double z2 = -12.0;  //-19.0; //-12.0;
    double z3 = 12.0;   // 19.0; //12.0;
    double z4 = 16.0;   // 22.0; //16.0;
    double phi = atan2(coord(1), coord(0));
    z1 = ((std::numbers::pi - abs(phi)) / std::numbers::pi * 0.75 + 0.25) * z1;
    z2 = ((std::numbers::pi - abs(phi)) / std::numbers::pi * 0.75 + 0.25) * z2;
    z3 = ((std::numbers::pi - abs(phi)) / std::numbers::pi * 0.75 + 0.25) * z3;
    z4 = ((std::numbers::pi - abs(phi)) / std::numbers::pi * 0.75 + 0.25) * z4;
    if (z1 < coord(2) && coord(2) < z2)
    {
      funcz = 0.5 * (1.0 - cos((coord(2) - z1) / (z2 - z1) * std::numbers::pi));
    }
    else if (z3 < coord(2) && coord(2) < z4)
    {
      funcz = 0.5 * (1.0 + cos((coord(2) - z3) / (z4 - z3) * std::numbers::pi));
    }
    else if (z2 <= coord(2) && coord(2) <= z3)
    {
      funcz = 1.0;
    }

    elastin_survival = 1.0 - funcz;
  }
  else if (params_->elastindegrad_ == "Circles")
  {
    double radmin = 10.0;
    double radmax = 15.0;
    double func1 = 0.0;
    Core::LinAlg::Matrix<1, 3> center1(Core::LinAlg::Initialization::zero);
    center1(0) = 12.0;
    center1(1) = 0.0;
    center1(2) = 10.0;
    Core::LinAlg::Matrix<1, 3> diff(coord.data());
    diff.update(-1.0, center1, 1.0);
    double rad1 = diff.norm2();
    if (rad1 < radmin)
    {
      func1 = 1.0;
    }
    else if (rad1 < radmax)
    {
      func1 = 0.5 * (1.0 - cos((rad1 - radmax) / (radmax - radmin) * std::numbers::pi));
    }
    double func2 = 0.0;
    Core::LinAlg::Matrix<1, 3> center2(Core::LinAlg::Initialization::zero);
    center2(0) = -12.0;
    center2(1) = 0.0;
    center2(2) = -10.0;
    diff.update_t(Core::LinAlg::make_matrix_view<3, 1>(coord));
    diff.update(-1.0, center2, 1.0);
    double rad2 = diff.norm2();
    if (rad2 < radmin)
    {
      func2 = 1.0;
    }
    else if (rad2 < radmax)
    {
      func2 = 0.5 * (1.0 - cos((rad2 - radmax) / (radmax - radmin) * std::numbers::pi));
    }
    double func = std::max(func1, func2);
    elastin_survival = 1.0 - func;
  }
}

/*----------------------------------------------------------------------*
 |  evaluate_implicit_all                           (private)        12/11|
 *----------------------------------------------------------------------*
 evaluate stress and cmat for implicit integration
 driving force of massproduction is the total stress S
 */
void Mat::ConstraintMixture::evaluate_implicit_all(Core::LinAlg::Matrix<3, 3> defgrd,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain, const int gp,
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, double dt, double time,
    Core::LinAlg::Matrix<4, 1> massprod, Core::LinAlg::Matrix<4, 1> massstress,
    double elastin_survival, double growthfactor)
{
  //--------------------------------------------------------------------------------------
  // build identity tensor I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) Id(i) = 1.0;
  // right Cauchy-Green Tensor  C = 2 * E + I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(*glstrain);
  C.scale(2.0);
  C += Id;

  //--------------------------------------------------------------------------------------
  // store stress and stretch in a matrix to do matrix-matrix multiplications
  Core::LinAlg::Matrix<3, 3> Smatrix(Core::LinAlg::Initialization::zero);
  Smatrix(0, 0) = (*stress)(0);
  Smatrix(0, 1) = (*stress)(3);
  Smatrix(0, 2) = (*stress)(5);
  Smatrix(1, 0) = Smatrix(0, 1);
  Smatrix(1, 1) = (*stress)(1);
  Smatrix(1, 2) = (*stress)(4);
  Smatrix(2, 0) = Smatrix(0, 2);
  Smatrix(2, 1) = Smatrix(1, 2);
  Smatrix(2, 2) = (*stress)(2);
  Core::LinAlg::Matrix<3, 3> Cmatrix(Core::LinAlg::Initialization::zero);
  Cmatrix(0, 0) = C(0);
  Cmatrix(0, 1) = 0.5 * C(3);
  Cmatrix(0, 2) = 0.5 * C(5);
  Cmatrix(1, 0) = Cmatrix(0, 1);
  Cmatrix(1, 1) = C(1);
  Cmatrix(1, 2) = 0.5 * C(4);
  Cmatrix(2, 0) = Cmatrix(0, 2);
  Cmatrix(2, 1) = Cmatrix(1, 2);
  Cmatrix(2, 2) = C(2);

  //--------------------------------------------------------------------------------------
  // some variables
  const int firstiter = 0;
  Core::LinAlg::Matrix<4, 1> prestretchcollagen = localprestretch_->at(gp);
  // store actual collagen stretches, do not change anymore
  Core::LinAlg::Matrix<4, 1> actcollstretch(Core::LinAlg::Initialization::zero);
  history_->back().get_stretches(gp, &actcollstretch);

  // isotropic invariant
  const double I3 = C(0) * C(1) * C(2) + 0.25 * C(3) * C(4) * C(5) - 0.25 * C(1) * C(5) * C(5) -
                    0.25 * C(2) * C(3) * C(3) -
                    0.25 * C(0) * C(4) * C(4);  // 3rd invariant, determinant
  const double J = sqrt(I3);                    // determinant of F

  //-------------------------------------
  // invert C
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv(Core::LinAlg::Initialization::zero);
  Cinv(0) = C(1) * C(2) - 0.25 * C(4) * C(4);
  Cinv(1) = C(0) * C(2) - 0.25 * C(5) * C(5);
  Cinv(2) = C(0) * C(1) - 0.25 * C(3) * C(3);
  Cinv(3) = 0.25 * C(5) * C(4) - 0.5 * C(3) * C(2);
  Cinv(4) = 0.25 * C(3) * C(5) - 0.5 * C(0) * C(4);
  Cinv(5) = 0.25 * C(3) * C(4) - 0.5 * C(5) * C(1);
  Cinv.scale(1.0 / I3);

  history_->back().set_mass(gp, massprod);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressresidual(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmattemp(Core::LinAlg::Initialization::zero);
  evaluate_stress(glstrain, gp, &cmattemp, &stressresidual, firstiter, time, elastin_survival);

  // determine residual
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Residual(*stress);
  Residual.update(-1.0, stressresidual, 1.0);


  //--------------------------------------------------------------------------------------
  // local Newton iteration
  int localistep = 0;
  int maxstep = 50;
  while (Residual.norm2() > params_->abstol_ * (*stress).norm_inf() && localistep < maxstep)
  {
    localistep += 1;
    //--------------------------------------------------------------------------------------
    // derivative of residual
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> DResidual(
        Core::LinAlg::Initialization::zero);
    for (int id = 0; id < NUM_STRESS_3D; id++) DResidual(id, id) = 1.0;

    // for all 4 fiber families
    double stretch = prestretchcollagen(0) / actcollstretch(0);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dstressdmass(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dmassdstress(Core::LinAlg::Initialization::zero);
    grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a1_->at(gp), stretch, J, dt, true);
    grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a1_->at(gp), J, massstress(0),
        localhomstress_->at(gp)(0), actcollstretch(0), growthfactor);
    DResidual.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

    stretch = prestretchcollagen(1) / actcollstretch(1);
    dstressdmass.put_scalar(0.0);
    dmassdstress.put_scalar(0.0);
    grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a2_->at(gp), stretch, J, dt, true);
    grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a2_->at(gp), J, massstress(1),
        localhomstress_->at(gp)(1), actcollstretch(1), growthfactor);
    DResidual.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

    stretch = prestretchcollagen(2) / actcollstretch(2);
    dstressdmass.put_scalar(0.0);
    dmassdstress.put_scalar(0.0);
    grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a3_->at(gp), stretch, J, dt, true);
    grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a3_->at(gp), J, massstress(2),
        localhomstress_->at(gp)(2), actcollstretch(2), growthfactor);
    DResidual.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

    stretch = prestretchcollagen(3) / actcollstretch(3);
    dstressdmass.put_scalar(0.0);
    dmassdstress.put_scalar(0.0);
    grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a4_->at(gp), stretch, J, dt, true);
    grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a4_->at(gp), J, massstress(3),
        localhomstress_->at(gp)(3), actcollstretch(3), growthfactor);
    DResidual.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

    //----------------------------------------------------
    // solve linear system of equations: gradF * incr = -F
    //----------------------------------------------------
    // F = F*-1.0
    Residual.scale(-1.0);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> increment(Core::LinAlg::Initialization::zero);
    // solve A.X=B
    Core::LinAlg::FixedSizeSerialDenseSolver<NUM_STRESS_3D, NUM_STRESS_3D, 1> solver;
    solver.set_matrix(DResidual);             // set A=DResidual
    solver.set_vectors(increment, Residual);  // set X=increment, B=Residual
    solver.factor_with_equilibration(true);   // "some easy type of preconditioning" (Michael)
    int err2 = solver.factor();               // ?
    int err = solver.solve();                 // X = A^-1 B
    if ((err != 0) || (err2 != 0))
      FOUR_C_THROW(
          "solving linear system in Newton-Raphson method for implicit integration failed");

    // damping strategy
    double omega = 2.0;
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stepstress(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Residualtemp(Residual);
    double omegamin = 1.0 / 64.0;
    while (Residualtemp.norm2() > (1.0 - 0.5 * omega) * Residual.norm2() && omega > omegamin)
    {
      // update of stress and mass
      omega = omega / 2.0;
      stepstress.update(1.0, *stress, omega, increment);

      mass_production(gp, defgrd, stepstress, &massstress, 0.0, &massprod, growthfactor);
      history_->back().set_mass(gp, massprod);
      stressresidual.put_scalar(0.0);
      evaluate_stress(glstrain, gp, &cmattemp, &stressresidual, firstiter, time, elastin_survival);

      Residualtemp.update(1.0, stepstress, -1.0, stressresidual);
    }
    // if (omega <= omegamin && Residualtemp.Norm2() > (1.0-0.5*omega)*Residual.Norm2())
    //  FOUR_C_THROW("no damping coefficient found");

    *stress = stepstress;
    Residual = Residualtemp;

    Smatrix(0, 0) = (*stress)(0);
    Smatrix(0, 1) = (*stress)(3);
    Smatrix(0, 2) = (*stress)(5);
    Smatrix(1, 0) = Smatrix(0, 1);
    Smatrix(1, 1) = (*stress)(1);
    Smatrix(1, 2) = (*stress)(4);
    Smatrix(2, 0) = Smatrix(0, 2);
    Smatrix(2, 1) = Smatrix(1, 2);
    Smatrix(2, 2) = (*stress)(2);

    if ((massprod(0) < 0.0) || (massprod(1) < 0.0) || (massprod(2) < 0.0) || (massprod(3) < 0.0))
    {
      std::cout << "1: " << massprod(0) << std::endl;
      std::cout << "2: " << massprod(1) << std::endl;
      std::cout << "3: " << massprod(2) << std::endl;
      std::cout << "4: " << massprod(3) << std::endl;
      FOUR_C_THROW("negative mass production computed for at least one collagen fiber family!");
    }

  }  // while loop
  if (localistep == maxstep && Residual.norm2() > params_->abstol_ * (*stress).norm_inf())
    FOUR_C_THROW("local Newton iteration did not converge {}", Residual.norm2());

  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatelastic(
      Core::LinAlg::Initialization::zero);
  evaluate_stress(glstrain, gp, &cmatelastic, stress, firstiter, time, elastin_survival);

  //--------------------------------------------------------------------------------------
  // compute cmat
  // right handside of the linear equations
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> RHS(cmatelastic);
  // left matrix of the linear equations
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> LM(Core::LinAlg::Initialization::zero);
  for (int id = 0; id < NUM_STRESS_3D; id++) LM(id, id) = 1.0;

  // Fiber1
  double stretch = prestretchcollagen(0) / actcollstretch(0);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dmassdstress(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dmassdstretch(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dstressdmass(Core::LinAlg::Initialization::zero);
  grad_mass_d_stretch(&dmassdstretch, defgrd, Smatrix, Cinv, a1_->at(gp), J, massstress(0),
      localhomstress_->at(gp)(0), actcollstretch(0), dt, growthfactor);
  grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a1_->at(gp), J, massstress(0),
      localhomstress_->at(gp)(0), actcollstretch(0), growthfactor);
  grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a1_->at(gp), stretch, J, dt, true);
  RHS.multiply_nt(2.0, dstressdmass, dmassdstretch, 1.0);
  LM.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

  // Fiber2
  stretch = prestretchcollagen(1) / actcollstretch(1);
  grad_mass_d_stretch(&dmassdstretch, defgrd, Smatrix, Cinv, a2_->at(gp), J, massstress(1),
      localhomstress_->at(gp)(1), actcollstretch(1), dt, growthfactor);
  grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a2_->at(gp), J, massstress(1),
      localhomstress_->at(gp)(1), actcollstretch(1), growthfactor);
  grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a2_->at(gp), stretch, J, dt, true);
  RHS.multiply_nt(2.0, dstressdmass, dmassdstretch, 1.0);
  LM.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

  // Fiber3
  stretch = prestretchcollagen(2) / actcollstretch(2);
  grad_mass_d_stretch(&dmassdstretch, defgrd, Smatrix, Cinv, a3_->at(gp), J, massstress(2),
      localhomstress_->at(gp)(2), actcollstretch(2), dt, growthfactor);
  grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a3_->at(gp), J, massstress(2),
      localhomstress_->at(gp)(2), actcollstretch(2), growthfactor);
  grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a3_->at(gp), stretch, J, dt, true);
  RHS.multiply_nt(2.0, dstressdmass, dmassdstretch, 1.0);
  LM.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

  // Fiber4
  stretch = prestretchcollagen(3) / actcollstretch(3);
  grad_mass_d_stretch(&dmassdstretch, defgrd, Smatrix, Cinv, a4_->at(gp), J, massstress(3),
      localhomstress_->at(gp)(3), actcollstretch(3), dt, growthfactor);
  grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a4_->at(gp), J, massstress(3),
      localhomstress_->at(gp)(3), actcollstretch(3), growthfactor);
  grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a4_->at(gp), stretch, J, dt, true);
  RHS.multiply_nt(2.0, dstressdmass, dmassdstretch, 1.0);
  LM.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

  (*cmat).put_scalar(0.0);
  //----------------------------------------------------
  // solve linear system of equations: A.X=B
  //----------------------------------------------------
  Core::LinAlg::FixedSizeSerialDenseSolver<NUM_STRESS_3D, NUM_STRESS_3D, NUM_STRESS_3D> solver;
  solver.set_matrix(LM);                   // set A=LM
  solver.set_vectors(*cmat, RHS);          // set X=increment, B=RHS
  solver.factor_with_equilibration(true);  // "some easy type of preconditioning" (Michael)
  int err2 = solver.factor();              // ?
  int err = solver.solve();                // X = A^-1 B
  if ((err != 0) || (err2 != 0)) FOUR_C_THROW("solving linear system for cmat failed");

  vismassstress_->at(gp)(0) = massstress(0);
  vismassstress_->at(gp)(1) = massstress(1);
  vismassstress_->at(gp)(2) = massstress(2);
}

/*----------------------------------------------------------------------*
 |  evaluate_implicit_single                        (private)        11/11|
 *----------------------------------------------------------------------*
evaluate stress and cmat for implicit integration
driving force of massproduction is the fiber stress S^k
Newton loop only for stresses
*/
void Mat::ConstraintMixture::evaluate_implicit_single(Core::LinAlg::Matrix<3, 3> defgrd,
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain, const int gp,
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress, double dt, double time, double elastin_survival,
    double growthfactor)
{
  //--------------------------------------------------------------------------------------
  // build identity tensor I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) Id(i) = 1.0;
  // right Cauchy-Green Tensor  C = 2 * E + I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(*glstrain);
  C.scale(2.0);
  C += Id;

  //--------------------------------------------------------------------------------------
  // store stretch in a matrix to do matrix-matrix multiplications
  Core::LinAlg::Matrix<3, 3> Cmatrix(Core::LinAlg::Initialization::zero);
  Cmatrix(0, 0) = C(0);
  Cmatrix(0, 1) = 0.5 * C(3);
  Cmatrix(0, 2) = 0.5 * C(5);
  Cmatrix(1, 0) = Cmatrix(0, 1);
  Cmatrix(1, 1) = C(1);
  Cmatrix(1, 2) = 0.5 * C(4);
  Cmatrix(2, 0) = Cmatrix(0, 2);
  Cmatrix(2, 1) = Cmatrix(1, 2);
  Cmatrix(2, 2) = C(2);

  // isotropic invariant
  const double I3 = C(0) * C(1) * C(2) + 0.25 * C(3) * C(4) * C(5) - 0.25 * C(1) * C(5) * C(5) -
                    0.25 * C(2) * C(3) * C(3) -
                    0.25 * C(0) * C(4) * C(4);  // 3rd invariant, determinant
  const double J = sqrt(I3);                    // determinant of F

  //-------------------------------------
  // invert C
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv(Core::LinAlg::Initialization::zero);
  Cinv(0) = C(1) * C(2) - 0.25 * C(4) * C(4);
  Cinv(1) = C(0) * C(2) - 0.25 * C(5) * C(5);
  Cinv(2) = C(0) * C(1) - 0.25 * C(3) * C(3);
  Cinv(3) = 0.25 * C(5) * C(4) - 0.5 * C(3) * C(2);
  Cinv(4) = 0.25 * C(3) * C(5) - 0.5 * C(0) * C(4);
  Cinv(5) = 0.25 * C(3) * C(4) - 0.5 * C(5) * C(1);
  Cinv.scale(1.0 / I3);

  //--------------------------------------------------------------------------------------
  // some variables
  const int firstiter = 0;
  double currmassdens = 0.0;
  double qdegrad = 0.0;
  degradation(0.0, qdegrad);
  double density = params_->density_;
  // store actual collagen stretches, do not change anymore
  Core::LinAlg::Matrix<4, 1> actcollstretch(Core::LinAlg::Initialization::zero);
  history_->back().get_stretches(gp, &actcollstretch);
  Core::LinAlg::Matrix<4, 1> massprod(Core::LinAlg::Initialization::zero);
  history_->back().get_mass(gp, &massprod);
  Core::LinAlg::Matrix<4, 1> massstress(Core::LinAlg::Initialization::zero);

  // set stress and cmat to zero, as they are not zero here
  (*stress).put_scalar(0.0);
  (*cmat).put_scalar(0.0);

  // everything related to the fiber families
  for (int idfiber = 0; idfiber < 4; idfiber++)
  {
    Core::LinAlg::Matrix<3, 1> a(Core::LinAlg::Initialization::zero);
    if (idfiber == 0)
      a = a1_->at(gp);
    else if (idfiber == 1)
      a = a2_->at(gp);
    else if (idfiber == 2)
      a = a3_->at(gp);
    else
      a = a4_->at(gp);

    double homstress = localhomstress_->at(gp)(idfiber);
    double prestretchcollagen = localprestretch_->at(gp)(idfiber);
    double stretch = prestretchcollagen / actcollstretch(idfiber);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressfiber(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressresidual(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatfiber(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmattemp(Core::LinAlg::Initialization::zero);
    double currmassdensfiber = 0.0;
    double currmassdenstemp = 0.0;
    double massprodfiber = 0.0;
    double massstressfiber = 0.0;
    evaluate_fiber_family(
        C, gp, &cmatfiber, &stressfiber, a, &currmassdensfiber, firstiter, time, idfiber);
    // mass always corresponds to the current stress
    mass_production_single_fiber(
        gp, defgrd, stressfiber, &massstressfiber, 0.0, &massprodfiber, a, idfiber, growthfactor);
    massprod(idfiber) = massprodfiber;
    history_->back().set_mass(gp, massprod);
    massstress(idfiber) = massstressfiber;
    // compute stresses for the computed mass
    evaluate_fiber_family(
        C, gp, &cmattemp, &stressresidual, a, &currmassdenstemp, firstiter, time, idfiber);

    Core::LinAlg::Matrix<3, 3> Smatrix(Core::LinAlg::Initialization::zero);
    Smatrix(0, 0) = stressfiber(0);
    Smatrix(0, 1) = stressfiber(3);
    Smatrix(0, 2) = stressfiber(5);
    Smatrix(1, 0) = Smatrix(0, 1);
    Smatrix(1, 1) = stressfiber(1);
    Smatrix(1, 2) = stressfiber(4);
    Smatrix(2, 0) = Smatrix(0, 2);
    Smatrix(2, 1) = Smatrix(1, 2);
    Smatrix(2, 2) = stressfiber(2);

    // determine residual
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Residual(stressfiber);
    Residual.update(-1.0, stressresidual, 1.0);

    //--------------------------------------------------------------------------------------
    // local Newton iteration to determine stress
    int localistep = 0;
    int maxstep = 50;
    while (Residual.norm2() > params_->abstol_ * stressfiber.norm_inf() && localistep < maxstep)
    {
      localistep += 1;
      //--------------------------------------------------------------------------------------
      // derivative of residual
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> DResidual(
          Core::LinAlg::Initialization::zero);
      for (int id = 0; id < NUM_STRESS_3D; id++) DResidual(id, id) = 1.0;

      // linearisation of stress formula
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dstressdmass(Core::LinAlg::Initialization::zero);
      grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a, stretch, J, dt, false);
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dmassdstress(Core::LinAlg::Initialization::zero);
      grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a, J, massstress(idfiber), homstress,
          actcollstretch(idfiber), growthfactor);
      DResidual.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

      //----------------------------------------------------
      // solve linear system of equations: gradF * incr = -F
      //----------------------------------------------------
      // F = F*-1.0
      Residual.scale(-1.0);
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> increment(Core::LinAlg::Initialization::zero);
      // solve A.X=B
      Core::LinAlg::FixedSizeSerialDenseSolver<NUM_STRESS_3D, NUM_STRESS_3D, 1> solver;
      solver.set_matrix(DResidual);             // set A=DResidual
      solver.set_vectors(increment, Residual);  // set X=increment, B=Residual
      solver.factor_with_equilibration(true);   // "some easy type of preconditioning" (Michael)
      int err2 = solver.factor();               // ?
      int err = solver.solve();                 // X = A^-1 B
      if ((err != 0) || (err2 != 0))
        FOUR_C_THROW(
            "solving linear system in Newton-Raphson method for implicit integration failed");

      // damping strategy
      double omega = 2.0;
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stepstress(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Residualtemp(Residual);
      double omegamin = 1.0 / 64.0;
      while (Residualtemp.norm2() > (1.0 - 0.5 * omega) * Residual.norm2() && omega > omegamin)
      {
        // update of stress
        omega = omega / 2.0;
        stepstress.update(1.0, stressfiber, omega, increment);

        // corresponding mass
        mass_production_single_fiber(gp, defgrd, stepstress, &massstressfiber, 0.0, &massprodfiber,
            a, idfiber, growthfactor);
        massprod(idfiber) = massprodfiber;
        history_->back().set_mass(gp, massprod);
        massstress(idfiber) = massstressfiber;
        // compute stresses for the computed mass
        stressresidual.put_scalar(0.0);
        evaluate_fiber_family(
            C, gp, &cmattemp, &stressresidual, a, &currmassdenstemp, firstiter, time, idfiber);

        Residualtemp.update(1.0, stepstress, -1.0, stressresidual);
      }
      if (omega <= omegamin && Residualtemp.norm2() > (1.0 - 0.5 * omega) * Residual.norm2())
        FOUR_C_THROW("no damping coefficient found");

      stressfiber = stepstress;
      Residual = Residualtemp;

      Smatrix(0, 0) = stressfiber(0);
      Smatrix(0, 1) = stressfiber(3);
      Smatrix(0, 2) = stressfiber(5);
      Smatrix(1, 0) = Smatrix(0, 1);
      Smatrix(1, 1) = stressfiber(1);
      Smatrix(1, 2) = stressfiber(4);
      Smatrix(2, 0) = Smatrix(0, 2);
      Smatrix(2, 1) = Smatrix(1, 2);
      Smatrix(2, 2) = stressfiber(2);

      if ((massprod(idfiber) < 0.0))
      {
        std::cout << idfiber + 1 << ": " << massprod(idfiber) << std::endl;
        FOUR_C_THROW("negative mass production computed for one collagen fiber family!");
      }

    }  // while loop
    if (localistep == maxstep && Residual.norm2() > params_->abstol_ * stressfiber.norm_inf())
      FOUR_C_THROW("local Newton iteration did not converge {}", Residual.norm2());

    currmassdensfiber = 0.0;
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatelastic(
        Core::LinAlg::Initialization::zero);
    stressfiber.put_scalar(0.0);
    evaluate_fiber_family(
        C, gp, &cmatelastic, &stressfiber, a, &currmassdensfiber, firstiter, time, idfiber);

    //--------------------------------------------------------------------------------------
    // compute cmat
    // right handside of the linear equations
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> RHS(cmatelastic);
    // left matrix of the linear equations
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> LM(Core::LinAlg::Initialization::zero);
    for (int id = 0; id < NUM_STRESS_3D; id++) LM(id, id) = 1.0;

    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dmassdstress(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dmassdstretch(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> dstressdmass(Core::LinAlg::Initialization::zero);
    grad_mass_d_stretch(&dmassdstretch, defgrd, Smatrix, Cinv, a, J, massstress(idfiber), homstress,
        actcollstretch(idfiber), dt, growthfactor);
    grad_mass_d_stress(&dmassdstress, defgrd, Smatrix, a, J, massstress(idfiber), homstress,
        actcollstretch(idfiber), growthfactor);
    grad_stress_d_mass(glstrain, &dstressdmass, Cinv, a, stretch, J, dt, false);
    RHS.multiply_nt(2.0, dstressdmass, dmassdstretch, 1.0);
    LM.multiply_nt(-1.0, dstressdmass, dmassdstress, 1.0);

    cmatfiber.put_scalar(0.0);
    //----------------------------------------------------
    // solve linear system of equations: A.X=B
    //----------------------------------------------------
    Core::LinAlg::FixedSizeSerialDenseSolver<NUM_STRESS_3D, NUM_STRESS_3D, NUM_STRESS_3D> solver;
    solver.set_matrix(LM);                   // set A=LM
    solver.set_vectors(cmatfiber, RHS);      // set X=increment, B=RHS
    solver.factor_with_equilibration(true);  // "some easy type of preconditioning" (Michael)
    int err2 = solver.factor();              // ?
    int err = solver.solve();                // X = A^-1 B
    if ((err != 0) || (err2 != 0)) FOUR_C_THROW("solving linear system for cmat failed");

    (*stress) += stressfiber;
    (*cmat) += cmatfiber;
    currmassdens += currmassdensfiber;

    // volumetric part, that is related to this fiber family
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> tempvol(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatvolfiber(
        Core::LinAlg::Initialization::zero);
    tempvol.multiply_tn(cmatfiber, dmassdstress);
    tempvol.update(2.0, dmassdstretch, 1.0);
    cmatvolfiber.multiply_nt(-1.0 * dt / density * qdegrad * params_->kappa_ * J, Cinv, tempvol);
    //(*cmat).multiply_nt(-2.0*dt/density*qdegrad*params_->kappa_*J,Cinv,dmassdstretch,1.0);
    (*cmat) += cmatvolfiber;
  }

  // elastin
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Siso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatiso(Core::LinAlg::Initialization::zero);
  evaluate_elastin(C, &cmatiso, &Siso, time, &currmassdens, elastin_survival);
  (*stress) += Siso;
  (*cmat) += cmatiso;

  // smooth muscle cells
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Smus(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatmus(Core::LinAlg::Initialization::zero);
  evaluate_muscle(C, &cmatmus, &Smus, gp, &currmassdens);
  (*stress) += Smus;
  (*cmat) += cmatmus;

  // volumetric part
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Svol(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatvol(Core::LinAlg::Initialization::zero);
  evaluate_volumetric(C, &cmatvol, &Svol, currmassdens, density);
  (*stress) += Svol;
  (*cmat) += cmatvol;

  vismassstress_->at(gp)(0) = massstress(0);
  vismassstress_->at(gp)(1) = massstress(1);
  vismassstress_->at(gp)(2) = massstress(2);
  refmassdens_->at(gp) = currmassdens;
}

/*----------------------------------------------------------------------*
 |  grad_stress_d_mass                               (private)        05/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::grad_stress_d_mass(
    const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* derivative, Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv,
    Core::LinAlg::Matrix<3, 1> a, double stretch, double J, double dt, bool option)
{
  double density = params_->density_;
  double qdegrad = 0.0;
  degradation(0.0, qdegrad);

  //--------------------------------------------------------------------------------------
  // build identity tensor I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) Id(i) = 1.0;
  // right Cauchy-Green Tensor  C = 2 * E + I
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C(*glstrain);
  C.scale(2.0);
  C += Id;

  //--------------------------------------------------------------------------------------
  // structural tensors in voigt notation
  // A = a x a
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> A;
  for (int i = 0; i < 3; i++) A(i) = a(i) * a(i);

  A(3) = a(0) * a(1);
  A(4) = a(1) * a(2);
  A(5) = a(0) * a(2);

  double I4 = A(0) * C(0) + A(1) * C(1) + A(2) * C(2) +
              1. * (A(3) * C(3) + A(4) * C(4) + A(5) * C(5));  // I4 = trace(A C)

  double facS = 0.0;
  double temp = 0.0;
  Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Saniso(A);
  I4 = I4 * stretch * stretch;  // account for prestretch and stretch at deposition time
  evaluate_single_fiber_scalars(I4, temp, facS);
  facS = facS * stretch * stretch * qdegrad / density * dt;
  Saniso.scale(facS);
  if (option)
    (*derivative).update(-dt / density * qdegrad * params_->kappa_ * J, Cinv, 1.0, Saniso);
  else
    *derivative = Saniso;
}

/*----------------------------------------------------------------------*
 |  grad_mass_d_stress                               (private)        05/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::grad_mass_d_stress(Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* derivative,
    Core::LinAlg::Matrix<3, 3> defgrd, Core::LinAlg::Matrix<3, 3> Smatrix,
    Core::LinAlg::Matrix<3, 1> a, double J, double massstress, double homstress,
    double actcollstretch, double growthfactor)
{
  // include the case homstress = 0.0!!
  double homstressfixed = 1.0;
  if (homstress != 0.0) homstressfixed = homstress;

  Core::LinAlg::Matrix<3, 3> Cmatrix(Core::LinAlg::Initialization::zero);
  Cmatrix.multiply_tn(defgrd, defgrd);

  Core::LinAlg::Matrix<3, 1> Ca(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> CSCa(Core::LinAlg::Initialization::zero);
  Ca.multiply(Cmatrix, a);
  Core::LinAlg::Matrix<3, 1> temp1(Core::LinAlg::Initialization::zero);
  temp1.multiply(Smatrix, Ca);
  CSCa.multiply(Cmatrix, temp1);
  double fac = massprodbasal_ * growthfactor / homstressfixed / (J * J) / massstress /
               (actcollstretch * actcollstretch);
  if (massstress == 0.0) fac = 0.0;
  (*derivative)(0) = fac * Ca(0) * CSCa(0);
  (*derivative)(1) = fac * Ca(1) * CSCa(1);
  (*derivative)(2) = fac * Ca(2) * CSCa(2);
  (*derivative)(3) = fac * 0.5 * (Ca(0) * CSCa(1) + Ca(1) * CSCa(0));
  (*derivative)(4) = fac * 0.5 * (Ca(1) * CSCa(2) + Ca(2) * CSCa(1));
  (*derivative)(5) = fac * 0.5 * (Ca(0) * CSCa(2) + Ca(2) * CSCa(0));
}

/*----------------------------------------------------------------------*
 |  grad_mass_d_stretch                              (private)        05/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::grad_mass_d_stretch(Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* derivative,
    Core::LinAlg::Matrix<3, 3> defgrd, Core::LinAlg::Matrix<3, 3> Smatrix,
    Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv, Core::LinAlg::Matrix<3, 1> a, double J,
    double massstress, double homstress, double actcollstretch, double dt, double growthfactor)
{
  // include the case homstress = 0.0!!
  double homstressfixed = 1.0;
  if (homstress != 0.0) homstressfixed = homstress;

  Core::LinAlg::Matrix<3, 3> Cmatrix(Core::LinAlg::Initialization::zero);
  Cmatrix.multiply_tn(defgrd, defgrd);

  Core::LinAlg::Matrix<3, 1> SCa(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> SCSCa(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> temp(Core::LinAlg::Initialization::zero);
  temp.multiply(Cmatrix, a);
  SCa.multiply(Smatrix, temp);
  temp.put_scalar(0.0);
  temp.multiply(Cmatrix, SCa);
  SCSCa.multiply(Smatrix, temp);

  (*derivative)(0) = a(0) * a(0);
  (*derivative)(1) = a(1) * a(1);
  (*derivative)(2) = a(2) * a(2);
  (*derivative)(3) = a(0) * a(1);
  (*derivative)(4) = a(1) * a(2);
  (*derivative)(5) = a(0) * a(2);
  (*derivative).update(1.0, Cinv, 1.0 / (actcollstretch * actcollstretch));
  (*derivative).scale(-1.0 * massstress);
  double fac = 1.0 / massstress / (actcollstretch * actcollstretch) / (J * J);
  (*derivative)(0) += fac * (2.0 * a(0) * SCSCa(0) + SCa(0) * SCa(0));
  (*derivative)(1) += fac * (2.0 * a(1) * SCSCa(1) + SCa(1) * SCa(1));
  (*derivative)(2) += fac * (2.0 * a(2) * SCSCa(2) + SCa(2) * SCa(2));
  (*derivative)(3) += fac * (a(0) * SCSCa(1) + a(1) * SCSCa(0) + SCa(0) * SCa(1));
  (*derivative)(4) += fac * (a(1) * SCSCa(2) + a(2) * SCSCa(1) + SCa(1) * SCa(2));
  (*derivative)(5) += fac * (a(0) * SCSCa(2) + a(2) * SCSCa(0) + SCa(0) * SCa(2));
  (*derivative).scale(0.5 * massprodbasal_ * growthfactor / homstressfixed);
}

/*----------------------------------------------------------------------*
 |  EvaluateFiberVecs                             (public)         02/11|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::evaluate_fiber_vecs(const int gp,
    const Core::LinAlg::Matrix<3, 3>& locsys, const Core::LinAlg::Matrix<3, 3>& defgrd)
{
  // locsys holds the principal directions
  // The deformation gradient (defgrd) is needed in remodeling as then locsys is given in the
  // spatial configuration and thus the fiber vectors have to be pulled back in the reference
  // configuration as the material is evaluated there.
  // If this function is called during Setup defgrd should be replaced by the Identity.

  const double gamma = (45 * std::numbers::pi) / 180.;  // angle for diagonal fibers
  Core::LinAlg::Matrix<3, 1> ca1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ca2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ca3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ca4(Core::LinAlg::Initialization::zero);

  for (int i = 0; i < 3; i++)
  {
    // a1 = e3, circumferential direction, used for collagen and smooth muscle
    ca1(i) = locsys(i, 2);
    // a2 = e2
    ca2(i) = locsys(i, 1);
    // a3 = cos gamma e3 + sin gamma e2
    ca3(i) = cos(gamma) * locsys(i, 2) + sin(gamma) * locsys(i, 1);
    // a4 = cos gamma e3 - sin gamma e2
    ca4(i) = cos(gamma) * locsys(i, 2) - sin(gamma) * locsys(i, 1);
  }

  // pull back in reference configuration
  Core::LinAlg::Matrix<3, 1> a1_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> a2_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> a3_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> a4_0(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> idefgrd(Core::LinAlg::Initialization::uninitialized);
  idefgrd.invert(defgrd);
  a1_0.multiply(idefgrd, ca1);
  a2_0.multiply(idefgrd, ca2);
  a3_0.multiply(idefgrd, ca3);
  a4_0.multiply(idefgrd, ca4);

  // normalize vectors
  double a1_0norm = a1_0.norm2();
  a1_->at(gp).update(1.0 / a1_0norm, a1_0);
  double a2_0norm = a2_0.norm2();
  a2_->at(gp).update(1.0 / a2_0norm, a2_0);
  double a3_0norm = a3_0.norm2();
  a3_->at(gp).update(1.0 / a3_0norm, a3_0);
  double a4_0norm = a4_0.norm2();
  a4_->at(gp).update(1.0 / a4_0norm, a4_0);

  return;
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data            (public)         03/13|
 *----------------------------------------------------------------------*/
void Mat::ConstraintMixture::vis_names(std::map<std::string, int>& names) const
{
  std::string fiber = "MassStress";
  names[fiber] = 3;
  fiber = "Fiber1";
  names[fiber] = 3;  // 3-dim vector
  fiber = "Fiber2";
  names[fiber] = 3;  // 3-dim vector
  fiber = "referentialMassDensity";
  names[fiber] = 1;
  fiber = "CollagenMassDensity";
  names[fiber] = 3;
  fiber = "Prestretch";
  names[fiber] = 3;
  fiber = "Homstress";
  names[fiber] = 3;
  fiber = "MassProd";
  names[fiber] = 3;
  fiber = "growthfactor";
  names[fiber] = 1;
  fiber = "elastin_survival";
  names[fiber] = 1;
}

/*----------------------------------------------------------------------*
 |  Return visualization data                     (public)         03/13|
 *----------------------------------------------------------------------*/
bool Mat::ConstraintMixture::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "MassStress")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<3, 1> temp(Core::LinAlg::Initialization::zero);
    for (int iter = 0; iter < numgp; iter++) temp.update(1.0, vismassstress_->at(iter), 1.0);
    data[0] = temp(0) / numgp;
    data[1] = temp(1) / numgp;
    data[2] = temp(2) / numgp;
  }
  else if (name == "Fiber1")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<3, 1> a1 = a1_->at(0);  // get a1 of first gp
    data[0] = a1(0);
    data[1] = a1(1);
    data[2] = a1(2);
  }
  else if (name == "Fiber2")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<3, 1> a2 = a2_->at(0);  // get a2 of first gp
    data[0] = a2(0);
    data[1] = a2(1);
    data[2] = a2(2);
  }
  else if (name == "referentialMassDensity")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int iter = 0; iter < numgp; iter++) temp += refmassdens_->at(iter);
    data[0] = temp / numgp;
  }
  else if (name == "CollagenMassDensity")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<3, 1> temp(Core::LinAlg::Initialization::zero);
    for (int iter = 0; iter < numgp; iter++) temp.update(1.0, visrefmassdens_->at(iter), 1.0);
    data[0] = temp(0) / numgp;
    data[1] = temp(1) / numgp;
    data[2] = temp(2) / numgp;
  }
  else if (name == "Prestretch")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<3, 1> temp(Core::LinAlg::Initialization::zero);
    for (int iter = 0; iter < numgp; iter++) temp.update(1.0, get_prestretch(iter), 1.0);
    data[0] = temp(0) / numgp;
    data[1] = temp(1) / numgp;
    data[2] = temp(2) / numgp;
  }
  else if (name == "Homstress")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<3, 1> temp(Core::LinAlg::Initialization::zero);
    for (int iter = 0; iter < numgp; iter++) temp.update(1.0, get_homstress(iter), 1.0);
    data[0] = temp(0) / numgp;
    data[1] = temp(1) / numgp;
    data[2] = temp(2) / numgp;
  }
  else if (name == "MassProd")
  {
    if ((int)data.size() != 3) FOUR_C_THROW("size mismatch");
    Core::LinAlg::Matrix<4, 1> temp(Core::LinAlg::Initialization::zero);
    int sizehistory = history_->size();
    for (int iter = 0; iter < numgp; iter++)
    {
      Core::LinAlg::Matrix<4, 1> temp_loc(Core::LinAlg::Initialization::zero);
      history_->at(sizehistory - 2).get_mass(iter, &temp_loc);
      // history_->at(0).get_mass(iter,&temp_loc);
      // history_->at(0).get_stretches(iter,&temp_loc);
      temp.update(1.0, temp_loc, 1.0);
    }
    data[0] = temp(0) / numgp;
    data[1] = temp(1) / numgp;
    data[2] = temp(2) / numgp;
  }
  else if (name == "growthfactor")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    // map in GetParameter can now calculate LID, so we do not need it here       05/2017 birzle
    // int eleLID = Global::Problem::instance()->GetDis("structure")->ElementColMap()->LID(eleID);
    data[0] = params_->get_parameter(params_->growthfactor, eleID);
  }
  else if (name == "elastin_survival")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    // map in GetParameter can now calculate LID, so we do not need it here       05/2017 birzle
    // int eleLID = Global::Problem::instance()->GetDis("structure")->ElementColMap()->LID(eleID);
    if (params_->elastindegrad_ == "InvEla")
      data[0] = params_->get_parameter(params_->elastin_survival, eleID);
    else if (params_->elastindegrad_ == "Rectangle" ||
             params_->elastindegrad_ == "RectanglePlate" || params_->elastindegrad_ == "Wedge" ||
             params_->elastindegrad_ == "Circles")
    {
      Core::Elements::Element* myele =
          Global::Problem::instance()->get_dis("structure")->g_element(eleID);
      Core::Nodes::Node** mynodes = myele->nodes();
      for (int idnodes = 0; idnodes < myele->num_node(); idnodes++)
      {
        Core::Nodes::Node* locnode = mynodes[idnodes];
        double elastin_survival = 0.0;
        Core::LinAlg::Tensor<double, 3> point_refe;
        point_refe(0) = locnode->x()[0];
        point_refe(1) = locnode->x()[1];
        point_refe(2) = locnode->x()[2];
        elastin_degradation(point_refe, elastin_survival);
        data[0] += elastin_survival;
      }
      data[0] = data[0] / myele->num_node();
    }
    else
      data[0] = 1.0;
  }
  else
  {
    return false;
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  Debug output to gmsh-file                                      01/11|
 *----------------------------------------------------------------------*
 this needs to be copied to Solid::TimInt::OutputStep() to enable debug output
 {
   discret_->set_state("displacement",Dis());
   Mat::ConstraintMixtureOutputToGmsh(discret_, StepOld(), 1);
 }
 just works with strtimint!
 don't forget to include constraintmixture.H */
void Mat::constraint_mixture_output_to_gmsh(
    Core::FE::Discretization& dis, const int timestep, const int iter)
{
  const std::string filebase = Global::Problem::instance()->output_control_file()->file_name();
  // file for stress
  std::stringstream filename;
  filename << filebase << "_massdensity" << std::setw(3) << std::setfill('0') << timestep
           << std::setw(2) << std::setfill('0') << iter << ".pos";
  std::ofstream f_system(filename.str().c_str());
  std::stringstream gmshfilecontent;
  gmshfilecontent << "View \" Time: " << timestep << " Iter: " << iter << " \" {" << std::endl;
  gmshfilecontent.precision(10);

  // file for prestretch
  std::stringstream filename_pre;
  filename_pre << filebase << "_circollagendens" << std::setw(3) << std::setfill('0') << timestep
               << std::setw(2) << std::setfill('0') << iter << ".pos";
  std::ofstream f_system_pre(filename_pre.str().c_str());
  std::stringstream gmshfilecontent_pre;
  gmshfilecontent_pre << "View \" Time: " << timestep << " Iter: " << iter << " \" {" << std::endl;
  gmshfilecontent_pre.precision(10);


  for (int iele = 0; iele < dis.num_my_col_elements(); ++iele)
  {
    const Core::Elements::Element* actele = dis.l_col_element(iele);

    // build current configuration
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    actele->location_vector(dis, lm, lmowner, lmstride);
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp = dis.get_state("displacement");
    std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

    std::shared_ptr<Core::Mat::Material> mat = actele->material();
    Mat::ConstraintMixture* grow = static_cast<Mat::ConstraintMixture*>(mat.get());

    // material plot at gauss points
    int ngp = grow->geta1()->size();

    // update element geometry
    const int numnode = actele->num_node();
    const int numdof = 3;
    Core::LinAlg::SerialDenseMatrix xcurr(numnode, 3);  // material coord. of element
    for (int i = 0; i < numnode; ++i)
    {
      xcurr(i, 0) = actele->nodes()[i]->x()[0] + mydisp[i * numdof + 0];
      xcurr(i, 1) = actele->nodes()[i]->x()[1] + mydisp[i * numdof + 1];
      xcurr(i, 2) = actele->nodes()[i]->x()[2] + mydisp[i * numdof + 2];
    }
    const Core::FE::CellType distype = actele->shape();
    Core::LinAlg::SerialDenseVector funct(numnode);

    // define gauss rule
    Core::FE::GaussRule3D gaussrule_ = Core::FE::GaussRule3D::undefined;
    switch (distype)
    {
      case Core::FE::CellType::hex8:
      {
        gaussrule_ = Core::FE::GaussRule3D::hex_8point;
        if (ngp != 8) FOUR_C_THROW("hex8 has not 8 gauss points: {}", ngp);
        break;
      }
      case Core::FE::CellType::wedge6:
      {
        gaussrule_ = Core::FE::GaussRule3D::wedge_6point;
        if (ngp != 6) FOUR_C_THROW("wedge6 has not 6 gauss points: {}", ngp);
        break;
      }
      case Core::FE::CellType::tet4:
      {
        gaussrule_ = Core::FE::GaussRule3D::tet_1point;
        if (ngp != 1) FOUR_C_THROW("tet4 has not 1 gauss point: {}", ngp);
        break;
      }
      default:
        FOUR_C_THROW("unknown element in ConstraintMixtureOutputToGmsh");
        break;
    }

    const Core::FE::IntegrationPoints3D intpoints(gaussrule_);

    for (int gp = 0; gp < ngp; ++gp)
    {
      Core::FE::shape_function_3d(
          funct, intpoints.qxg[gp][0], intpoints.qxg[gp][1], intpoints.qxg[gp][2], distype);
      Core::LinAlg::SerialDenseMatrix point(1, 3);
      Core::LinAlg::multiply_tn(point, funct, xcurr);

      // write mandel stress
      // Core::LinAlg::Matrix<3,1> mandelgp = grow->GetHomstress(gp);
      double mandelgp = grow->get_mass_density(gp);
      gmshfilecontent << "SP(" << std::scientific << point(0, 0) << ",";
      gmshfilecontent << std::scientific << point(0, 1) << ",";
      gmshfilecontent << std::scientific << point(0, 2) << ")";
      gmshfilecontent << "{" << std::scientific << mandelgp << "};" << std::endl;

      // write prestretch
      // Core::LinAlg::Matrix<3,1> prestretchgp = grow->GetPrestretch(gp);
      Core::LinAlg::Matrix<3, 1> prestretchgp = grow->get_mass_density_collagen(gp);
      gmshfilecontent_pre << "SP(" << std::scientific << point(0, 0) << ",";
      gmshfilecontent_pre << std::scientific << point(0, 1) << ",";
      gmshfilecontent_pre << std::scientific << point(0, 2) << ")";
      gmshfilecontent_pre << "{" << std::scientific << prestretchgp(0) << "};" << std::endl;
    }
  }

  gmshfilecontent << "};" << std::endl;
  f_system << gmshfilecontent.str();
  f_system.close();

  gmshfilecontent_pre << "};" << std::endl;
  f_system_pre << gmshfilecontent_pre.str();
  f_system_pre.close();

  return;
}

FOUR_C_NAMESPACE_CLOSE
