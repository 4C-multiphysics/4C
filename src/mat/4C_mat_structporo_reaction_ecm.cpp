/*----------------------------------------------------------------------*/
/*! \file
 \brief solid material for  porous model with special implementations
        for ECM model (dissolution of ECM)


\level 3
 *----------------------------------------------------------------------*/



#include "4C_mat_structporo_reaction_ecm.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_poro_law.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::StructPoroReactionECM::StructPoroReactionECM(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : StructPoroReaction(matdata), densCollagen_(matdata.parameters.get<double>("DENSCOLLAGEN"))
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::Mat::Material> Mat::PAR::StructPoroReactionECM::create_material()
{
  return Teuchos::rcp(new Mat::StructPoroReactionECM(this));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::StructPoroReactionECMType Mat::StructPoroReactionECMType::instance_;

Core::Communication::ParObject* Mat::StructPoroReactionECMType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::StructPoroReactionECM* struct_poro = new Mat::StructPoroReactionECM();
  struct_poro->unpack(buffer);
  return struct_poro;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::StructPoroReactionECM::StructPoroReactionECM()
    : refporosity_old_(-1.0),
      refporositydot_old_(0.0),
      chempot_(0),
      chempot_init_(0),
      params_(nullptr)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::StructPoroReactionECM::StructPoroReactionECM(Mat::PAR::StructPoroReactionECM* params)
    : StructPoroReaction(params),
      refporosity_old_(-1.0),
      refporositydot_old_(0.0),
      chempot_(0),
      chempot_init_(0),
      params_(params)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::setup(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  StructPoroReaction::setup(numgp, container);
  refporosity_old_ = params_->init_porosity_;

  double dpsidphiref = 0.0;
  Teuchos::ParameterList params;
  params_->poro_law_->constitutive_derivatives(params, 0.0, 1.0, params_->init_porosity_,
      refporosity_, nullptr, nullptr, nullptr, &dpsidphiref, nullptr);

  const double initphi = params_->init_porosity_;
  const double deltaphi = refporosity_ - initphi;

  chempot_.resize(numgp, 0.0);
  chempot_init_.resize(numgp, 0.0);

  for (std::vector<double>::size_type i = 0; i < chempot_init_.size(); i++)
    chempot_init_[i] = -(1.0 - deltaphi / (1.0 - initphi)) / mat_->density() * dpsidphiref;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // refporosity_
  add_to_pack(data, refporosity_old_);
  // refporositydot_old_
  add_to_pack(data, refporositydot_old_);
  // chempot_init_
  add_to_pack(data, chempot_init_);
  // chempot_
  add_to_pack(data, chempot_);

  // add base class material
  StructPoroReaction::pack(data);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != Teuchos::null)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::StructPoroReactionECM*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }

  extract_from_pack(buffer, refporosity_old_);
  extract_from_pack(buffer, refporositydot_old_);
  extract_from_pack(buffer, chempot_init_);
  extract_from_pack(buffer, chempot_);

  // extract base class material
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer basedata_buffer(basedata);
  StructPoroReaction::unpack(basedata_buffer);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::reaction(const double porosity, const double J,
    Teuchos::RCP<std::vector<double>> scalars, Teuchos::ParameterList& params)
{
  double dt = params.get<double>("delta time", -1.0);
  // double time = params.get<double>("total time",-1.0);

  // concentration C1
  double c1 = scalars->at(params_->dofIDReacScalar_);

  if (dt < 0.0) FOUR_C_THROW("time step not available");

  refporosity_ = 1.0 - J * c1 / params_->densCollagen_;
  refporositydot_ = (refporosity_ - refporosity_old_) / dt;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::update()
{
  refporosity_old_ = refporosity_;
  refporositydot_old_ = refporositydot_;

  StructPoroReaction::update();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::vis_names(std::map<std::string, int>& names) const
{
  // call base class
  StructPoroReaction::vis_names(names);

  std::string name = "chemical_potential";
  names[name] = 1;  // scalar
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Mat::StructPoroReactionECM::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  // call base class
  if (StructPoroReaction::vis_data(name, data, numgp, eleID)) return true;
  if (name == "chemical_potential")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    for (std::vector<double>::size_type i = 0; i < chempot_.size(); i++)
      data[0] += chempot_[i] / chempot_.size();
    return true;
  }
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::StructPoroReactionECM::chem_potential(
    const Core::LinAlg::Matrix<6, 1>& glstrain,  ///< (i) green lagrange strain
    const double porosity,                       ///< (i) porosity
    const double press,                          ///< (i) pressure at gauss point
    const double J,                              ///< (i) determinant of jacobian at gauss point
    int EleID,                                   ///< (i) element GID
    double& pot,                                 ///< (o) chemical potential
    const int gp)
{
  FOUR_C_ASSERT(gp < (int)chempot_.size(),
      "invalid gauss point number for calculation of chemical potential!");
  FOUR_C_ASSERT(gp < (int)chempot_init_.size(),
      "invalid gauss point number for calculation of chemical potential!");

  // dummy parameter list
  Teuchos::ParameterList params;

  double psi = 0.0;
  // evaluate strain energy
  mat_->strain_energy(glstrain, psi, gp, EleID);

  // derivative of
  double dpsidphiref = 0.0;

  params_->poro_law_->constitutive_derivatives(
      params, press, J, porosity, refporosity_, nullptr, nullptr, nullptr, &dpsidphiref, nullptr);

  pot = 1.0 / density() * psi - 1.0 / mat_->density() * dpsidphiref - chempot_init_[gp];
  chempot_[gp] = pot;

  return;
}

FOUR_C_NAMESPACE_CLOSE
