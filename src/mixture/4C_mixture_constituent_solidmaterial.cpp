/*----------------------------------------------------------------------*/
/*! \file

\brief Implementation of the general solid material constituent

\level 3


*/
/*----------------------------------------------------------------------*/

#include "4C_mixture_constituent_solidmaterial.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_mixture.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"



FOUR_C_NAMESPACE_OPEN

// Constructor for the parameter class
MIXTURE::PAR::MixtureConstituentSolidMaterial::MixtureConstituentSolidMaterial(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituent(matdata), matid_(matdata.parameters.get<int>("MATID"))
{
}

// Create an instance of MIXTURE::MixtureConstituentSolidMaterial from the parameters
std::unique_ptr<MIXTURE::MixtureConstituent>
MIXTURE::PAR::MixtureConstituentSolidMaterial::create_constituent(int id)
{
  return std::unique_ptr<MIXTURE::MixtureConstituentSolidMaterial>(
      new MIXTURE::MixtureConstituentSolidMaterial(this, id));
}

// Constructor of the constituent holding the material parameters
MIXTURE::MixtureConstituentSolidMaterial::MixtureConstituentSolidMaterial(
    MIXTURE::PAR::MixtureConstituentSolidMaterial* params, int id)
    : MixtureConstituent(params, id), params_(params), material_()
{
  // take the matid (i.e. here the id of the solid material), read the type and
  // create the corresponding material
  auto mat = Mat::factory(params_->matid_);

  // cast to an So3Material
  material_ = Teuchos::rcp_dynamic_cast<Mat::So3Material>(mat);

  // ensure cast was successfull
  if (Teuchos::is_null(mat))
    FOUR_C_THROW(
        "The solid material constituent with ID %d needs to be an So3Material.", params_->matid_);

  if (material_->density() - 1.0 > 1e-16)
    FOUR_C_THROW(
        "Please set the density of the solid material constituent with ID %d to 1.0 and prescribe "
        "a combined density for the entire mixture material.",
        material_->parameter()->id());
}

Core::Materials::MaterialType MIXTURE::MixtureConstituentSolidMaterial::material_type() const
{
  return Core::Materials::mix_solid_material;
}

void MIXTURE::MixtureConstituentSolidMaterial::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  // pack constituent data
  MixtureConstituent::pack_constituent(data);

  // add the matid of the Mixture_SolidMaterial
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack data of the solid material
  material_->pack(data);
}

void MIXTURE::MixtureConstituentSolidMaterial::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  // unpack constituent data
  MixtureConstituent::unpack_constituent(buffer);

  // make sure we have a pristine material
  params_ = nullptr;
  material_ = Teuchos::null;

  // extract the matid of the Mixture_SolidMaterial
  int matid;
  extract_from_pack(buffer, matid);

  // recover the params_ of the Mixture_SolidMaterial
  if (Global::Problem::instance()->materials() != Teuchos::null)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const unsigned int probinst =
          Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = dynamic_cast<MIXTURE::PAR::MixtureConstituentSolidMaterial*>(mat);
      }
      else
      {
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
      }
    }
  }

  // unpack the data of the solid material
  if (params_ != nullptr)
  {
    auto so3mat = Mat::factory(params_->matid_);
    material_ = Teuchos::rcp_dynamic_cast<Mat::So3Material>(so3mat);
    if (Teuchos::is_null(so3mat)) FOUR_C_THROW("Failed to allocate");

    // solid material packed: 1. the data size, 2. the packed data of size sm
    // extract_from_pack extracts a sub_vec of size sm from data and updates the position vector
    std::vector<char> sub_vec;
    extract_from_pack(buffer, sub_vec);
    Core::Communication::UnpackBuffer sub_vec_buffer(sub_vec);
    material_->unpack(sub_vec_buffer);
  }
}

Core::Materials::MaterialType material_type() { return Core::Materials::mix_solid_material; }

void MIXTURE::MixtureConstituentSolidMaterial::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  MixtureConstituent::read_element(numgp, container);
  material_->setup(numgp, container);
}

void MIXTURE::MixtureConstituentSolidMaterial::update(Core::LinAlg::Matrix<3, 3> const& defgrd,
    Teuchos::ParameterList& params, const int gp, const int eleGID)
{
  material_->update(defgrd, gp, params, eleGID);
}

void MIXTURE::MixtureConstituentSolidMaterial::evaluate(const Core::LinAlg::Matrix<3, 3>& F,
    const Core::LinAlg::Matrix<6, 1>& E_strain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>& S_stress, Core::LinAlg::Matrix<6, 6>& cmat, const int gp,
    const int eleGID)
{
  material_->evaluate(&F, &E_strain, params, &S_stress, &cmat, gp, eleGID);
}

void MIXTURE::MixtureConstituentSolidMaterial::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  material_->register_output_data_names(names_and_size);
}

bool MIXTURE::MixtureConstituentSolidMaterial::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  return material_->evaluate_output_data(name, data);
}
FOUR_C_NAMESPACE_CLOSE
