// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_chemoreac.hpp"
#include "4C_mat_list_chemotaxis.hpp"
#include "4C_mat_list_reactions.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_mat_scatra_chemotaxis.hpp"
#include "4C_mat_scatra_reaction.hpp"
#include "4C_scatra_ele_calc_utils.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::Elements::TransportType Discret::Elements::TransportType::instance_;

Discret::Elements::TransportType& Discret::Elements::TransportType::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::TransportType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Transport* object = new Discret::Elements::Transport(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::TransportType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "TRANSP" or eletype == "CONDIF2" or eletype == "CONDIF3")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Transport>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::TransportType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Transport>(id, owner);
  return ele;
}


void Discret::Elements::TransportType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = dwele->num_dof_per_node(*(dwele->nodes()[0]));
  dimns = numdf;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::TransportType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

void Discret::Elements::TransportType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["TRANSP"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::hex8] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::hex20] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::hex27] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::nurbs27] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::nurbs8] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::tet4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::tet10] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::wedge6] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::wedge15] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::pyramid5] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::quad8] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::nurbs4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::nurbs9] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::line2] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::line3] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::nurbs2] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });

  defs[Core::FE::CellType::nurbs3] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("TYPE"),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
  });
}

/*----------------------------------------------------------------------*
 |  init the element (public)                                           |
 *----------------------------------------------------------------------*/
int Discret::Elements::TransportType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    Discret::Elements::Transport* actele =
        dynamic_cast<Discret::Elements::Transport*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Transport element failed");
    actele->initialize();
  }
  return 0;
}


Discret::Elements::TransportBoundaryType Discret::Elements::TransportBoundaryType::instance_;

Discret::Elements::TransportBoundaryType& Discret::Elements::TransportBoundaryType::instance()
{
  return instance_;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::TransportBoundaryType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new TransportBoundary( id, owner ) );
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                             gjb 05/08 |
 *----------------------------------------------------------------------*/
Discret::Elements::Transport::Transport(int id, int owner)
    : Core::Elements::Element(id, owner),
      distype_(Core::FE::CellType::dis_none),
      name_(),
      vis_map_(),
      numdofpernode_(-1),
      impltype_(Inpar::ScaTra::impltype_undefined)
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        gjb 05/08 |
 *----------------------------------------------------------------------*/
Discret::Elements::Transport::Transport(const Discret::Elements::Transport& old)
    : Core::Elements::Element(old),
      distype_(old.distype_),
      name_(old.name_),
      vis_map_(old.vis_map_),
      numdofpernode_(old.numdofpernode_),
      impltype_(old.impltype_)
{
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Transport and return pointer to it (public) |
 |                                                            gjb 05/08 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Transport::clone() const
{
  Discret::Elements::Transport* newelement = new Discret::Elements::Transport(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |  create material class (public)                            gjb 07/08 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Transport::set_material(
    const int index, std::shared_ptr<Core::Mat::Material> mat)
{
  // the standard part:
  Core::Elements::Element::set_material(index, mat);

  if (mat->material_type() == Core::Materials::m_scatra or
      mat->material_type() == Core::Materials::m_scatra_multiscale or
      mat->material_type() == Core::Materials::m_myocard or
      mat->material_type() == Core::Materials::m_sutherland or
      mat->material_type() == Core::Materials::m_ion or
      mat->material_type() == Core::Materials::m_thermo_fourier or
      mat->material_type() == Core::Materials::m_thermostvenant or
      mat->material_type() == Core::Materials::m_soret or
      mat->material_type() == Core::Materials::m_scatra_in_fluid_porofluid_pressure_based or
      mat->material_type() == Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based or
      mat->material_type() == Core::Materials::m_scatra_in_solid_porofluid_pressure_based or
      mat->material_type() == Core::Materials::m_scatra_as_temperature_porofluid_pressure_based or
      (mat->material_type() == Core::Materials::m_electrode and
          impltype_ == Inpar::ScaTra::impltype_std))
    numdofpernode_ = 1;  // we only have a single scalar
  else if (mat->material_type() == Core::Materials::m_electrode)
    numdofpernode_ = 2;  // concentration and electric potential
  else if (mat->material_type() == Core::Materials::m_matlist)  // we have a system of scalars
  {
    const Mat::MatList* actmat = static_cast<const Mat::MatList*>(mat.get());
    numdofpernode_ = actmat->num_mat();

    // for problem type ELCH we have one additional degree of freedom per node
    // for the electric potential
    if (Global::Problem::instance()->get_problem_type() == Core::ProblemType::elch)
    {
      for (int ii = 0; ii < numdofpernode_; ++ii)
      {
        // In the context of ELCH the only valid material combination is m_matlist and m_ion
        if (actmat->material_by_id(actmat->mat_id(ii))->material_type() != Core::Materials::m_ion)
          FOUR_C_THROW(
              "In the context of ELCH the material Mat_matlist can be only used in combination "
              "with Mat_ion");
      }
      numdofpernode_ += 1;
    }
  }
  else if (mat->material_type() ==
           Core::Materials::m_matlist_reactions)  // we have a system of reactive scalars
  {
    // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are in
    // a diamond inheritance structure
    const Mat::MatListReactions* actmat = dynamic_cast<const Mat::MatListReactions*>(mat.get());
    numdofpernode_ = actmat->num_mat();

    for (int ii = 0; ii < numdofpernode_; ++ii)
    {
      // In the context of reactions the only valid material combination is m_matlist and m_scatra
      if (actmat->material_by_id(actmat->mat_id(ii))->material_type() !=
              Core::Materials::m_scatra and
          actmat->material_by_id(actmat->mat_id(ii))->material_type() !=
              Core::Materials::m_scatra_in_fluid_porofluid_pressure_based and
          actmat->material_by_id(actmat->mat_id(ii))->material_type() !=
              Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based and
          actmat->material_by_id(actmat->mat_id(ii))->material_type() !=
              Core::Materials::m_scatra_as_temperature_porofluid_pressure_based and
          actmat->material_by_id(actmat->mat_id(ii))->material_type() !=
              Core::Materials::m_scatra_in_solid_porofluid_pressure_based)
        FOUR_C_THROW(
            "The material Mat_matlist_reaction only supports MAT_scatra and MAT_scatra_multiporo "
            "as valid main Material");
    }

    int numreac = actmat->num_reac();
    for (int jj = 0; jj < numreac; ++jj)
    {
      // In the context of reactions the only valid material combination is m_matlist and
      // m_scatra_reaction
      if (actmat->material_by_id(actmat->reac_id(jj))->material_type() !=
              Core::Materials::m_scatra_reaction and
          actmat->material_by_id(actmat->reac_id(jj))->material_type() !=
              Core::Materials::m_scatra_reaction_poroECM)
        FOUR_C_THROW(
            "The material MAT_matlist_reaction only supports MAT_scatra_reaction and "
            "MAT_scatra_reaction_poro as valid reaction Material");

      // some safety check for the MAT_scatra_reaction materials
      const std::shared_ptr<const Mat::ScatraReactionMat>& reacmat =
          std::static_pointer_cast<const Mat::ScatraReactionMat>(
              actmat->material_by_id(actmat->reac_id(jj)));
      const int stoichlength = reacmat->num_scal();
      if (stoichlength != numdofpernode_)
        FOUR_C_THROW(
            "The number of scalars in your MAT_scatra_reaction material with ID {} does not fit to "
            "the number of scalars!",
            actmat->reac_id(jj));
    }
  }
  else if (mat->material_type() ==
           Core::Materials::m_matlist_chemotaxis)  // we have a system of chemotactic scalars
  {
    // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are in
    // a diamond inheritance structure
    const Mat::MatListChemotaxis* actmat = dynamic_cast<const Mat::MatListChemotaxis*>(mat.get());
    numdofpernode_ = actmat->num_mat();

    for (int ii = 0; ii < numdofpernode_; ++ii)
    {
      // In the context of chemotaxis the only valid material combination is m_matlist and m_scatra
      if (actmat->material_by_id(actmat->mat_id(ii))->material_type() != Core::Materials::m_scatra)
        FOUR_C_THROW(
            "The material Mat_matlist_chemotaxis only supports MAT_scatra as valid main Material");
    }

    int numpair = actmat->num_pair();
    for (int jj = 0; jj < numpair; ++jj)
    {
      // In the context of chemotaxis the only valid material combination is m_matlist and
      // m_scatra_chemotaxis
      if (actmat->material_by_id(actmat->pair_id(jj))->material_type() !=
          Core::Materials::m_scatra_chemotaxis)
        FOUR_C_THROW(
            "The material MAT_matlist_chemotaxis only supports MAT_scatra_chemotaxis as valid "
            "reaction Material");

      // some safety check for the MAT_scatra_chemotaxis materials
      const std::shared_ptr<const Mat::ScatraChemotaxisMat>& reacmat =
          std::static_pointer_cast<const Mat::ScatraChemotaxisMat>(
              actmat->material_by_id(actmat->pair_id(jj)));
      const int pairlength = reacmat->pair()->size();
      if (pairlength != numdofpernode_)
        FOUR_C_THROW(
            "The number of scalars in your MAT_scatra_chemotaxis material with ID {} does not fit "
            "to the number of scalars!",
            actmat->pair_id(jj));
    }
  }
  else if (mat->material_type() ==
           Core::Materials::m_matlist_chemoreac)  // we have a system of chemotactic scalars
  {
    // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are in
    // a diamond inheritance structure
    const Mat::MatListChemoReac* actmat = dynamic_cast<const Mat::MatListChemoReac*>(mat.get());
    numdofpernode_ = actmat->num_mat();

    for (int ii = 0; ii < numdofpernode_; ++ii)
    {
      // In the context of reactions/chemotaxis the only valid material combination is m_matlist and
      // m_scatra
      if (actmat->material_by_id(actmat->mat_id(ii))->material_type() != Core::Materials::m_scatra)
        FOUR_C_THROW(
            "The material Mat_matlist_chemoreac only supports MAT_scatra as valid main Material");
    }

    int numreac = actmat->num_reac();
    for (int jj = 0; jj < numreac; ++jj)
    {
      // In the context of reactions the only valid material combination is m_matlist and
      // m_scatra_reaction
      if (actmat->material_by_id(actmat->reac_id(jj))->material_type() !=
          Core::Materials::m_scatra_reaction)
        FOUR_C_THROW(
            "The material MAT_matlist_reaction only supports MAT_scatra_reaction as valid reaction "
            "Material");

      // some safety check for the MAT_scatra_reaction materials
      const std::shared_ptr<const Mat::ScatraReactionMat>& reacmat =
          std::static_pointer_cast<const Mat::ScatraReactionMat>(
              actmat->material_by_id(actmat->reac_id(jj)));
      const int stoichlength = reacmat->num_scal();
      if (stoichlength != numdofpernode_)
        FOUR_C_THROW(
            "The number of scalars in your MAT_scatra_reaction material with ID {} does not fit to "
            "the number of scalars!",
            actmat->reac_id(jj));
    }

    int numpair = actmat->num_pair();
    for (int jj = 0; jj < numpair; ++jj)
    {
      // In the context of chemotaxis the only valid material combination is m_matlist and
      // m_scatra_chemotaxis
      if (actmat->material_by_id(actmat->pair_id(jj))->material_type() !=
          Core::Materials::m_scatra_chemotaxis)
        FOUR_C_THROW(
            "The material MAT_matlist_chemotaxis only supports MAT_scatra_chemotaxis as valid "
            "reaction Material");

      // some safety check for the MAT_scatra_chemotaxis materials
      const std::shared_ptr<const Mat::ScatraChemotaxisMat>& reacmat =
          std::static_pointer_cast<const Mat::ScatraChemotaxisMat>(
              actmat->material_by_id(actmat->pair_id(jj)));
      const int pairlength = reacmat->pair()->size();
      if (pairlength != numdofpernode_)
        FOUR_C_THROW(
            "The number of scalars in your MAT_scatra_chemotaxis material with ID {} does not fit "
            "to the number of scalars!",
            actmat->pair_id(jj));
    }
  }
  else if (mat->material_type() == Core::Materials::m_elchmat)
  {
    const Mat::ElchMat* actmat = static_cast<const Mat::ElchMat*>(mat.get());

    numdofpernode_ = actmat->num_dof();
  }
  else
    FOUR_C_THROW("Transport element got unsupported material type {}", mat->material_type());

  return;
}

/*----------------------------------------------------------------------*
 |  create material class (public)                            gjb 07/08 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Transport::set_material(int matnum, Core::Elements::Element* oldele)
{
  set_material(0, Mat::factory(matnum));

  std::shared_ptr<Core::Mat::Material> mat = material();

  if (mat->material_type() == Core::Materials::m_myocard)
  {
    std::shared_ptr<Mat::Myocard> actmat = std::dynamic_pointer_cast<Mat::Myocard>(mat);

    std::shared_ptr<Mat::ElastHyper> somat =
        std::dynamic_pointer_cast<Mat::ElastHyper>(oldele->material());
    if (somat == nullptr) FOUR_C_THROW("cast to ElastHyper failed");

    // copy fiber information from solid material to scatra material (for now, only one fiber
    // vector)
    std::vector<Core::LinAlg::Tensor<double, 3>> fibervecs;
    somat->get_fiber_vecs(fibervecs);
    actmat->setup(fibervecs[0]);
  }
}

/*----------------------------------------------------------------------*
 |  Return the shape of a Transport element                      (public) |
 |                                                            gjb 05/08 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Transport::shape() const { return distype_; }

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            gjb 05/08 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Transport::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  // add internal data
  add_to_pack(data, name_);
  add_to_pack(data, vis_map_);
  add_to_pack(data, numdofpernode_);
  add_to_pack(data, distype_);
  add_to_pack(data, impltype_);
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            gjb 05/08 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Transport::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  // extract internal data
  extract_from_pack(buffer, name_);
  extract_from_pack(buffer, vis_map_);
  extract_from_pack(buffer, numdofpernode_);
  extract_from_pack(buffer, distype_);
  extract_from_pack(buffer, impltype_);
}

/*----------------------------------------------------------------------*
 |  Return number of lines of this element (public)           gjb 07/08 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::num_line() const
{
  return Core::FE::get_number_of_element_lines(distype_);
}


/*----------------------------------------------------------------------*
 |  Return number of surfaces of this element (public)        gjb 07/08 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(distype_);
}


/*----------------------------------------------------------------------*
 | Return number of volumes of this element (public)          gjb 07/08 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(distype_);
}



/*----------------------------------------------------------------------*
 |  print this element (public)                               gjb 05/08 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Transport::print(std::ostream& os) const
{
  os << "Transport element";
  Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(distype_) << std::endl;
  std::cout << std::endl;
  std::cout << "Number DOF per Node: " << numdofpernode_ << std::endl;
  std::cout << std::endl;
  std::cout << "Type of scalar transport: " << ScaTra::impl_type_to_string(impltype_) << std::endl;
  std::cout << std::endl;
}


/*----------------------------------------------------------------------*
 |  get vector of lines            (public)                  g.bau 03/07|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Transport::lines()
{
  return Core::Communication::get_element_lines<TransportBoundary, Transport>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                          g.bau 03/07|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Transport::surfaces()
{
  return Core::Communication::get_element_surfaces<TransportBoundary, Transport>(*this);
}

/*----------------------------------------------------------------------*
 | set implementation type                                   fang 02/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Transport ::set_impl_type(const Inpar::ScaTra::ImplType impltype)
{
  // set implementation type
  impltype_ = impltype;
}

/*----------------------------------------------------------------------*
 |  init the element                                        vuong08/16 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Transport::initialize()
{
  std::shared_ptr<Core::Mat::Material> mat = material();
  // for now, we only need to do something in case of reactions (for the initialization of functions
  // in case of reactions by function)
  if (mat->material_type() == Core::Materials::m_matlist_reactions or
      mat->material_type() == Core::Materials::m_matlist_chemoreac)
  {
    // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction are in
    // a diamond inheritance structure
    std::shared_ptr<Mat::MatListReactions> actmat =
        std::dynamic_pointer_cast<Mat::MatListReactions>(mat);
    actmat->initialize();
  }
  else if (mat->material_type() == Core::Materials::m_myocard)
  {
    std::shared_ptr<Mat::Myocard> actmat = std::dynamic_pointer_cast<Mat::Myocard>(mat);
    int deg = 0;
    if (this->degree() == 1)
      deg = 4 * this->degree();
    else
      deg = 3 * this->degree();
    std::shared_ptr<Core::FE::GaussPoints> quadrature(
        Core::FE::create_gauss_points(this->shape(), deg));
    int gp = quadrature->num_points();
    if (actmat->parameter() != nullptr and
        !actmat->myocard_mat())  // in case we are not in post-process mode
    {
      actmat->set_gp(gp);
      actmat->initialize();
    }
  }

  return 0;
}


//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================


/*----------------------------------------------------------------------*
 |  ctor (public)                                             gjb 01/09 |
 *----------------------------------------------------------------------*/
Discret::Elements::TransportBoundary::TransportBoundary(int id, int owner, int nnode,
    const int* nodeids, Core::Nodes::Node** nodes, Discret::Elements::Transport* parent,
    const int lsurface)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_master_element(parent, lsurface);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        gjb 01/09 |
 *----------------------------------------------------------------------*/
Discret::Elements::TransportBoundary::TransportBoundary(
    const Discret::Elements::TransportBoundary& old)
    : Core::Elements::FaceElement(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it     (public) gjb 01/09 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::TransportBoundary::clone() const
{
  Discret::Elements::TransportBoundary* newelement =
      new Discret::Elements::TransportBoundary(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Return shape of this element                    (public)  gjb 01/09 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::TransportBoundary::shape() const
{
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_element()->shape());
}

/*----------------------------------------------------------------------*
 |  Pack data (public)                                        gjb 01/09 |
 *----------------------------------------------------------------------*/
void Discret::Elements::TransportBoundary::pack(Core::Communication::PackBuffer& data) const
{
  FOUR_C_THROW("This TransportBoundary element does not support communication");

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data (public)                                      gjb 01/09 |
 *----------------------------------------------------------------------*/
void Discret::Elements::TransportBoundary::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("This TransportBoundary element does not support communication");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                               gjb 01/09 |
 *----------------------------------------------------------------------*/
void Discret::Elements::TransportBoundary::print(std::ostream& os) const
{
  os << "TransportBoundary element";
  Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(shape()) << std::endl;
  std::cout << std::endl;
  return;
}

/*----------------------------------------------------------------------*
 | Return number of lines of boundary element (public)        gjb 01/09 |
 *----------------------------------------------------------------------*/
int Discret::Elements::TransportBoundary::num_line() const
{
  return Core::FE::get_number_of_element_lines(shape());
}

/*----------------------------------------------------------------------*
 |  Return number of surfaces of boundary element (public)    gjb 01/09 |
 *----------------------------------------------------------------------*/
int Discret::Elements::TransportBoundary::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(shape());
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                              gjb 01/09 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::TransportBoundary::lines()
{
  FOUR_C_THROW("Lines of TransportBoundary not implemented");
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                              gjb 01/09 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::TransportBoundary::surfaces()
{
  FOUR_C_THROW("Surfaces of TransportBoundary not implemented");
}

FOUR_C_NAMESPACE_CLOSE
