/*----------------------------------------------------------------------*/
/*! \file
\brief Evaluation of thermo-structure-interaction elements
       (structural part of coupling matrices)
\level 1
*/


/*----------------------------------------------------------------------*
 | headers                                                   dano 11/12 |
 *----------------------------------------------------------------------*/
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_robinson.hpp"
#include "4C_mat_thermoplastichyperelast.hpp"
#include "4C_mat_thermoplasticlinelast.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_mat_trait_thermo_solid.hpp"
#include "4C_so3_thermo.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | pre-evaluate the element (public)                         dano 08/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::pre_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::Element::LocationArray& la)
{
  // if the coupling variables are required before Evaluate() is called the 1st
  // time
  // here for Robinson's material
  if (la.Size() > 1)
  {
    // the temperature field has only one dof per node, disregarded by the
    // dimension of the problem
    const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);

    if (discretization.HasState(1, "temperature"))
    {
      if (la[1].Size() != nen_ * numdofpernode_thr)
        FOUR_C_THROW(
            "Location vector length for temperatures does not match!\n"
            "la[1].Size()= %i\tnen_*numdofpernode_thr= %i",
            la[1].Size(), nen_ * numdofpernode_thr);
      // check if you can get the temperature state
      Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
      if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'.");

      // extract local values of the global vectors
      Teuchos::RCP<std::vector<double>> nodaltempnp =
          Teuchos::rcp(new std::vector<double>(la[1].lm_.size()));
      Core::FE::ExtractMyValues(*tempnp, *nodaltempnp, la[1].lm_);

      // now set the current temperature vector in the parameter list
      params.set<Teuchos::RCP<std::vector<double>>>("nodal_tempnp", nodaltempnp);
    }
  }  // initial temperature dependence

  return;
}  // pre_evaluate()


/*----------------------------------------------------------------------*
 | evaluate the element (public)                             dano 08/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
int Discret::ELEMENTS::So3Thermo<so3_ele, distype>::Evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::Element::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // set the pointer to the parameter list in element
  so3_ele::set_params_interface_ptr(params);


  // what actions are available
  // (action == "calc_struct_stifftemp")
  // (action == "calc_struct_stress")
  // (action == default)

  // start with "none"
  typename So3Thermo::ActionType act = So3Thermo::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_stifftemp")
    act = So3Thermo::calc_struct_stifftemp;
  else if (action == "calc_struct_stress")
    act = So3Thermo::calc_struct_stress;

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // coupling terms K_dT in stiffness matrix K^{TSI} for monolithic TSI
    case So3Thermo::calc_struct_stifftemp:
    {
      if (la.Size() > 1)
      {
        evaluate_coupl_with_thr(params, discretization, la, elemat1_epetra, elemat2_epetra,
            elevec1_epetra, elevec2_epetra, elevec3_epetra);
      }
      break;
    }

    //==================================================================================
    default:
    {
      // in some cases we need to write/change some data before evaluating
      // here you can pass, e.g. for Robinson's material the current temperature
      // T_n+1 needed to calculate e.g. the young's modulus E(T_n+1)
      pre_evaluate(params, discretization, la);

      // call the purely structural methods
      so3_ele::Evaluate(params, discretization,
          la[0].lm_,  // only the first column, i.e. the structural field is passed
          elemat1_epetra, elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);

      // add the temperature-dependent terms to the structural field, i.e.
      // it's a TSI problem
      if (la.Size() > 1)
      {
        evaluate_coupl_with_thr(params, discretization,
            la,  // coupled TSI is considered, i.e. pass the compled location array
            elemat1_epetra, elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);
      }
      break;
    }  // default

  }  // action

  return 0;
}  // Evaluate()


/*----------------------------------------------------------------------*
 | evaluate the element (public)                             dano 08/12 |
 | here is the action for the coupling to the thermal field             |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
int Discret::ELEMENTS::So3Thermo<so3_ele, distype>::evaluate_coupl_with_thr(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::Element::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // start with "none"
  ActionType act = none;

  // get the required action for coupling with the thermal field
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_internalforce")
    act = calc_struct_internalforce;
  else if (action == "calc_struct_nlnstiff")
    act = calc_struct_nlnstiff;
  else if (action == "calc_struct_nlnstiffmass")
    act = calc_struct_nlnstiffmass;
  else if (action == "calc_struct_nlnstifflmass")
    act = calc_struct_nlnstifflmass;
  else if (action == "calc_struct_stifftemp")
    act = calc_struct_stifftemp;
  else if (action == "calc_struct_stress")
    act = calc_struct_stress;
  else if (action == "calc_struct_reset_istep")
    act = calc_struct_reset_istep;
  else if (action == "calc_struct_update_istep")
    act = calc_struct_update_istep;
  else if (action == "calc_struct_energy")
    act = calc_struct_energy;
  else if (action == "calc_struct_predict")
    return 0;
  else if (action == "calc_struct_recover")
    return 0;
  else
    FOUR_C_THROW("Unknown type of action for So3_Thermo: %s", action.c_str());

  // what should the element do
  switch (act)
  {
    //============================================================================
    // internal force vector for TSI only
    case calc_struct_internalforce:
    {
      // internal force vector
      Core::LinAlg::Matrix<numdofperelement_, 1> elevec1(elevec1_epetra.values(), true);
      // elemat1+2, elevec2+3 are not used anyway

      // need current displacement and residual/incremental displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState(0, "displacement");
      Teuchos::RCP<const Epetra_Vector> res = discretization.GetState(0, "residual displacement");

      if ((disp == Teuchos::null) or (res == Teuchos::null))
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");

      // build the location vector only for the structure field
      std::vector<double> mydisp((la[0].lm_).size());
      Core::FE::ExtractMyValues(*disp, mydisp, la[0].lm_);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      Core::LinAlg::Matrix<numdofperelement_, numdofperelement_> myemat(true);

      // initialise the vectors
      // Evaluate() is called the first time in ThermoBaseAlgorithm: at this stage the
      // coupling field is not yet known. Pass coupling vectors filled with zeros
      // the size of the vectors is the length of the location vector/nsd_
      std::vector<double> mytempnp(((la[0].lm_).size()) / nsd_, 0.0);

      // need current temperature state, call the temperature discretization
      // disassemble temperature
      if (discretization.HasState(1, "temperature"))
      {
        // check if you can get the temperature state
        Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
        if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'");

        // the temperature field has only one dof per node, disregarded by the
        // dimension of the problem
        const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);
        if (la[1].Size() != nen_ * numdofpernode_thr)
          FOUR_C_THROW("Location vector length for temperature does not match!");
        // extract the current temperatures
        Core::FE::ExtractMyValues(*tempnp, mytempnp, la[1].lm_);

        // default: geometrically non-linear analysis with Total Lagrangean approach
        if (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)
        {
          Core::LinAlg::Matrix<numdofperelement_, numdofperelement_> elemat1(
              elemat1_epetra.values(), true);

          // in case we have a finite strain thermoplastic material use hex8fbar element
          // to cirucumvent volumetric locking
          Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*
              eleFBAR = dynamic_cast<Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar,
                  Core::FE::CellType::hex8>*>(this);

          // default structural element
          if (!eleFBAR)
          {
            nln_stifffint_tsi(la,        // location array
                discretization,          // discr
                mydisp,                  // current displacements
                mytempnp,                // current temperature
                nullptr,                 // element stiffness matrix
                &elevec1,                // element internal force vector
                nullptr,                 // stresses at GP
                params,                  // algorithmic parameters e.g. time
                Inpar::STR::stress_none  // stress output option
            );
          }     // so3_ele
          else  // Hex8Fbar
          {
            nln_stifffint_tsi_fbar(la,   // location array
                mydisp,                  // current displacements
                mytempnp,                // current temperature
                nullptr,                 // element stiffness matrix
                &elevec1,                // element internal force vector
                nullptr,                 // stresses at GP
                params,                  // algorithmic parameters e.g. time
                Inpar::STR::stress_none  // stress output option
            );
          }  // Hex8Fbar
        }    // (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)

        // geometric Inpar::STR::KinemType::linear
        else if (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
        {
          // calculate the THERMOmechanical term for fint
          lin_fint_tsi(la, mydisp, mytempnp, &elevec1, nullptr, params, Inpar::STR::stress_none);
        }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
      }
      break;
    }  // calc_struct_internalforce

    //============================================================================
    // (non)linear stiffness for TSI
    case calc_struct_nlnstiff:
    {
      // internal force vector
      Core::LinAlg::Matrix<numdofperelement_, 1> elevec1(elevec1_epetra.values(), true);
      // elemat2, elevec2+3 are not used anyway
      // elemat1 only for geometrically nonlinear analysis

      // need current displacement and residual/incremental displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState(0, "displacement");

      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'displacement' ");

      // build the location vector only for the structure field
      std::vector<double> mydisp((la[0].lm_).size());
      Core::FE::ExtractMyValues(*disp, mydisp, la[0].lm_);

      // initialise the vectors
      // Evaluate() is called the first time in ThermoBaseAlgorithm: at this stage the
      // coupling field is not yet known. Pass coupling vectors filled with zeros
      // the size of the vectors is the length of the location vector/nsd_
      std::vector<double> mytempnp(((la[0].lm_).size()) / nsd_, 0.0);

      // need current temperature state, call the temperature discretization
      // disassemble temperature
      if (discretization.HasState(1, "temperature"))
      {
        // check if you can get the temperature state
        Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
        if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'");

        // the temperature field has only one dof per node, disregarded by the
        // dimension of the problem
        const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);
        if (la[1].Size() != nen_ * numdofpernode_thr)
          FOUR_C_THROW("Location vector length for temperature does not match!");
        // extract the current temperatures
        Core::FE::ExtractMyValues(*tempnp, mytempnp, la[1].lm_);

        // default: geometrically non-linear analysis with Total Lagrangean approach
        if (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)
        {
          // stiffness
          Core::LinAlg::Matrix<numdofperelement_, numdofperelement_> elemat1(
              elemat1_epetra.values(), true);

          Core::LinAlg::Matrix<numdofperelement_, numdofperelement_>* matptr = nullptr;
          if (elemat1.IsInitialized()) matptr = &elemat1;

          // in case we have a finite strain thermoplastic material use hex8fbar element
          // to cirucumvent volumetric locking
          Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*
              eleFBAR = dynamic_cast<Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar,
                  Core::FE::CellType::hex8>*>(this);

          // default structural element
          if (!eleFBAR)
          {
            nln_stifffint_tsi(la,        // location array
                discretization,          // discr
                mydisp,                  // current displacements
                mytempnp,                // current temperature
                matptr,                  // element stiffness matrix
                &elevec1,                // element internal force vector
                nullptr,                 // stresses at GP
                params,                  // algorithmic parameters e.g. time
                Inpar::STR::stress_none  // stress output option
            );
          }     // so3_ele
          else  // Hex8Fbar
          {
            nln_stifffint_tsi_fbar(la,   // location array
                mydisp,                  // current displacements
                mytempnp,                // current temperature
                matptr,                  // element stiffness matrix
                &elevec1,                // element internal force vector
                nullptr,                 // stresses at GP
                params,                  // algorithmic parameters e.g. time
                Inpar::STR::stress_none  // stress output option
            );
          }  // Hex8Fbar
        }    // (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)

        // geometric linear
        else if (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
        {
          // calculate the THERMOmechanical term for fint
          lin_fint_tsi(la, mydisp, mytempnp, &elevec1, nullptr, params, Inpar::STR::stress_none);
        }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
      }

      break;
    }  // calc_struct_nlnstiff

    //============================================================================
    // (non)linear stiffness, mass matrix and internal force vector for TSI
    case calc_struct_nlnstiffmass:
    case calc_struct_nlnstifflmass:
    {
      // internal force vector
      Core::LinAlg::Matrix<numdofperelement_, 1> elevec1(elevec1_epetra.values(), true);
      // elevec2+3 and elemat2 are not used anyway,
      // elemat1 only for geometrically nonlinear analysis

      // need current displacement and residual/incremental displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState(0, "displacement");

      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'displacement'");

      // build the location vector only for the structure field
      std::vector<double> mydisp((la[0].lm_).size());
      Core::FE::ExtractMyValues(*disp, mydisp, la[0].lm_);

      // initialise the vectors
      // Evaluate() is called the first time in structure_base_algorithm: at this
      // stage the coupling field is not yet known. Pass coupling vectors filled
      // with zeros
      // the size of the vectors is the length of the location vector/nsd_
      std::vector<double> mytempnp(((la[0].lm_).size()) / nsd_, 0.0);

      // need current temperature state, call the temperature discretization
      // disassemble temperature
      if (discretization.HasState(1, "temperature"))
      {
        // check if you can get the temperature state
        Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
        if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'");

        // the temperature field has only one dof per node, disregarded by the
        // dimension of the problem
        const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);
        if (la[1].Size() != nen_ * numdofpernode_thr)
          FOUR_C_THROW("Location vector length for temperature does not match!");
        // extract the current temperatures
        Core::FE::ExtractMyValues(*tempnp, mytempnp, la[1].lm_);

        // default: geometrically non-linear analysis with Total Lagrangean approach
        if (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)
        {
          // stiffness
          Core::LinAlg::Matrix<numdofperelement_, numdofperelement_> elemat1(
              elemat1_epetra.values(), true);

          // in case we have a finite strain thermoplastic material use hex8fbar element
          // to cirucumvent volumetric locking
          Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*
              eleFBAR = dynamic_cast<Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar,
                  Core::FE::CellType::hex8>*>(this);

          // default structural element
          if (!eleFBAR)
          {
            nln_stifffint_tsi(la,        // location array
                discretization,          // discr
                mydisp,                  // current displacements
                mytempnp,                // current temperature
                &elemat1,                // element stiffness matrix
                &elevec1,                // element internal force vector
                nullptr,                 // stresses at GP
                params,                  // algorithmic parameters e.g. time
                Inpar::STR::stress_none  // stress output option
            );
          }     // so3_ele
          else  // Hex8Fbar
          {
            nln_stifffint_tsi_fbar(la,   // location array
                mydisp,                  // current displacements
                mytempnp,                // current temperature
                &elemat1,                // element stiffness matrix
                &elevec1,                // element internal force vector
                nullptr,                 // stresses at GP
                params,                  // algorithmic parameters e.g. time
                Inpar::STR::stress_none  // stress output option
            );
          }  // Hex8Fbar

        }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)

        // geometric linear
        else if (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
        {
          // build the current temperature vector
          Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> etemp(&(mytempnp[1]), true);  // view only!
          // calculate the THERMOmechanical term for fint
          lin_fint_tsi(la, mydisp, mytempnp, &elevec1, nullptr, params, Inpar::STR::stress_none);
        }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
      }

      break;
    }  // calc_struct_nlnstiff(l)mass

    //==================================================================================
    // evaluate stresses and strains at gauss points
    case calc_struct_stress:
    {
      // elemat1+2,elevec1-3 are not used anyway
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState(0, "displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'displacement'");

      std::vector<double> mydisp((la[0].lm_).size());
      Core::FE::ExtractMyValues(*disp, mydisp, la[0].lm_);

      Teuchos::RCP<std::vector<char>> couplstressdata;
      Inpar::STR::StressType iocouplstress;
      if (this->IsParamsInterface())
      {
        couplstressdata = this->str_params_interface().coupling_stress_data_ptr();
        iocouplstress = this->str_params_interface().get_coupling_stress_output_type();
      }
      else
      {
        couplstressdata = params.get<Teuchos::RCP<std::vector<char>>>("couplstress", Teuchos::null);
        iocouplstress = Core::UTILS::GetAsEnum<Inpar::STR::StressType>(
            params, "iocouplstress", Inpar::STR::stress_none);
      }

      // get the temperature dependent stress
      Core::LinAlg::Matrix<numgpt_post, numstr_> couplstress(true);

      // initialise the vectors
      // Evaluate() is called the first time in ThermoBaseAlgorithm: at this stage the
      // coupling field is not yet known. Pass coupling vectors filled with zeros
      // the size of the vectors is the length of the location vector/nsd_
      std::vector<double> mytempnp(((la[0].lm_).size()) / nsd_, 0.0);

      // need current temperature state,
      // call the temperature discretization: thermo equates 2nd dofset
      // disassemble temperature
      if (discretization.HasState(1, "temperature"))
      {
        // check if you can get the temperature state
        Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
        if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'");

        // the temperature field has only one dof per node, disregarded by the
        // dimension of the problem
        const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);
        if (la[1].Size() != nen_ * numdofpernode_thr)
          FOUR_C_THROW("Location vector length for temperature does not match!");

        // extract the current temperatures
        Core::FE::ExtractMyValues(*tempnp, mytempnp, la[1].lm_);

        // default: geometrically non-linear analysis with Total Lagrangean approach
        if (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)
        {
          // in case we have a finite strain thermoplastic material use hex8fbar element
          // to cirucumvent volumetric locking
          Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*
              eleFBAR = dynamic_cast<Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar,
                  Core::FE::CellType::hex8>*>(this);

#ifdef TSIASOUTPUT
          std::cout << "thermal stress" << couplstress << std::endl;
          std::cout << "iocouplstress = " << iocouplstress << std::endl;
#endif

          // default structural element
          if (!eleFBAR)
          {
            // calculate the thermal stress
            nln_stifffint_tsi(la,  // location array
                discretization,    // discr
                mydisp,            // current displacements
                mytempnp,          // current temperature
                nullptr,           // element stiffness matrix
                nullptr,           // element internal force vector
                &couplstress,      // stresses at GP
                params,            // algorithmic parameters e.g. time
                iocouplstress      // stress output option
            );
          }     // so3_ele
          else  // Hex8Fbar
          {
            nln_stifffint_tsi_fbar(la,  // location array
                mydisp,                 // current displacements
                mytempnp,               // current temperature
                nullptr,                // element stiffness matrix
                nullptr,                // element internal force vector
                &couplstress,           // stresses at GP
                params,                 // algorithmic parameters e.g. time
                iocouplstress           // stress output option
            );
          }  // Hex8Fbar

#ifdef TSIASOUTPUT
          std::cout << "thermal stress" << couplstress << std::endl;
#endif

        }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)

        // geometric linear
        else if (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
        {
          // purely structural method, this is the coupled routine, i.e., a 2nd
          // discretisation exists, i.e., --> we always have a temperature state

          // calculate the THERMOmechanical term for fint: temperature stresses
          lin_fint_tsi(la, mydisp, mytempnp, nullptr, &couplstress, params, iocouplstress);
        }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)

#ifdef TSIASOUTPUT
        std::cout << "thermal stress" << couplstress << std::endl;
#endif
      }

      // total stress is the sum of the mechanical stress and the thermal stress
      // stress = stress_d + stress_T
      //        stress.Update(1.0,couplstress,1.0);
      // --> so far the addition of s_d and s_T was realised here
      // ==> from now on: we fill 2 different vectors (stressdata,couplstressdata)
      //     which are used in the post processing.
      //     --> advantage: different numbers of Gauss points for the stress
      //         and couplstress are possible
      //         --> important e.g. in case of Tet4 (s_d: 1GP, s_T: 5GP)
      //             --> for s_T we use the library intrepid
      //     --> in ParaView you can visualise the mechanical and the thermal
      //         stresses separately
      //     --> to get the total stress you have to calculate both vectors
      //         within ParaView using programmable filters

      // pack the data for postprocessing
      {
        Core::Communication::PackBuffer data;
        // get the size of stress
        so3_ele::AddtoPack(data, couplstress);
        data.StartPacking();
        // pack the stresses
        so3_ele::AddtoPack(data, couplstress);
        std::copy(data().begin(), data().end(), std::back_inserter(*couplstressdata));
      }

      break;
    }  // calc_struct_stress

    //============================================================================
    // required for predictor TangDis --> can be helpful in compressible case!
    case calc_struct_reset_istep:
    {
      // do nothing, actual implementation is in so3_ele
      break;
    }

    //============================================================================
    case calc_struct_update_istep:
    {
      auto thermoSolid = Teuchos::rcp_dynamic_cast<Mat::Trait::ThermoSolid>(material(), false);
      if (thermoSolid != Teuchos::null)
      {
        std::vector<double> mytempnp(((la[0].lm_).size()) / nsd_, 0.0);
        if (discretization.HasState(1, "temperature"))
        {
          // check if you can get the temperature state
          Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
          if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'");

          // the temperature field has only one dof per node, disregarded by the
          // dimension of the problem
          const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);
          if (la[1].Size() != nen_ * numdofpernode_thr)
            FOUR_C_THROW("Location vector length for temperature does not match!");
          // extract the current temperatures
          Core::FE::ExtractMyValues(*tempnp, mytempnp, la[1].lm_);
        }

        // vector of the current element temperatures
        Core::LinAlg::Matrix<nen_, 1> etemp(mytempnp.data());
        Core::LinAlg::Matrix<nen_, 1> shapefunct;

        // --------------------------------------------------
        // Initialisation of nurbs specific stuff
        std::vector<Core::LinAlg::SerialDenseVector> myknots(3);
        Core::LinAlg::Matrix<27, 1> weights;

        // get nurbs specific infos
        if (so3_ele::Shape() == Core::FE::CellType::nurbs27)
        {
          // cast to nurbs discretization
          Discret::Nurbs::NurbsDiscretization* nurbsdis =
              dynamic_cast<Discret::Nurbs::NurbsDiscretization*>(&(discretization));
          if (nurbsdis == nullptr)
            FOUR_C_THROW("So_nurbs27 appeared in non-nurbs discretisation\n");

          // zero-sized element
          if ((*((*nurbsdis).GetKnotVector())).GetEleKnots(myknots, id())) return 1;

          // get weights from cp's
          for (int inode = 0; inode < nen_; inode++)
            weights(inode) = dynamic_cast<Discret::Nurbs::ControlPoint*>(Nodes()[inode])->W();
        }

        for (int gp = 0; gp < numgpt_; ++gp)
        {
          if (so3_ele::Shape() != Core::FE::CellType::nurbs27)
          {
            Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
          }
          // evaluate shape functions NURBS-style
          else
            Core::FE::Nurbs::nurbs_get_3D_funct(
                shapefunct, xsi_[gp], myknots, weights, Core::FE::CellType::nurbs27);

          // product of shapefunctions and element temperatures
          Core::LinAlg::Matrix<1, 1> NT(false);
          NT.MultiplyTN(shapefunct, etemp);

          thermoSolid->Reinit(nullptr, nullptr, NT(0), map_my_gp_to_so_hex8(gp));
          thermoSolid->CommitCurrentState();
        }
      }

      break;
    }  // calc_struct_update_istep

    //============================================================================
    // coupling term k_dT of stiffness matrix for monolithic TSI
    case calc_struct_stifftemp:
    {
      // mechanical-thermal system matrix
      Core::LinAlg::Matrix<numdofperelement_, nen_> stiffmatrix_kdT(elemat1_epetra.values(), true);
      // elemat2,elevec1-3 are not used anyway
      // need current displacement and residual/incremental displacements
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState(0, "displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp((la[0].lm_).size());
      // build the location vector only for the structure field
      Core::FE::ExtractMyValues(*disp, mydisp, la[0].lm_);

      // initialise the vectors
      // Evaluate() is called the first time in structure_base_algorithm: at this
      // stage the coupling field is not yet known. Pass coupling vectors filled
      // with zeros
      // the size of the vectors is the length of the location vector/nsd_
      std::vector<double> mytempnp(((la[0].lm_).size()) / nsd_, 0.0);

      // need current temperature state, call the temperature discretization
      // disassemble temperature
      if (discretization.HasState(1, "temperature"))
      {
        // check if you can get the temperature state
        Teuchos::RCP<const Epetra_Vector> tempnp = discretization.GetState(1, "temperature");
        if (tempnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'tempnp'");

        // the temperature field has only one dof per node, disregarded by the
        // dimension of the problem
        const int numdofpernode_thr = discretization.NumDof(1, Nodes()[0]);
        if (la[1].Size() != nen_ * numdofpernode_thr)
          FOUR_C_THROW("Location vector length for temperature does not match!");
        // extract the current temperatures
        Core::FE::ExtractMyValues(*tempnp, mytempnp, la[1].lm_);
      }
      // default: geometrically non-linear analysis with Total Lagrangean approach
      if (so3_ele::KinematicType() == Inpar::STR::KinemType::nonlinearTotLag)
      {
        // in case we have a finite strain thermoplastic material use hex8fbar element
        // to cirucumvent volumetric locking
        Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*
            eleFBAR = dynamic_cast<Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar,
                Core::FE::CellType::hex8>*>(this);

        // default structural element
        if (!eleFBAR)
        {
          // calculate the mechanical-thermal sub matrix k_dT of K_TSI
          nln_kd_t_tsi(la, discretization, mydisp, mytempnp, &stiffmatrix_kdT, params);
        }     // so3_ele
        else  // Hex8Fbar
        {
          nln_kd_t_tsi_fbar(la, mydisp, mytempnp, &stiffmatrix_kdT, params);
        }  // Hex8Fbar
      }    // (so3_ele::KinematicType() == nonlinear)

      // geometric linear
      else if (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)
      {
        // calculate the mechanical-thermal sub matrix k_dT of K_TSI
        lin_kd_t_tsi(la, mydisp, mytempnp, &stiffmatrix_kdT, params);
      }  // (so3_ele::KinematicType() == Inpar::STR::KinemType::linear)

      break;
    }  // calc_struct_stifftemp

    // strain energy: the solid element knows what to do...
    case calc_struct_energy:
    {
      so3_ele::Evaluate(params, discretization, la[0].lm_, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }
    //============================================================================
    default:
      FOUR_C_THROW("Unknown type of action for So3_Thermo");
      break;
  }  // action

  return 0;
}  // evaluate_coupl_with_thr()


/*----------------------------------------------------------------------*
 | evaluate only the temperature fraction for the element    dano 05/10 |
 | contribution to r_d (private)                                        |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::lin_fint_tsi(
    Core::Elements::Element::LocationArray& la,             // location array
    std::vector<double>& disp,                              // current displacements
    std::vector<double>& temp,                              // current temperature
    Core::LinAlg::Matrix<numdofperelement_, 1>* force,      // element internal force vector
    Core::LinAlg::Matrix<numgpt_post, numstr_>* elestress,  // stresses at GP
    Teuchos::ParameterList& params,                         // algorithmic parameters e.g. time
    const Inpar::STR::StressType iostress                   // stress output option
)
{
  // update element geometry hex8, 3D: (8x3)
  Core::LinAlg::Matrix<nen_, nsd_> xrefe;  // X, material coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurr;  // x, current  coord. of element
  // vector of the current element temperatures
  Core::LinAlg::Matrix<nen_, 1> etemp;

  Core::Nodes::Node** nodes = Nodes();
  for (int i = 0; i < nen_; ++i)
  {
    const auto& x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];

    etemp(i, 0) = temp[i + 0];
  }

  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  Core::LinAlg::Matrix<nsd_, nen_> N_XYZ;
  // build deformation gradient w.r.t. to material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(true);
  // shape functions and their first derivatives
  Core::LinAlg::Matrix<nen_, 1> shapefunct;
  Core::LinAlg::Matrix<nsd_, nen_> deriv;

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // shape functions (shapefunct) and their first derivatives (deriv)
    Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
    Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);

    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
    double detJ = detJ_[gp];           // (6.22)

    // geometrically linear, i.e. reference == current state, i.e. F == I
    // set to initial state (defgrd == identity)
    for (int i = 0; i < 3; ++i) defgrd(i, i) = 1.0;

    // calculate the linear B-operator
    Core::LinAlg::Matrix<numstr_, numdofperelement_> boplin;
    calculate_boplin(&boplin, &N_XYZ);

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc

    // product of shapefunctions and element temperatures for couplstress
    // N_T . T
    Core::LinAlg::Matrix<1, 1> NT(false);
    NT.MultiplyTN(shapefunct, etemp);
    // scalar-valued current element temperature T_{n+1}
    // temperature-dependent material parameters, i.e. E(T), pass T_{n+1}
    double scalartemp = NT(0, 0);
    // insert T_{n+1} into parameter list
    params.set<double>("scalartemp", scalartemp);

    // calculate the stress part dependent on the temperature in the material
    Core::LinAlg::Matrix<numstr_, 1> ctemp(true);
    Core::LinAlg::Matrix<numstr_, 1> couplstress(true);
    Core::LinAlg::Matrix<numstr_, numstr_> cmat(true);
    Core::LinAlg::Matrix<numstr_, 1> glstrain(true);

    // insert strain increment into parameter list
    // calculate iterative strains
    Core::LinAlg::Matrix<numstr_, 1> straininc(true);
    params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("straininc", straininc);

    // take care: current temperature ( N . T ) is passed to the element
    //            in the material: 1.) Delta T = subtract ( N . T - T_0 )
    //                             2.) couplstress = C . Delta T
    // do not call the material for Robinson's material
    if (material()->MaterialType() != Core::Materials::m_vp_robinson)
      materialize(&couplstress, &ctemp, &NT, &cmat, &glstrain, params);

    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp stresses
    switch (iostress)
    {
      case Inpar::STR::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");

        for (int i = 0; i < numstr_; ++i) (*elestress)(gp, i) = couplstress(i);
        break;
      }
      case Inpar::STR::stress_cauchy:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");

        // push forward of material stress to the spatial configuration
        Core::LinAlg::Matrix<nsd_, nsd_> cauchycouplstress;
        p_k2to_cauchy(&couplstress, &defgrd, &cauchycouplstress);

        (*elestress)(gp, 0) = cauchycouplstress(0, 0);
        (*elestress)(gp, 1) = cauchycouplstress(1, 1);
        (*elestress)(gp, 2) = cauchycouplstress(2, 2);
        (*elestress)(gp, 3) = cauchycouplstress(0, 1);
        (*elestress)(gp, 4) = cauchycouplstress(1, 2);
        (*elestress)(gp, 5) = cauchycouplstress(0, 2);
        break;
      }
      case Inpar::STR::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not available");
        break;
    }

    // integrate internal force vector r_d
    // f = f + (B^T . sigma_temp) * detJ * w(gp)
    if (force != nullptr)
    {
      // old implementation hex8_thermo double detJ_w = detJ*gpweights[gp];
      double detJ_w = detJ * intpoints_.Weight(gp);  // gpweights[gp];
      force->MultiplyTN(detJ_w, boplin, couplstress, 1.0);
    }  // if (force != nullptr)

    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/

  return;
}  // lin_fint_tsi()


/*----------------------------------------------------------------------*
 | evaluate only the mechanical-thermal stiffness term       dano 03/11 |
 | for monolithic TSI, contribution to k_dT (private)                   |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::lin_kd_t_tsi(
    Core::Elements::Element::LocationArray& la,
    std::vector<double>& disp,                                       // current displacement
    std::vector<double>& temp,                                       // current temperatures
    Core::LinAlg::Matrix<numdofperelement_, nen_>* stiffmatrix_kdT,  // (nsd_*nen_ x nen_)
    Teuchos::ParameterList& params)
{
  // update element geometry (8x3)
  Core::LinAlg::Matrix<nen_, nsd_> xrefe(false);  // X, material coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurr(false);  // x, current  coord. of element
  Core::Nodes::Node** nodes = Nodes();
  for (int i = 0; i < nen_; ++i)
  {
    const auto& x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];
  }

  // get current element displacements in vector notation --> use for elestrain
  Core::LinAlg::Matrix<numdofperelement_, 1> edisp(false);
  for (int i = 0; i < numdofperelement_; i++)
  {
    edisp(i, 0) = disp[i + 0];
  }

  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  Core::LinAlg::Matrix<nsd_, nen_> N_XYZ;
  // shape functions and their first derivatives
  Core::LinAlg::Matrix<nen_, 1> shapefunct;
  Core::LinAlg::Matrix<nsd_, nen_> deriv;

  // ------------------------------------------------ initialise material
  // get the thermal material tangent
  Core::LinAlg::Matrix<numstr_, 1> ctemp(true);

  // for the stiffness matrix k_dT
  Core::LinAlg::Matrix<nen_, 1> etemp(false);

  // get the temperature vector
  for (int i = 0; i < nen_; ++i) etemp(i, 0) = temp[i];

  // initialize matrices required for k_dT
  Core::LinAlg::Matrix<numdofperelement_, 1> Bstress_T(true);
  Core::LinAlg::Matrix<numdofperelement_, 1> Bcouplstress_T(true);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // shape functions (shapefunct) and their first derivatives (deriv)
    Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
    Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);

    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
    double detJ = detJ_[gp];           // (6.22)

    // calculate the linear B-operator B_L = N_XYZ
    Core::LinAlg::Matrix<numstr_, numdofperelement_> boplin;
    calculate_boplin(&boplin, &N_XYZ);

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc

    Teuchos::RCP<Mat::Trait::ThermoSolid> thermoSolidMaterial =
        Teuchos::rcp_dynamic_cast<Mat::Trait::ThermoSolid>(material(), false);
    if (thermoSolidMaterial != Teuchos::null)
    {
      // calculate the nodal strains: strain = B . d
      Core::LinAlg::Matrix<numstr_, 1> strain(false);
      strain.Multiply(boplin, edisp);

      Core::LinAlg::Matrix<1, 1> NT(false);
      NT.MultiplyTN(shapefunct, etemp);  // (1x1)
      thermoSolidMaterial->Reinit(nullptr, &strain, NT(0), map_my_gp_to_so_hex8(gp));
      // full thermal derivative of stress wrt to scalar temperature (needs to be post-multiplied
      // with shape functions)
      thermoSolidMaterial->GetdSdT(&ctemp);
    }
    // get thermal material tangent
    else
    {
      compute_ctemp(&ctemp, params);
    }
    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    double detJ_w = detJ * intpoints_.Weight(gp);
    // update linear coupling matrix K_dT
    if (stiffmatrix_kdT != nullptr)
    {
      // C_T . N_T
      Core::LinAlg::Matrix<numstr_, nen_> cn(false);
      cn.MultiplyNT(ctemp, shapefunct);  // (6x8)=(6x1)(1x8)
      // integrate stiffness term
      // k_dT = k_dT + (B^T . C_T . N_T) * detJ * w(gp)
      stiffmatrix_kdT->MultiplyTN(detJ_w, boplin, cn, 1.0);

      // in case of temperature-dependent Young's modulus, additional term for
      // coupling stiffness matrix k_dT
      {
        // k_dT += B_d^T . dC/dT  . B_d . d . N_T
        stiffmatrix_kdT->MultiplyNT(detJ_w, Bstress_T, shapefunct, 1.0);

        // k_dT += B_d^T . dC_T/dT  . N_T . T . N_T
        // (24x8)                          (24x1)        (8x1)
        stiffmatrix_kdT->MultiplyNT(detJ_w, Bcouplstress_T, shapefunct, 1.0);
      }

      // Be careful: scaling with time factor is done in tsi_monolithic!!
    }  // (stiffmatrix_kdT != nullptr)
    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/

  return;
}  // lin_kdT_tsi()


/*----------------------------------------------------------------------*
 | evaluate only the temperature fraction for the element    dano 03/10 |
 | originally by maf 04/07  (private)                                   |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::nln_stifffint_tsi(
    Core::Elements::Element::LocationArray& la,  // location array
    Core::FE::Discretization& discretization,    ///< discretisation to extract knot vector
    std::vector<double>& disp,                   // current displacements
    std::vector<double>& temp,                   // current temperature
    Core::LinAlg::Matrix<numdofperelement_, numdofperelement_>*
        stiffmatrix,                                        // element stiffness matrix
    Core::LinAlg::Matrix<numdofperelement_, 1>* force,      // element internal force vector
    Core::LinAlg::Matrix<numgpt_post, numstr_>* elestress,  // stresses at GP
    Teuchos::ParameterList& params,                         // algorithmic parameters e.g. time
    const Inpar::STR::StressType iostress                   // stress output option
)
{
  // update element geometry hex8, 3D: (8x3)
  Core::LinAlg::Matrix<nen_, nsd_> xrefe;  // X, material coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurr;  // x, current coord. of element
  // vector of the current element temperatures
  Core::LinAlg::Matrix<nen_, 1> etemp;

  Core::Nodes::Node** nodes = Nodes();
  for (int i = 0; i < nen_; ++i)
  {
    const auto& x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];

    etemp(i, 0) = temp[i + 0];
  }

  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  Core::LinAlg::Matrix<nsd_, nen_> N_XYZ(false);
  // build deformation gradient w.r.t. to material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(false);
  // shape functions and their first derivatives
  Core::LinAlg::Matrix<nen_, 1> shapefunct(false);
  Core::LinAlg::Matrix<nsd_, nen_> deriv(false);

  // --------------------------------------------------
  // Initialisation of nurbs specific stuff
  std::vector<Core::LinAlg::SerialDenseVector> myknots(3);
  Core::LinAlg::Matrix<27, 1> weights;

  // get nurbs specific infos
  if (so3_ele::Shape() == Core::FE::CellType::nurbs27)
  {
    // cast to nurbs discretization
    Discret::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Discret::Nurbs::NurbsDiscretization*>(&(discretization));
    if (nurbsdis == nullptr) FOUR_C_THROW("So_nurbs27 appeared in non-nurbs discretisation\n");

    // zero-sized element
    if ((*((*nurbsdis).GetKnotVector())).GetEleKnots(myknots, id())) return;

    // get weights from cp's
    for (int inode = 0; inode < nen_; inode++)
      weights(inode) = dynamic_cast<Discret::Nurbs::ControlPoint*>(Nodes()[inode])->W();
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // shape functions (shapefunct) and their first derivatives (deriv)
    if (so3_ele::Shape() != Core::FE::CellType::nurbs27)
    {
      Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);
    }
    // evaluate shape functions NURBS-style
    else
      Core::FE::Nurbs::nurbs_get_3D_funct_deriv(
          shapefunct, deriv, xsi_[gp], myknots, weights, Core::FE::CellType::nurbs27);

    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
    double detJ = detJ_[gp];           // (6.22)

    // (material) deformation gradient
    // F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
    defgrd.MultiplyTT(xcurr, N_XYZ);

    // right Cauchy-Green tensor = F^T . F
    Core::LinAlg::Matrix<nsd_, nsd_> cauchygreen(false);
    cauchygreen.MultiplyTN(defgrd, defgrd);

    // inverse of right Cauchy-Green tensor = F^{-1} . F^{-T}
    Core::LinAlg::Matrix<nsd_, nsd_> Cinv(false);
    Cinv.Invert(cauchygreen);
    Core::LinAlg::Matrix<numstr_, 1> Cinv_vct(false);
    Cinv_vct(0) = Cinv(0, 0);
    Cinv_vct(1) = Cinv(1, 1);
    Cinv_vct(2) = Cinv(2, 2);
    Cinv_vct(3) = Cinv(0, 1);
    Cinv_vct(4) = Cinv(1, 2);
    Cinv_vct(5) = Cinv(2, 0);

    // calculate linear B-operator
    Core::LinAlg::Matrix<numstr_, numdofperelement_> boplin(false);
    calculate_boplin(&boplin, &N_XYZ);

    // calculate nonlinear B-operator
    Core::LinAlg::Matrix<numstr_, numdofperelement_> bop(false);
    calculate_bop(&bop, &defgrd, &N_XYZ);

    // temperature
    // described as a matrix (for stress calculation): NT = N_T . T
    Core::LinAlg::Matrix<1, 1> NT(false);
    NT.MultiplyTN(shapefunct, etemp);
    // scalar-valued current element temperature T_{n+1}
    // temperature-dependent material parameters, i.e. E(T), pass T_{n+1}
    // insert T_{n+1} into parameter list
    params.set<double>("scalartemp", NT(0, 0));

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc

    // calculate the stress part dependent on the temperature in the material
    Core::LinAlg::Matrix<numstr_, 1> ctemp(true);
    Core::LinAlg::Matrix<numstr_, 1> couplstress(true);
    Core::LinAlg::Matrix<numstr_, numstr_> cmat_T(true);
    Core::LinAlg::Matrix<numstr_, 1> glstrain(true);

    // insert strain increment into parameter list
    // calculate iterative strains
    Core::LinAlg::Matrix<numstr_, 1> straininc(true);
    params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("straininc", straininc);
    // insert matrices into parameter list which are only required for thrplasthyperelast
    params.set<Core::LinAlg::Matrix<nsd_, nsd_>>("defgrd", defgrd);
    params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("Cinv_vct", Cinv_vct);

    // take care: current temperature ( N . T ) is passed to the element
    //            in the material: 1.) Delta T = subtract ( N . T - T_0 )
    //                             2.) couplstress = C . Delta T
    // do not call the material for Robinson's material
    if (material()->MaterialType() != Core::Materials::m_vp_robinson)
      materialize(&couplstress, &ctemp, &NT, &cmat_T, &glstrain, params);

    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp stresses
    switch (iostress)
    {
      case Inpar::STR::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        for (int i = 0; i < numstr_; ++i) (*elestress)(gp, i) = couplstress(i);
        break;
      }
      case Inpar::STR::stress_cauchy:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");

        // push forward of material stress to the spatial configuration
        // sigma = 1/J . F . S_temp . F^T
        Core::LinAlg::Matrix<nsd_, nsd_> cauchycouplstress(false);
        p_k2to_cauchy(&couplstress, &defgrd, &cauchycouplstress);

        (*elestress)(gp, 0) = cauchycouplstress(0, 0);
        (*elestress)(gp, 1) = cauchycouplstress(1, 1);
        (*elestress)(gp, 2) = cauchycouplstress(2, 2);
        (*elestress)(gp, 3) = cauchycouplstress(0, 1);
        (*elestress)(gp, 4) = cauchycouplstress(1, 2);
        (*elestress)(gp, 5) = cauchycouplstress(0, 2);
        break;
      }
      case Inpar::STR::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not available");
        break;
    }

    // integrate internal force vector r_d
    // f = f + (B^T . sigma_temp) . detJ . w(gp)
    double detJ_w = detJ * intpoints_.Weight(gp);
    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) . detJ . w(gp)
      force->MultiplyTN(detJ_w, bop, couplstress, 1.0);
    }

    // update stiffness matrix k_dd
    if (stiffmatrix != nullptr)
    {
      // integrate temperature-dependent `elastic' and `initial-displacement'
      // stiffness matrix
      // keu = keu + (B^T . Cmat_T . B) . detJ . w(gp)
      // Neo-Hookean type: dC_T/dd = m . Delta T . (-1) . ( Cinv boeppel Cinv )_{abcd}
      // St.Venant Kirchhoff: dC_T/dd == 0
      // with ( Cinv boeppel Cinv )_{abcd} = 1/2 . ( Cinv_{ac} Cinv_{bd} + Cinv_{ad} Cinv_{bc} )
      Core::LinAlg::Matrix<numstr_, numdofperelement_> cb(false);  // cb(6x24) // cmattemp (6x6)
      cb.Multiply(cmat_T, bop);
      stiffmatrix->MultiplyTN(detJ_w, bop, cb, 1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************

      // kgeo += ( B_L^T . B_L . sigma_temp) . detJ . w(gp)
      // (B_L^T . sigma . B_L) = (24x6)(6x1)(6x24)
      // --> size of matrices do not fit --> multiply component-by-component
      // with linear B-operator B_L = Ni,Xj, see NiliFEM-Skript (6.20)

      Core::LinAlg::Matrix<numstr_, 1> sfac(couplstress);  // auxiliary integrated stress
      // detJ . w(gp) . [S11,S22,S33,S12=S21,S23=S32,S13=S31]
      sfac.Scale(detJ_w);
      // intermediate sigma_temp . B_L (6x1).(6x24)
      std::vector<double> StempB_L(3);
      for (int inod = 0; inod < nen_; ++inod)
      {
        // (3x1) = (6x1) (6x24)
        // S11*N_XYZ(1,i)+S23*N_XYZ(2,i)+S12*N_XYZ(3,i)
        StempB_L[0] =
            sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
        // S23*N_XYZ(1,i)+S22*N_XYZ(2,i)+S13*N_XYZ(3,i)
        StempB_L[1] =
            sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
        // S12*N_XYZ(1,i)+S13*N_XYZ(2,i)+S33*N_XYZ(3,i)
        StempB_L[2] =
            sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
        // (B_L^T . sigma . B_L) = (24x6)(6x24)
        for (int jnod = 0; jnod < nen_; ++jnod)
        {
          double bopstrbop = 0.0;  // intermediate value
          for (int idim = 0; idim < nsd_; ++idim)
          {
            // double     (3x8)                 (3x1)
            bopstrbop += N_XYZ(idim, jnod) * StempB_L[idim];
          }
          // (24x24)
          (*stiffmatrix)(3 * inod + 0, 3 * jnod + 0) += bopstrbop;
          (*stiffmatrix)(3 * inod + 1, 3 * jnod + 1) += bopstrbop;
          (*stiffmatrix)(3 * inod + 2, 3 * jnod + 2) += bopstrbop;
        }
      }  // end of integrate `geometric' stiffness******************************
    }    // fill k_dd
    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/

  return;
}  // nln_stifffint_tsi()


/*----------------------------------------------------------------------*
 | evaluate only the mechanical-thermal stiffness term       dano 11/12 |
 | for monolithic TSI, contribution to k_dT (private)                   |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::nln_kd_t_tsi(
    Core::Elements::Element::LocationArray& la,
    Core::FE::Discretization& discretization,  ///< discretisation to extract knot vector
    std::vector<double>& disp,                 // current displacement
    std::vector<double>& temp,                 // current temperature
    Core::LinAlg::Matrix<numdofperelement_, nen_>* stiffmatrix_kdT,  // (nsd_*nen_ x nen_)
    Teuchos::ParameterList& params)
{
  // update element geometry (8x3)
  Core::LinAlg::Matrix<nen_, nsd_> xrefe(false);  // X, material coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurr(false);  // x, current  coord. of element
  Core::Nodes::Node** nodes = Nodes();
  for (int i = 0; i < nen_; ++i)
  {
    const auto& x = nodes[i]->X();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];
  }

  // ------------------------------------------------ initialise material
  // get the thermal material tangent
  Core::LinAlg::Matrix<numstr_, 1> ctemp(true);

  // ------------------------------ temperature-dependent Young's modulus
  // if young's modulus is temperature-dependent, E(T) additional terms arise
  // for the stiffness matrix k_dT
  Core::LinAlg::Matrix<nen_, 1> etemp(true);

  // get the temperature vector
  for (int i = 0; i < nen_; ++i) etemp(i, 0) = temp[i];

  // initialise matrices required for k_dT
  Core::LinAlg::Matrix<numdofperelement_, 1> Bstress_T(true);
  Core::LinAlg::Matrix<numdofperelement_, 1> Bcouplstress_T(true);

  // shape functions and their first derivatives
  Core::LinAlg::Matrix<nen_, 1> shapefunct(false);
  Core::LinAlg::Matrix<nsd_, nen_> deriv(false);
  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  Core::LinAlg::Matrix<nsd_, nen_> N_XYZ(false);
  // build deformation gradient w.r.t. to material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(false);

  // --------------------------------------------------
  // Initialisation of nurbs specific stuff
  std::vector<Core::LinAlg::SerialDenseVector> myknots(3);
  Core::LinAlg::Matrix<27, 1> weights;

  // get nurbs specific infos
  if (so3_ele::Shape() == Core::FE::CellType::nurbs27)
  {
    // cast to nurbs discretization
    Discret::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Discret::Nurbs::NurbsDiscretization*>(&(discretization));
    if (nurbsdis == nullptr) FOUR_C_THROW("So_nurbs27 appeared in non-nurbs discretisation\n");

    // zero-sized element
    if ((*((*nurbsdis).GetKnotVector())).GetEleKnots(myknots, id())) return;

    // get weights from cp's
    for (int inode = 0; inode < nen_; inode++)
      weights(inode) = dynamic_cast<Discret::Nurbs::ControlPoint*>(Nodes()[inode])->W();
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // shape functions (shapefunct) and their first derivatives (deriv)
    if (so3_ele::Shape() != Core::FE::CellType::nurbs27)
    {
      Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);
    }
    // evaluate shape functions NURBS-style
    else
      Core::FE::Nurbs::nurbs_get_3D_funct_deriv(
          shapefunct, deriv, xsi_[gp], myknots, weights, Core::FE::CellType::nurbs27);

    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 . N_rst
    N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
    double detJ = detJ_[gp];           // (6.22)

    // (material) deformation gradient
    // F = d xcurr / d xrefe = xcurr^T . N_XYZ^T
    defgrd.MultiplyTT(xcurr, N_XYZ);

    // right Cauchy-Green tensor = F^T . F
    Core::LinAlg::Matrix<3, 3> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);
    // initialise inverse of right Cauchy-Green tensor = F^{-1} . F^{-T}
    Core::LinAlg::Matrix<numstr_, 1> Cinv_vct(true);

    // calculate nonlinear B-operator
    Core::LinAlg::Matrix<numstr_, numdofperelement_> bop(false);
    calculate_bop(&bop, &defgrd, &N_XYZ);

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc

    Teuchos::RCP<Mat::Trait::ThermoSolid> thermoSolidMaterial =
        Teuchos::rcp_dynamic_cast<Mat::Trait::ThermoSolid>(material(), false);
    if (thermoSolidMaterial != Teuchos::null)
    {
      Core::LinAlg::Matrix<numstr_, 1> glstrain(false);
      glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
      glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
      glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
      glstrain(3) = cauchygreen(0, 1);  // Voigt notation
      glstrain(4) = cauchygreen(1, 2);
      glstrain(5) = cauchygreen(2, 0);

      Core::LinAlg::Matrix<1, 1> NT(false);
      NT.MultiplyTN(shapefunct, etemp);  // (1x1)
      thermoSolidMaterial->Reinit(nullptr, &glstrain, NT(0), map_my_gp_to_so_hex8(gp));
      // full thermal derivative of stress wrt to scalar temperature (needs to be post-multiplied
      // with shape functions)
      thermoSolidMaterial->GetdSdT(&ctemp);
    }
    else if (material()->MaterialType() == Core::Materials::m_thermoplhyperelast)
    {
      // inverse of Right Cauchy-Green tensor = F^{-1} . F^{-T}
      Core::LinAlg::Matrix<nsd_, nsd_> Cinv(false);
      Cinv.Invert(cauchygreen);
      Cinv_vct(0) = Cinv(0, 0);
      Cinv_vct(1) = Cinv(1, 1);
      Cinv_vct(2) = Cinv(2, 2);
      Cinv_vct(3) = Cinv(0, 1);
      Cinv_vct(4) = Cinv(1, 2);
      Cinv_vct(5) = Cinv(2, 0);

      params.set<Core::LinAlg::Matrix<nsd_, nsd_>>("defgrd", defgrd);
      params.set<Core::LinAlg::Matrix<numstr_, 1>>("Cinv_vct", Cinv_vct);

      Teuchos::RCP<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
          Teuchos::rcp_dynamic_cast<Mat::ThermoPlasticHyperElast>(material(), true);
      // get thermal material tangent
      thermoplhyperelast->setup_cthermo(ctemp, params);
    }  // m_thermoplhyperelast

    // get thermal material tangent
    else
      compute_ctemp(&ctemp, params);

    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    double detJ_w = detJ * intpoints_.Weight(gp);
    // update linear coupling matrix K_dT
    if (stiffmatrix_kdT != nullptr)
    {
      // C_temp . N_temp
      Core::LinAlg::Matrix<numstr_, nen_> cn(false);
      cn.MultiplyNT(ctemp, shapefunct);  // (6x8)=(6x1)(1x8)
      // integrate stiffness term
      // k_dT = k_dT + (B^T . C_T . N_temp) . detJ . w(gp)
      stiffmatrix_kdT->MultiplyTN(detJ_w, bop, cn, 1.0);

      // in case of temperature-dependent Young's modulus, additional term for
      // coupling stiffness matrix k_dT
      if ((material()->MaterialType() == Core::Materials::m_thermostvenant))
      {
        // k_dT += B_d^T . stress_T . N_T
        stiffmatrix_kdT->MultiplyNT(detJ_w, Bstress_T, shapefunct, 1.0);

        // k_dT += B_d^T . couplstress_T . N_T
        // (24x8)                          (24x1)        (8x1)
        stiffmatrix_kdT->MultiplyNT(detJ_w, Bcouplstress_T, shapefunct, 1.0);

        // Be careful: scaling with time factor is done in tsi_monolithic!!
      }  // m_thermostvenant
    }
    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/

  return;
}  // nln_kdT_tsi()


/*----------------------------------------------------------------------*
 | evaluate only the temperature fraction for the element    dano 05/13 |
 | only in case of hex8fbar element AND geo_nln analysis (protected)    |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::nln_stifffint_tsi_fbar(
    Core::Elements::Element::LocationArray& la,  // location array
    std::vector<double>& disp,                   // current displacements
    std::vector<double>& temp,                   // current temperature
    Core::LinAlg::Matrix<numdofperelement_, numdofperelement_>*
        stiffmatrix,                                        // element stiffness matrix
    Core::LinAlg::Matrix<numdofperelement_, 1>* force,      // element internal force vector
    Core::LinAlg::Matrix<numgpt_post, numstr_>* elestress,  // stresses at GP
    Teuchos::ParameterList& params,                         // algorithmic parameters e.g. time
    const Inpar::STR::StressType iostress                   // stress output option
)
{
  // in case we have a finite strain thermoplastic material use hex8fbar element
  // to cirucumvent volumetric locking
  Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>* eleFBAR =
      dynamic_cast<
          Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*>(
          this);

  if ((distype == Core::FE::CellType::hex8) and (eleFBAR))
  {
    // update element geometry hex8, 3D: (8x3)
    Core::LinAlg::Matrix<nen_, nsd_> xrefe(false);  // X, material coord. of element
    Core::LinAlg::Matrix<nen_, nsd_> xcurr(false);  // x, current  coord. of element
    // vector of the current element temperatures
    Core::LinAlg::Matrix<nen_, 1> etemp(false);

    Core::Nodes::Node** nodes = Nodes();
    for (int i = 0; i < nen_; ++i)
    {
      const auto& x = nodes[i]->X();
      xrefe(i, 0) = x[0];
      xrefe(i, 1) = x[1];
      xrefe(i, 2) = x[2];

      xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
      xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
      xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];

      etemp(i, 0) = temp[i + 0];
    }

    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    Core::LinAlg::Matrix<nsd_, nen_> N_XYZ(false);
    // build deformation gradient w.r.t. to material configuration
    Core::LinAlg::Matrix<nsd_, nsd_> defgrd(false);
    // shape functions and their first derivatives
    Core::LinAlg::Matrix<nen_, 1> shapefunct(false);
    Core::LinAlg::Matrix<nsd_, nen_> deriv(false);

    // ---------------------- deformation gradient at centroid of element
    double detF_0 = -1.0;
    Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd_0(false);
    Core::LinAlg::Matrix<nsd_, nen_> N_XYZ_0(false);
    // element coordinate derivatives at centroid
    Core::LinAlg::Matrix<nsd_, nen_> N_rst_0(false);
    Core::FE::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);

    // inverse jacobian matrix at centroid
    Core::LinAlg::Matrix<nsd_, nsd_> invJ_0(false);
    invJ_0.Multiply(N_rst_0, xrefe);
    invJ_0.Invert();
    // material derivatives at centroid
    N_XYZ_0.Multiply(invJ_0, N_rst_0);

    // deformation gradient and its determinant at centroid
    Core::LinAlg::Matrix<3, 3> defgrd_0(false);
    defgrd_0.MultiplyTT(xcurr, N_XYZ_0);
    invdefgrd_0.Invert(defgrd_0);
    detF_0 = defgrd_0.Determinant();

    /* =========================================================================*/
    /* ================================================= Loop over Gauss Points */
    /* =========================================================================*/
    for (int gp = 0; gp < numgpt_; ++gp)
    {
      // shape functions (shapefunct) and their first derivatives (deriv)
      Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);

      /* get the inverse of the Jacobian matrix which looks like:
      **            [ x_,r  y_,r  z_,r ]^-1
      **     J^-1 = [ x_,s  y_,s  z_,s ]
      **            [ x_,t  y_,t  z_,t ]
      */
      // compute derivatives N_XYZ at gp w.r.t. material coordinates
      // by N_XYZ = J^-1 * N_rst
      N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
      double detJ = detJ_[gp];           // (6.22)

      // (material) deformation gradient
      // F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
      defgrd.MultiplyTT(xcurr, N_XYZ);
      double detF = defgrd.Determinant();
      Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd(false);
      invdefgrd.Invert(defgrd);

      // Right Cauchy-Green tensor = F^T . F
      Core::LinAlg::Matrix<nsd_, nsd_> cauchygreen(false);
      cauchygreen.MultiplyTN(defgrd, defgrd);

      // build the inverse of the right Cauchy-Green deformation gradient C^{-1}
      // C^{-1} = F^{-1} . F^{-T}
      Core::LinAlg::Matrix<nsd_, nsd_> invC(false);
      invC.MultiplyNT(invdefgrd, invdefgrd);
      // invCvct: C^{-1} in Voight-/vector notation
      // C^{-1} = { C11^{-1}, C22^{-1}, C33^{-1}, C12^{-1}, C23^{-1}, C31^{-1} }
      Core::LinAlg::Matrix<6, 1> invCvct(false);
      invCvct(0) = invC(0, 0);
      invCvct(1) = invC(1, 1);
      invCvct(2) = invC(2, 2);
      invCvct(3) = invC(0, 1);
      invCvct(4) = invC(1, 2);
      invCvct(5) = invC(2, 0);

      // -------------------------------------------- F_bar modifications
      // F_bar deformation gradient: F_bar := (detF_0 / detF)^1/3 . F
      Core::LinAlg::Matrix<nsd_, nsd_> defgrd_bar(defgrd);
      // f_bar_factor := (detF_0/detF)^{1/3}
      double f_bar_factor = std::pow(detF_0 / detF, 1.0 / 3.0);
      defgrd_bar.Scale(f_bar_factor);

      // Right Cauchy-Green tensor(Fbar) = F_bar^T . F_bar
      Core::LinAlg::Matrix<nsd_, nsd_> cauchygreen_bar(false);
      cauchygreen_bar.MultiplyTN(defgrd_bar, defgrd_bar);

      // inverse of Right Cauchy-Green tensor(Fbar) = F_bar^{-1} . F_bar^{-T}
      Core::LinAlg::Matrix<nsd_, nsd_> Cinv_bar(false);
      Cinv_bar.Invert(cauchygreen_bar);
      Core::LinAlg::Matrix<numstr_, 1> Cinv_barvct(false);
      Cinv_barvct(0) = Cinv_bar(0, 0);
      Cinv_barvct(1) = Cinv_bar(1, 1);
      Cinv_barvct(2) = Cinv_bar(2, 2);
      Cinv_barvct(3) = Cinv_bar(0, 1);
      Cinv_barvct(4) = Cinv_bar(1, 2);
      Cinv_barvct(5) = Cinv_bar(2, 0);

      // Green-Lagrange strains(F_bar) matrix E = 0.5 . (Cauchygreen(F_bar) - Identity)
      // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
      Core::LinAlg::SerialDenseVector glstrain_bar_epetra(numstr_);
      Core::LinAlg::Matrix<numstr_, 1> glstrain_bar(glstrain_bar_epetra.values(), true);
      glstrain_bar(0) = 0.5 * (cauchygreen_bar(0, 0) - 1.0);
      glstrain_bar(1) = 0.5 * (cauchygreen_bar(1, 1) - 1.0);
      glstrain_bar(2) = 0.5 * (cauchygreen_bar(2, 2) - 1.0);
      glstrain_bar(3) = cauchygreen_bar(0, 1);  // Voigt notation: 2*C_12
      glstrain_bar(4) = cauchygreen_bar(1, 2);
      glstrain_bar(5) = cauchygreen_bar(2, 0);

      // calculate linear B-operator (WITHOUT F-bar modification)
      Core::LinAlg::Matrix<numstr_, numdofperelement_> boplin(false);
      calculate_boplin(&boplin, &N_XYZ);

      // calculate nonlinear B-operator (WITHOUT F-bar modification)
      Core::LinAlg::Matrix<numstr_, numdofperelement_> bop(false);
      calculate_bop(&bop, &defgrd, &N_XYZ);

      // ----------------------------------------- initialise temperature
      // described as a matrix (for stress calculation): Ntemp = N_T . T
      Core::LinAlg::Matrix<1, 1> NT(false);
      NT.MultiplyTN(shapefunct, etemp);
      // scalar-valued current element temperature T_{n+1}
      // temperature-dependent material parameters, i.e. E(T), pass T_{n+1}
      // insert T_{n+1} into parameter list
      params.set<double>("scalartemp", NT(0, 0));

      // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccc

      // calculate the stress part dependent on the temperature in the material
      Core::LinAlg::Matrix<numstr_, 1> ctemp_bar(true);
      Core::LinAlg::Matrix<numstr_, 1> couplstress_bar(true);   // S_bar_{vol,T}
      Core::LinAlg::Matrix<numstr_, numstr_> cmat_T_bar(true);  // dC_T_dE
      Core::LinAlg::Matrix<numstr_, 1> glstrain(true);

      // insert strain increment into parameter list which is only required for robinson
      // calculate iterative strains
      Core::LinAlg::Matrix<numstr_, 1> straininc(true);
      params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("straininc", straininc);
      // insert matrices into parameter list which are only required for thrplasthyperelast
      params.set<Core::LinAlg::Matrix<nsd_, nsd_>>("defgrd", defgrd_bar);
      params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("Cinv_vct", Cinv_barvct);
      // take care: current temperature (N . T) is passed to the element
      //            in the material: 1.) Delta T = subtract (N . T - T_0)
      //                             2.) couplstress = C . Delta T
      // do not call the material for Robinson's material
      if (material()->MaterialType() != Core::Materials::m_vp_robinson)
        materialize(&couplstress_bar,
            &ctemp_bar,  // is not filled! pass an empty matrix
            &NT, &cmat_T_bar, &glstrain_bar, params);

      // end of call material law ccccccccccccccccccccccccccccccccccccccccccccc

      // return gp stresses
      switch (iostress)
      {
        case Inpar::STR::stress_2pk:
        {
          if (elestress == nullptr) FOUR_C_THROW("stress data not available");
          for (int i = 0; i < numstr_; ++i) (*elestress)(gp, i) = couplstress_bar(i);
          break;
        }
        case Inpar::STR::stress_cauchy:
        {
          if (elestress == nullptr) FOUR_C_THROW("stress data not available");
          // push forward of material stress to the spatial configuration
          // sigma = 1/J . F . S_temp . F^T
          Core::LinAlg::Matrix<nsd_, nsd_> cauchycouplstress_bar;
          p_k2to_cauchy(&couplstress_bar, &defgrd_bar, &cauchycouplstress_bar);

          (*elestress)(gp, 0) = cauchycouplstress_bar(0, 0);
          (*elestress)(gp, 1) = cauchycouplstress_bar(1, 1);
          (*elestress)(gp, 2) = cauchycouplstress_bar(2, 2);
          (*elestress)(gp, 3) = cauchycouplstress_bar(0, 1);
          (*elestress)(gp, 4) = cauchycouplstress_bar(1, 2);
          (*elestress)(gp, 5) = cauchycouplstress_bar(0, 2);
          break;
        }
        case Inpar::STR::stress_none:
          break;
        default:
          FOUR_C_THROW("requested stress type not available");
          break;
      }

      // integrate internal force vector r_d
      // f = f + (B^T . sigma_temp) . detJ_bar . w(gp)
      // with detJ_bar = detJ/f_bar_factor
      double detJ_w = detJ * intpoints_.Weight(gp);
      // update internal force vector
      if (force != nullptr)
      {
        // integrate internal force vector f = f + (B^T . sigma) . detJ_bar . w(gp)
        force->MultiplyTN(detJ_w / f_bar_factor, bop, couplstress_bar, 1.0);
      }  // (force != nullptr)

      // update stiffness matrix k_dd
      if (stiffmatrix != nullptr)
      {
        // --------------------------------------------------------------
        // integrate temperature-dependent `elastic' and `initial-displacement'
        // stiffness matrix
        // --------------------------------------------------------------

        // keu = keu + (detF_0/detF)^(-1/3) .(B^T . Cmat_T . B) . detJ_bar . w(gp)
        // Neo-Hookean type: Cmat_T = m_0 . Delta T . (-1) . ( Cinv boeppel Cinv )_{abcd}
        // St.Venant Kirchhoff: dC_T/dd == 0
        // with dCinv/dC = ( Cinv boeppel Cinv )_{abcd} = 1/2 * ( Cinv_{ac} Cinv_{bd} + Cinv_{ad}
        // Cinv_{bc} )
        Core::LinAlg::Matrix<numstr_, numdofperelement_> cb(false);
        cb.Multiply(cmat_T_bar, bop);
        stiffmatrix->MultiplyTN((detJ_w * f_bar_factor), bop, cb, 1.0);

        // --------------------------------------------------------------
        // integrate `geometric' stiffness matrix and add to keu
        // --------------------------------------------------------------

        // kgeo += ( B_L^T . B_L . sigma_T) . detJ_bar . w(gp)
        // (B_L^T . sigma_T . B_L) = (24x6)(6x1)(6x24)
        // --> size of matrices do not fit --> multiply component-by-component
        // with linear B-operator B_L = Ni,Xj, see NiliFEM-Skript (6.20)

        Core::LinAlg::Matrix<numstr_, 1> sfac(couplstress_bar);  // auxiliary integrated stress
        // detJ_bar . w(gp) . [S11,S22,S33,S12=S21,S23=S32,S13=S31]
        sfac.Scale(detJ_w / f_bar_factor);
        // intermediate sigma_temp . B_L (6x1).(6x24)
        std::vector<double> StempB_L(3);
        for (int inod = 0; inod < nen_; ++inod)
        {
          // (3x1) = (6x1) (6x24)
          // S11 * N_XYZ(1,i) + S23 * N_XYZ(2,i) + S12 * N_XYZ(3,i)
          StempB_L[0] =
              sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
          // S23 * N_XYZ(1,i) + S22 * N_XYZ(2,i) + S13 * N_XYZ(3,i)
          StempB_L[1] =
              sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
          // S12 * N_XYZ(1,i) + S13 * N_XYZ(2,i) + S33 * N_XYZ(3,i)
          StempB_L[2] =
              sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
          // (B_L^T . sigma . B_L) = (24x6)(6x24)
          for (int jnod = 0; jnod < nen_; ++jnod)
          {
            double bopstrbop = 0.0;  // intermediate value
            for (int idim = 0; idim < nsd_; ++idim)
            {
              // double     (3x8)                 (3x1)
              bopstrbop += N_XYZ(idim, jnod) * StempB_L[idim];
            }
            // (24x24)
            (*stiffmatrix)(3 * inod + 0, 3 * jnod + 0) += bopstrbop;
            (*stiffmatrix)(3 * inod + 1, 3 * jnod + 1) += bopstrbop;
            (*stiffmatrix)(3 * inod + 2, 3 * jnod + 2) += bopstrbop;
          }
        }  // end of integrate `geometric' stiffness

        // ----------------------- linearisation of f_bar_factor w.r.t. d
        // k_dd = -1/3 . (detF_0/detF)^{-1/3} . B^T . sigmaT_bar . H^T

        // integrate additional fbar matrix including Voigt-notation!
        Core::LinAlg::Matrix<numstr_, 1> cauchygreenvct(false);
        cauchygreenvct(0) = cauchygreen(0, 0);
        cauchygreenvct(1) = cauchygreen(1, 1);
        cauchygreenvct(2) = cauchygreen(2, 2);
        cauchygreenvct(3) = 2 * cauchygreen(0, 1);
        cauchygreenvct(4) = 2 * cauchygreen(1, 2);
        cauchygreenvct(5) = 2 * cauchygreen(2, 0);

        Core::LinAlg::Matrix<numstr_, 1> ccg(false);
        ccg.Multiply(cmat_T_bar, cauchygreenvct);  // (6x1) = (6x6)(6x1)

        Core::LinAlg::Matrix<numdofperelement_, 1> bopccg(
            true);  // auxiliary integrated stress (24x1)
        bopccg.MultiplyTN((detJ_w * f_bar_factor / 3.0), bop, ccg);  // (24x1) = (24x6)(6x1)

        // calculate the auxiliary tensor H (24x1)
        // with H_i = tr(F_0^{-1} . dF_0/dd_i) - tr (F^{-1} . dF/dd_i)
        double htensor[numdofperelement_];
        for (int n = 0; n < numdofperelement_; n++)
        {
          htensor[n] = 0;
          for (int i = 0; i < nsd_; i++)
          {
            htensor[n] +=
                invdefgrd_0(i, n % 3) * N_XYZ_0(i, n / 3) - invdefgrd(i, n % 3) * N_XYZ(i, n / 3);
          }
        }
        Core::LinAlg::Matrix<numdofperelement_, 1> bops(false);  // auxiliary integrated stress
        bops.MultiplyTN((-detJ_w / f_bar_factor / 3.0), bop, couplstress_bar);
        for (int i = 0; i < numdofperelement_; i++)
        {
          for (int j = 0; j < numdofperelement_; j++)
          {
            // (24x24)             (1x24)                (24x1)
            (*stiffmatrix)(i, j) += htensor[j] * (bops(i, 0) + bopccg(i, 0));
          }
        }  // end of integrate additional `fbar' stiffness

      }  // if (stiffmatrix != nullptr), fill k_dd

      /* =====================================================================*/
    } /* ================================================ end of Loop over GP */
    /* =====================================================================*/

  }  // end HEX8FBAR
  else
    FOUR_C_THROW("call method only for HEX8FBAR elements!");

  return;

}  // nln_stiffint_tsi_fbar()


/*----------------------------------------------------------------------*
 | evaluate only the mechanical-thermal stiffness term       dano 05/13 |
 | for monolithic TSI, contribution to k_dT (protected)                 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::nln_kd_t_tsi_fbar(
    Core::Elements::Element::LocationArray& la,
    std::vector<double>& disp,                                       // current displacement
    std::vector<double>& temp,                                       // current temperature
    Core::LinAlg::Matrix<numdofperelement_, nen_>* stiffmatrix_kdT,  // (nsd_*nen_ x nen_)
    Teuchos::ParameterList& params)
{
  // in case we have a finite strain thermoplastic material use hex8fbar element
  // to cirucumvent volumetric locking
  Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>* eleFBAR =
      dynamic_cast<
          Discret::ELEMENTS::So3Thermo<Discret::ELEMENTS::SoHex8fbar, Core::FE::CellType::hex8>*>(
          this);

  if ((distype == Core::FE::CellType::hex8) && (eleFBAR))
  {
    // update element geometry (8x3)
    Core::LinAlg::Matrix<nen_, nsd_> xrefe(false);  // X, material coord. of element
    Core::LinAlg::Matrix<nen_, nsd_> xcurr(false);  // x, current  coord. of element
    Core::LinAlg::Matrix<nen_, 1> etemp(true);
    Core::Nodes::Node** nodes = Nodes();
    for (int i = 0; i < nen_; ++i)
    {
      const auto& x = nodes[i]->X();
      xrefe(i, 0) = x[0];
      xrefe(i, 1) = x[1];
      xrefe(i, 2) = x[2];

      xcurr(i, 0) = xrefe(i, 0) + disp[i * numdofpernode_ + 0];
      xcurr(i, 1) = xrefe(i, 1) + disp[i * numdofpernode_ + 1];
      xcurr(i, 2) = xrefe(i, 2) + disp[i * numdofpernode_ + 2];

      etemp(i, 0) = temp[i];
    }

    // shape functions and their first derivatives
    Core::LinAlg::Matrix<nen_, 1> shapefunct(false);
    Core::LinAlg::Matrix<nsd_, nen_> deriv(false);
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 . N_rst
    Core::LinAlg::Matrix<nsd_, nen_> N_XYZ(false);
    // build deformation gradient w.r.t. to material configuration
    Core::LinAlg::Matrix<nsd_, nsd_> defgrd(false);

    // ---------------------------- deformation gradient at centroid of element
    double detF_0 = -1.0;
    Core::LinAlg::Matrix<nsd_, nen_> N_XYZ_0(false);
    // element coordinate derivatives at centroid
    Core::LinAlg::Matrix<nsd_, nen_> N_rst_0(false);
    Core::FE::shape_function_3D_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);

    // inverse jacobian matrix at centroid
    Core::LinAlg::Matrix<nsd_, nsd_> invJ_0(false);
    invJ_0.Multiply(N_rst_0, xrefe);
    invJ_0.Invert();
    // material derivatives at centroid
    N_XYZ_0.Multiply(invJ_0, N_rst_0);

    // deformation gradient and its determinant at centroid
    Core::LinAlg::Matrix<3, 3> defgrd_0(false);
    defgrd_0.MultiplyTT(xcurr, N_XYZ_0);
    detF_0 = defgrd_0.Determinant();

    /* =========================================================================*/
    /* ================================================= Loop over Gauss Points */
    /* =========================================================================*/
    for (int gp = 0; gp < numgpt_; ++gp)
    {
      // shape functions (shapefunct) and their first derivatives (deriv)
      Core::FE::shape_function<distype>(xsi_[gp], shapefunct);
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);

      // copy structural shape functions needed for the thermo field
      // identical shapefunctions for the displacements and the temperatures
      Core::LinAlg::Matrix<1, 1> NT(false);
      NT.MultiplyTN(shapefunct, etemp);  // (1x1)
      // scalar-valued current element temperature T_{n+1}
      // temperature-dependent material parameters, i.e. E(T), pass T_{n+1}
      // insert T_{n+1} into parameter list
      params.set<double>("scalartemp", NT(0, 0));

      /* get the inverse of the Jacobian matrix which looks like:
      **            [ x_,r  y_,r  z_,r ]^-1
      **     J^-1 = [ x_,s  y_,s  z_,s ]
      **            [ x_,t  y_,t  z_,t ]
      */
      // compute derivatives N_XYZ at gp w.r.t. material coordinates
      // by N_XYZ = J^-1 . N_rst
      N_XYZ.Multiply(invJ_[gp], deriv);  // (6.21)
      double detJ = detJ_[gp];           // (6.22)

      // (material) deformation gradient
      // F = d xcurr / d xrefe = xcurr^T . N_XYZ^T
      defgrd.MultiplyTT(xcurr, N_XYZ);
      double detF = defgrd.Determinant();

      // -------------------------------------------- F_bar modifications
      // F_bar deformation gradient: F_bar := (detF_0 / detF)^1/3 . F
      Core::LinAlg::Matrix<nsd_, nsd_> defgrd_bar(defgrd);
      // f_bar_factor := (detF_0/detF)^{1/3}
      double f_bar_factor = std::pow(detF_0 / detF, 1.0 / 3.0);
      defgrd_bar.Scale(f_bar_factor);

      // Right Cauchy-Green tensor(Fbar) = F_bar^T . F_bar
      Core::LinAlg::Matrix<nsd_, nsd_> cauchygreen_bar(false);
      cauchygreen_bar.MultiplyTN(defgrd_bar, defgrd_bar);

      // inverse of Right Cauchy-Green tensor(Fbar) = F_bar^{-1} . F_bar^{-T}
      Core::LinAlg::Matrix<nsd_, nsd_> Cinv_bar(false);
      Cinv_bar.Invert(cauchygreen_bar);
      Core::LinAlg::Matrix<numstr_, 1> Cinv_barvct(false);
      Core::LinAlg::Voigt::Strains::MatrixToVector(Cinv_bar, Cinv_barvct);

      // calculate nonlinear B-operator
      Core::LinAlg::Matrix<numstr_, numdofperelement_> bop(false);
      calculate_bop(&bop, &defgrd, &N_XYZ);

      // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccc
      // get the thermal material tangent
      Core::LinAlg::Matrix<numstr_, 1> ctemp(true);
      Core::LinAlg::Matrix<numstr_, 1> Cmat_kdT(true);

      // --------------------------------------------------------------
      // in case of thermo-elasto-plastic material we get additional terms due
      // to temperature-dependence of the plastic multiplier Dgamma
      if (material()->MaterialType() == Core::Materials::m_thermoplhyperelast)
      {
        params.set<Core::LinAlg::Matrix<nsd_, nsd_>>("defgrd", defgrd_bar);
        params.set<Core::LinAlg::Matrix<numstr_, 1>>("Cinv_vct", Cinv_barvct);

        Teuchos::RCP<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
            Teuchos::rcp_dynamic_cast<Mat::ThermoPlasticHyperElast>(material(), true);
        // dCmat_dT = F^{-1} . 1/Dt . ds_{n+1}/dT_{n+1} . F^{-T}
        //          = - 2 . mubar . 1/Dt . dDgamma/dT . N_bar
        // with dDgamma/dT= - sqrt(2/3) . dsigma_y(astrain_p^{n+1},T_{n+1})/dT
        //                  . 1/(2 . mubar . beta0)
        Cmat_kdT.Update(thermoplhyperelast->CMat_kdT(gp));
      }
      Teuchos::RCP<Mat::Trait::ThermoSolid> thermoSolidMaterial =
          Teuchos::rcp_dynamic_cast<Mat::Trait::ThermoSolid>(material(), false);
      if (thermoSolidMaterial != Teuchos::null)
      {
        Core::LinAlg::Matrix<numstr_, 1> glstrain(false);
        glstrain(0) = 0.5 * (cauchygreen_bar(0, 0) - 1.0);
        glstrain(1) = 0.5 * (cauchygreen_bar(1, 1) - 1.0);
        glstrain(2) = 0.5 * (cauchygreen_bar(2, 2) - 1.0);
        glstrain(3) = cauchygreen_bar(0, 1);
        glstrain(4) = cauchygreen_bar(1, 2);
        glstrain(5) = cauchygreen_bar(2, 0);

        thermoSolidMaterial->Reinit(nullptr, &glstrain, NT(0), map_my_gp_to_so_hex8(gp));
        // full thermal derivative of stress wrt to scalar temperature (needs to be post-multiplied
        // with shape functions)
        thermoSolidMaterial->GetdSdT(&ctemp);
      }
      else
        // get temperature-dependent material tangent
        // in case of m_thermoplhyperelast: F, Cinv are passed via params
        compute_ctemp(&ctemp, params);

      // end of call material law ccccccccccccccccccccccccccccccccccccccccccccc

      double detJ_w = detJ * intpoints_.Weight(gp);
      // update linear coupling matrix K_dT
      if (stiffmatrix_kdT != nullptr)
      {
        // C_temp . N_temp
        Core::LinAlg::Matrix<numstr_, nen_> cn(false);
        cn.MultiplyNT(ctemp, shapefunct);  // (6x8)=(6x1)(1x8)
        // integrate stiffness term
        // k_dT = k_dT + (detF_0/detF)^{-1/3} (B^T . C_T . N_T) . detJ . w(gp)
        stiffmatrix_kdT->MultiplyTN((detJ_w / f_bar_factor), bop, cn, 1.0);

        if (material()->MaterialType() == Core::Materials::m_thermoplhyperelast)
        {
          // k_dT = k_dT + (detF_0/detF)^{-1/3} (B^T . dCmat/dT . N_temp) . detJ . w(gp)
          // (24x8)                            (24x6)         (6x1)    (1x8)
          Core::LinAlg::Matrix<numstr_, nen_> cmatn(false);
          cmatn.MultiplyNT(Cmat_kdT, shapefunct);  // (6x8)=(6x1)(1x8)
          stiffmatrix_kdT->MultiplyTN((detJ_w / f_bar_factor), bop, cmatn, 1.0);

          // Be careful: scaling with time factor is done in tsi_monolithic!!
        }
      }  // (stiffmatrix_kdT != nullptr)

      /* =====================================================================*/
    } /* ================================================ end of Loop over GP */
    /* =====================================================================*/

  }  // end HEX8FBAR
  else
    FOUR_C_THROW("call method only for HEX8FBAR elements!");
  return;

}  // nln_kd_t_tsi_fbar()


/*----------------------------------------------------------------------*
 | material law with temperature part for So3_thermo         dano 05/10 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::materialize(
    Core::LinAlg::Matrix<numstr_, 1>* couplstress,  // temperature-dependent stress part
    Core::LinAlg::Matrix<numstr_, 1>* ctemp,        // temperature-dependent material tangent
    Core::LinAlg::Matrix<1, 1>* Ntemp,              // temperature of element
    Core::LinAlg::Matrix<numstr_, numstr_>* cmat,   // (mechanical) material tangent
    Core::LinAlg::Matrix<numstr_, 1>* glstrain,     // Green-Lagrange strain tensor
    Teuchos::ParameterList& params                  // parameter
)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (!couplstress) FOUR_C_THROW("No stress vector supplied");
#endif

  // backwards compatibility: not all materials use the new interface
  Teuchos::RCP<Mat::Trait::ThermoSolid> thermoSolidMaterial =
      Teuchos::rcp_dynamic_cast<Mat::Trait::ThermoSolid>(material(), false);
  if (thermoSolidMaterial != Teuchos::null)
  {
    // new interface already includes temperature terms in stress evaluation
    // no additive splitting into stress and couplstress -> nothing to do here
    return;
  }

  // All materials that have a pure Core::LinAlg::Matrix
  // interface go to the material law here.
  // the old interface does not exist anymore...
  Teuchos::RCP<Core::Mat::Material> mat = material();
  switch (mat->MaterialType())
  {
    // small strain von Mises thermoelastoplastic material
    case Core::Materials::m_thermopllinelast:
    {
      Teuchos::RCP<Mat::ThermoPlasticLinElast> thrpllinelast =
          Teuchos::rcp_dynamic_cast<Mat::ThermoPlasticLinElast>(material(), true);
      thrpllinelast->Evaluate(*Ntemp, *ctemp, *couplstress);
      return;
      break;
    }
    // visco-plastic Robinson's material
    case Core::Materials::m_vp_robinson:
    {
      // no temperature-dependent terms
      return;
      break;
    }
    // thermo-hyperelasto-plastic material
    case Core::Materials::m_thermoplhyperelast:
    {
      Teuchos::RCP<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
          Teuchos::rcp_dynamic_cast<Mat::ThermoPlasticHyperElast>(material(), true);
      thermoplhyperelast->Evaluate(*Ntemp, *ctemp, *cmat, *couplstress, params);
      return;
      break;
    }

    default:
      FOUR_C_THROW("Unknown type of temperature dependent material");
      break;
  }  // switch (mat->MaterialType())

  return;
}  // Materialize()


/*----------------------------------------------------------------------*
 | get the constant temperature fraction for couplstress      dano 05/10 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::compute_ctemp(
    Core::LinAlg::Matrix<numstr_, 1>* ctemp, Teuchos::ParameterList& params)
{
  switch (material()->MaterialType())
  {
    // thermo st.venant-kirchhoff-material
    case Core::Materials::m_thermostvenant:
    {
      FOUR_C_THROW("Ctemp call no longer valid for ThermoStVenantKirchhoff. Fix implementation.");
    }
    // small strain von Mises thermoelastoplastic material
    case Core::Materials::m_thermopllinelast:
    {
      Teuchos::RCP<Mat::ThermoPlasticLinElast> thrpllinelast =
          Teuchos::rcp_dynamic_cast<Mat::ThermoPlasticLinElast>(material(), true);
      return thrpllinelast->setup_cthermo(*ctemp);
      break;
    }
    // visco-plastic Robinson's material
    case Core::Materials::m_vp_robinson:
    {
      // so far: do nothing, because the displacement-dependent coupling term
      // is neglected
      return;
      break;
    }
    // thermo-hyperelasto-plastic material
    case Core::Materials::m_thermoplhyperelast:
    {
      Teuchos::RCP<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
          Teuchos::rcp_dynamic_cast<Mat::ThermoPlasticHyperElast>(material(), true);
      thermoplhyperelast->setup_cthermo(*ctemp, params);
      return;
      break;
    }
    default:
      FOUR_C_THROW("Cannot ask material for the temperature-dependent material tangent");
      break;
  }  // switch (mat->MaterialType())

}  // Ctemp()


/*----------------------------------------------------------------------*
 | calculate the nonlinear B-operator                        dano 11/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::calculate_bop(
    Core::LinAlg::Matrix<numstr_, numdofperelement_>* bop, Core::LinAlg::Matrix<nsd_, nsd_>* defgrd,
    Core::LinAlg::Matrix<nsd_, nen_>* N_XYZ)
{
  // lump mass matrix
  if (bop != nullptr)
  {
    /* non-linear B-operator (may so be called, meaning of B-operator is not so
    **  sharp in the non-linear realm) *
    **   B = F^{i,T} . B_L *
    ** with linear B-operator B_L =  N_XYZ (6x24) = (3x8)
    **
    **   B    =   F  . N_XYZ
    ** (6x24)   (3x3) (3x8)
    **
    **      [ ... | F_11*N_{,1}^k  F_21*N_{,1}^k  F_31*N_{,1}^k | ... ]
    **      [ ... | F_12*N_{,2}^k  F_22*N_{,2}^k  F_32*N_{,2}^k | ... ]
    **      [ ... | F_13*N_{,3}^k  F_23*N_{,3}^k  F_33*N_{,3}^k | ... ]
    ** B =  [ ~~~   ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~   ~~~ ]
    **      [       F_11*N_{,2}^k+F_12*N_{,1}^k                       ]
    **      [ ... |          F_21*N_{,2}^k+F_22*N_{,1}^k        | ... ]
    **      [                       F_31*N_{,2}^k+F_32*N_{,1}^k       ]
    **      [                                                         ]
    **      [       F_12*N_{,3}^k+F_13*N_{,2}^k                       ]
    **      [ ... |          F_22*N_{,3}^k+F_23*N_{,2}^k        | ... ]
    **      [                       F_32*N_{,3}^k+F_33*N_{,2}^k       ]
    **      [                                                         ]
    **      [       F_13*N_{,1}^k+F_11*N_{,3}^k                       ]
    **      [ ... |          F_23*N_{,1}^k+F_21*N_{,3}^k        | ... ]
    **      [                       F_33*N_{,1}^k+F_31*N_{,3}^k       ]
    */
    for (int i = 0; i < nen_; ++i)
    {
      (*bop)(0, numdofpernode_ * i + 0) = (*defgrd)(0, 0) * (*N_XYZ)(0, i);
      (*bop)(0, numdofpernode_ * i + 1) = (*defgrd)(1, 0) * (*N_XYZ)(0, i);
      (*bop)(0, numdofpernode_ * i + 2) = (*defgrd)(2, 0) * (*N_XYZ)(0, i);
      (*bop)(1, numdofpernode_ * i + 0) = (*defgrd)(0, 1) * (*N_XYZ)(1, i);
      (*bop)(1, numdofpernode_ * i + 1) = (*defgrd)(1, 1) * (*N_XYZ)(1, i);
      (*bop)(1, numdofpernode_ * i + 2) = (*defgrd)(2, 1) * (*N_XYZ)(1, i);
      (*bop)(2, numdofpernode_ * i + 0) = (*defgrd)(0, 2) * (*N_XYZ)(2, i);
      (*bop)(2, numdofpernode_ * i + 1) = (*defgrd)(1, 2) * (*N_XYZ)(2, i);
      (*bop)(2, numdofpernode_ * i + 2) = (*defgrd)(2, 2) * (*N_XYZ)(2, i);
      /* ~~~ */
      (*bop)(3, numdofpernode_ * i + 0) =
          (*defgrd)(0, 0) * (*N_XYZ)(1, i) + (*defgrd)(0, 1) * (*N_XYZ)(0, i);
      (*bop)(3, numdofpernode_ * i + 1) =
          (*defgrd)(1, 0) * (*N_XYZ)(1, i) + (*defgrd)(1, 1) * (*N_XYZ)(0, i);
      (*bop)(3, numdofpernode_ * i + 2) =
          (*defgrd)(2, 0) * (*N_XYZ)(1, i) + (*defgrd)(2, 1) * (*N_XYZ)(0, i);
      (*bop)(4, numdofpernode_ * i + 0) =
          (*defgrd)(0, 1) * (*N_XYZ)(2, i) + (*defgrd)(0, 2) * (*N_XYZ)(1, i);
      (*bop)(4, numdofpernode_ * i + 1) =
          (*defgrd)(1, 1) * (*N_XYZ)(2, i) + (*defgrd)(1, 2) * (*N_XYZ)(1, i);
      (*bop)(4, numdofpernode_ * i + 2) =
          (*defgrd)(2, 1) * (*N_XYZ)(2, i) + (*defgrd)(2, 2) * (*N_XYZ)(1, i);
      (*bop)(5, numdofpernode_ * i + 0) =
          (*defgrd)(0, 2) * (*N_XYZ)(0, i) + (*defgrd)(0, 0) * (*N_XYZ)(2, i);
      (*bop)(5, numdofpernode_ * i + 1) =
          (*defgrd)(1, 2) * (*N_XYZ)(0, i) + (*defgrd)(1, 0) * (*N_XYZ)(2, i);
      (*bop)(5, numdofpernode_ * i + 2) =
          (*defgrd)(2, 2) * (*N_XYZ)(0, i) + (*defgrd)(2, 0) * (*N_XYZ)(2, i);
    }
  }
}  // calculate_bop()


/*----------------------------------------------------------------------*
 | calculate the nonlinear B-operator in vector notation     dano 11/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::calculate_bop_vec(
    Core::LinAlg::Matrix<1, numdofperelement_>& bopvec, Core::LinAlg::Matrix<nsd_, nsd_>& defgrd,
    Core::LinAlg::Matrix<nsd_, nen_>& N_XYZ)
{
  // ---------------------------------------------- build F as vector 9x1
  // F != F^T, i.e. Voigt notation NOT admissible
  // F (3x3) --> (9x1)
  Core::LinAlg::Matrix<nsd_ * nsd_, 1> defgrd_vec(false);
  defgrd_vec(0) = defgrd(0, 0);
  defgrd_vec(1) = defgrd(0, 1);
  defgrd_vec(2) = defgrd(0, 2);
  defgrd_vec(3) = defgrd(1, 0);
  defgrd_vec(4) = defgrd(1, 1);
  defgrd_vec(5) = defgrd(1, 2);
  defgrd_vec(6) = defgrd(2, 0);
  defgrd_vec(7) = defgrd(2, 1);
  defgrd_vec(8) = defgrd(2, 2);

  // ------------------------ build N_X operator (w.r.t. material config)
  // N_XYZ (3x8) --> 9x24
  Core::LinAlg::Matrix<9, numdofperelement_> N_X(true);  // set to zero
  for (int i = 0; i < nen_; ++i)
  {
    N_X(0, 3 * i + 0) = N_XYZ(0, i);
    N_X(1, 3 * i + 1) = N_XYZ(0, i);
    N_X(2, 3 * i + 2) = N_XYZ(0, i);

    N_X(3, 3 * i + 0) = N_XYZ(1, i);
    N_X(4, 3 * i + 1) = N_XYZ(1, i);
    N_X(5, 3 * i + 2) = N_XYZ(1, i);

    N_X(6, 3 * i + 0) = N_XYZ(2, i);
    N_X(7, 3 * i + 1) = N_XYZ(2, i);
    N_X(8, 3 * i + 2) = N_XYZ(2, i);
  }
  bopvec.MultiplyTN(1.0, defgrd_vec, N_X);

}  // Calculate (1x24) B-Operator


/*----------------------------------------------------------------------*
 | calculate the linear B-operator                           dano 11/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::calculate_boplin(
    Core::LinAlg::Matrix<numstr_, numdofperelement_>* boplin,
    Core::LinAlg::Matrix<nsd_, nen_>* N_XYZ)
{
  // lump mass matrix
  if (boplin != nullptr)
  {
    // linear B-operator B = N_XYZ
    // disperse global derivatives to bop-lines
    // bop is arranged as usual (refer to script FE or elsewhere):
    // [ N1,X  0  0  | N2,X  0  0  | ... | Ni,X  0  0  ]
    // [ 0  N1,Y  0  | 0  N2,Y  0  | ... | 0  Ni,Y  0  ]
    // [ 0  0  N1,Z  | 0  0  N2,Z  | ... | 0  0  Ni,Z  ]
    // [ N1,Y N1,X 0 | N2,Y N2,X 0 | ... | Ni,Y Ni,X 0 ]
    // [ 0 N1,Z N1,Y | 0 N2,Z N2,Y | ... | 0 Ni,Z Ni,Y ]
    // [ N1,Z 0 N1,X | N2,Z 0 N2,X | ... | Ni,Z 0 Ni,X ]
    for (int i = 0; i < nen_; ++i)
    {
      (*boplin)(0, numdofpernode_ * i + 0) = (*N_XYZ)(0, i);
      (*boplin)(0, numdofpernode_ * i + 1) = 0.0;
      (*boplin)(0, numdofpernode_ * i + 2) = 0.0;
      (*boplin)(1, numdofpernode_ * i + 0) = 0.0;
      (*boplin)(1, numdofpernode_ * i + 1) = (*N_XYZ)(1, i);
      (*boplin)(1, numdofpernode_ * i + 2) = 0.0;
      (*boplin)(2, numdofpernode_ * i + 0) = 0.0;
      (*boplin)(2, numdofpernode_ * i + 1) = 0.0;
      (*boplin)(2, numdofpernode_ * i + 2) = (*N_XYZ)(2, i);
      /* ~~~ */
      (*boplin)(3, numdofpernode_ * i + 0) = (*N_XYZ)(1, i);
      (*boplin)(3, numdofpernode_ * i + 1) = (*N_XYZ)(0, i);
      (*boplin)(3, numdofpernode_ * i + 2) = 0.0;
      (*boplin)(4, numdofpernode_ * i + 0) = 0.0;
      (*boplin)(4, numdofpernode_ * i + 1) = (*N_XYZ)(2, i);
      (*boplin)(4, numdofpernode_ * i + 2) = (*N_XYZ)(1, i);
      (*boplin)(5, numdofpernode_ * i + 0) = (*N_XYZ)(2, i);
      (*boplin)(5, numdofpernode_ * i + 1) = 0.0;
      (*boplin)(5, numdofpernode_ * i + 2) = (*N_XYZ)(0, i);
    }
  }
}  // calculate_boplin()


/*----------------------------------------------------------------------*
 | push forward of material to spatial stresses              dano 11/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::p_k2to_cauchy(
    Core::LinAlg::Matrix<numstr_, 1>* stress, Core::LinAlg::Matrix<nsd_, nsd_>* defgrd,
    Core::LinAlg::Matrix<nsd_, nsd_>* cauchystress)
{
  // calculate the Jacobi-deterinant
  const double detF = (*defgrd).Determinant();

  // sigma = 1/J . F . S . F^T
  Core::LinAlg::Matrix<nsd_, nsd_> pkstress(false);
  pkstress(0, 0) = (*stress)(0);
  pkstress(0, 1) = (*stress)(3);
  pkstress(0, 2) = (*stress)(5);
  pkstress(1, 0) = pkstress(0, 1);
  pkstress(1, 1) = (*stress)(1);
  pkstress(1, 2) = (*stress)(4);
  pkstress(2, 0) = pkstress(0, 2);
  pkstress(2, 1) = pkstress(1, 2);
  pkstress(2, 2) = (*stress)(2);

  Core::LinAlg::Matrix<nsd_, nsd_> temp(false);
  temp.Multiply((1.0 / detF), (*defgrd), pkstress);
  (*cauchystress).MultiplyNT(temp, (*defgrd));

}  // PK2toCauchy()


/*----------------------------------------------------------------------*
 | push forward of material to spatial stresses              dano 11/12 |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::g_lto_ea(
    Core::LinAlg::Matrix<numstr_, 1>* glstrain, Core::LinAlg::Matrix<nsd_, nsd_>* defgrd,
    Core::LinAlg::Matrix<nsd_, nsd_>* euler_almansi)
{
  // e = F^{T-1} . E . F^{-1}

  // rewrite Green-Lagrange strain in tensor notation
  Core::LinAlg::Matrix<nsd_, nsd_> gl;
  gl(0, 0) = (*glstrain)(0);
  gl(0, 1) = 0.5 * (*glstrain)(3);
  gl(0, 2) = 0.5 * (*glstrain)(5);
  gl(1, 0) = gl(0, 1);
  gl(1, 1) = (*glstrain)(1);
  gl(1, 2) = 0.5 * (*glstrain)(4);
  gl(2, 0) = gl(0, 2);
  gl(2, 1) = gl(1, 2);
  gl(2, 2) = (*glstrain)(2);

  // inverse of deformation gradient
  Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd;
  invdefgrd.Invert(*defgrd);

  // (3x3) = (3x3) (3x3) (3x3)
  Core::LinAlg::Matrix<nsd_, nsd_> temp(false);
  temp.Multiply(gl, invdefgrd);
  (*euler_almansi).MultiplyTN(invdefgrd, temp);

}  // GLtoEA()


/*----------------------------------------------------------------------*
 | initialise Jacobian                                       dano 08/12 |
 | is called once in Initialize() in so3_thermo_eletypes.cpp            |
 *----------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
void Discret::ELEMENTS::So3Thermo<so3_ele, distype>::init_jacobian_mapping_special_for_nurbs(
    Core::FE::Discretization& dis)
{
  // get the material coordinates
  Core::LinAlg::Matrix<nen_, nsd_> xrefe;
  for (int i = 0; i < nen_; ++i)
  {
    Core::Nodes::Node** nodes = Nodes();
    if (!nodes) FOUR_C_THROW("Nodes() returned null pointer");
    xrefe(i, 0) = Nodes()[i]->X()[0];
    xrefe(i, 1) = Nodes()[i]->X()[1];
    xrefe(i, 2) = Nodes()[i]->X()[2];
  }
  invJ_.resize(numgpt_);
  detJ_.resize(numgpt_);
  xsi_.resize(numgpt_);

  // initialise the derivatives of the shape functions
  Core::LinAlg::Matrix<nsd_, nen_> deriv;
  Core::LinAlg::Matrix<nen_, 1> funct;

  // --------------------------------------------------
  // Initialisation of nurbs specific stuff
  std::vector<Core::LinAlg::SerialDenseVector> myknots(3);
  Core::LinAlg::Matrix<27, 1> weights;

  // get nurbs specific infos
  if (so3_ele::Shape() == Core::FE::CellType::nurbs27)
  {
    // cast to nurbs discretization
    Discret::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Discret::Nurbs::NurbsDiscretization*>(&(dis));
    if (nurbsdis == nullptr) FOUR_C_THROW("So_nurbs27 appeared in non-nurbs discretisation\n");

    // zero-sized element
    if ((*((*nurbsdis).GetKnotVector())).GetEleKnots(myknots, id())) return;

    // get weights from cp's
    for (int inode = 0; inode < nen_; inode++)
      weights(inode) = dynamic_cast<Discret::Nurbs::ControlPoint*>(Nodes()[inode])->W();
  }

  // coordinates of the current integration point (xsi_)
  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // get the coordinates of Gauss points, here use intrepid
    const double* gpcoord = intpoints_.Point(gp);
    for (int idim = 0; idim < nsd_; idim++)
    {
      xsi_[gp](idim) = gpcoord[idim];
    }
    // first derivatives of shape functions (deriv)
    if (so3_ele::Shape() != Core::FE::CellType::nurbs27)
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);
    else
      Core::FE::Nurbs::nurbs_get_3D_funct_deriv(
          funct, deriv, xsi_[gp], myknots, weights, Core::FE::CellType::nurbs27);

    // compute Jacobian matrix and determinant
    // actually compute its transpose....
    /*
      +-            -+ T      +-            -+
      | dx   dx   dx |        | dx   dy   dz |
      | --   --   -- |        | --   --   -- |
      | dr   ds   dt |        | dr   dr   dr |
      |              |        |              |
      | dy   dy   dy |        | dx   dy   dz |
      | --   --   -- |   =    | --   --   -- |
      | dr   ds   dt |        | ds   ds   ds |
      |              |        |              |
      | dz   dz   dz |        | dx   dy   dz |
      | --   --   -- |        | --   --   -- |
      | dr   ds   dt |        | dt   dt   dt |
      +-            -+        +-            -+
     */
    // derivatives of coordinates w.r.t. material coordinates xjm_ = dx/ds
    invJ_[gp].Multiply(deriv, xrefe);
    // xij_ = ds/dx
    detJ_[gp] = invJ_[gp].Invert();
    if (detJ_[gp] < 1.0E-16) FOUR_C_THROW("ZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", detJ_[gp]);
  }  // end gp loop

  return;
}  // init_jacobian_mapping()

/*----------------------------------------------------------------------------------*
 | map the GP ordering as defined in Intrepid to the So_hex8 ordering  proell 05/18 |
 *----------------------------------------------------------------------------------*/
template <class so3_ele, Core::FE::CellType distype>
int Discret::ELEMENTS::So3Thermo<so3_ele, distype>::map_my_gp_to_so_hex8(int myGp)
{
  switch (distype)
  {
    case Core::FE::CellType::hex8:
    {
      const std::array<int, NUMGPT_SOH8> hex8_map = {6, 7, 5, 4, 2, 3, 1, 0};
      return hex8_map[myGp];
    }
    default:
      return myGp;
  }
}


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

// --- explicit instantiations --- //
#include "4C_so3_thermo_fwd.hpp"
