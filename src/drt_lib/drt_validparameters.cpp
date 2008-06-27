/*----------------------------------------------------------------------*/
/*!
\file drt_validparameters.cpp

\brief Setup of the list of valid input parameters

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/

#ifdef CCADISCRET

#include <iostream>

#include <Teuchos_Array.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_StrUtils.hpp>
#include <Teuchos_any.hpp>

#include <AztecOO.h>


#include "drt_validparameters.H"
#include "drt_colors.H"

#ifdef PARALLEL
#include <mpi.h>
#endif

extern "C" /* stuff which is c and is accessed from c++ */
{
#include "../headers/standardtypes.h"
}


/*----------------------------------------------------------------------*/
//! Print function to be called from C
/*----------------------------------------------------------------------*/
extern "C"
void PrintValidParameters()
{
  Teuchos::RCP<const Teuchos::ParameterList> list = DRT::INPUT::ValidParameters();
  list->print(std::cout,
              Teuchos::ParameterList::PrintOptions()
              .showDoc(true)
              .showFlags(false)
              .indent(4)
              .showTypes(false));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::PrintDatHeader(std::ostream& stream, const Teuchos::ParameterList& list, std::string parentname, bool color)
{
  std::string blue2light = "";
  std::string bluelight = "";
  std::string redlight = "";
  std::string yellowlight = "";
  std::string greenlight = "";
  std::string magentalight = "";
  std::string endcolor = "";
  if (color)
  {
    blue2light = BLUE2_LIGHT;
    bluelight = BLUE_LIGHT;
    redlight = RED_LIGHT;
    yellowlight = YELLOW_LIGHT;
    greenlight = GREEN_LIGHT;
    magentalight = MAGENTA_LIGHT;
    endcolor = END_COLOR;
  }

  // prevent invalid ordering of parameters caused by alphabetical output:
  // in the first run, print out all list elements that are not a sublist
  // in the second run, do the recursive call for all the sublists in the list
  for (int j=0; j<2; ++j)
  {
    for (Teuchos::ParameterList::ConstIterator i = list.begin();
    i!=list.end();
    ++i)
    {
      const Teuchos::ParameterEntry& entry = list.entry(i);
      if (entry.isList() && j==0) continue;
      if ((!entry.isList()) && j==1) continue;
      const std::string &name = list.name(i);
      Teuchos::RCP<const Teuchos::ParameterEntryValidator> validator = entry.validator();

      stream << blue2light << "//" << endcolor << '\n';

      std::string doc = entry.docString();
      if (doc!="")
      {
        Teuchos::StrUtils::printLines(stream,blue2light + "// ",doc);
        stream << endcolor;
      }

      if (entry.isList())
      {
        std::string secname = parentname;
        if (secname!="")
          secname += "/";
        secname += name;
        unsigned l = secname.length();
        stream << redlight << "--";
        for (int i=0; i<std::max<int>(65-l,0); ++i) stream << '-';
        stream << greenlight << secname << endcolor << '\n';
        PrintDatHeader(stream,list.sublist(name),secname,color);
      }
      else
      {
        if (validator!=Teuchos::null)
        {
          Teuchos::RCP<const Teuchos::Array<std::string> > values = validator->validStringValues();
          if (values!=Teuchos::null)
          {
            unsigned len = 0;
            for (unsigned i=0; i<values->size(); ++i)
            {
              len += (*values)[i].length()+1;
            }
            if (len<74)
            {
              stream << blue2light << "//     ";
              for (int i=0; i<static_cast<int>(values->size())-1; ++i)
              {
                stream << magentalight << (*values)[i] << blue2light << ",";
              }
              stream << magentalight << (*values)[values->size()-1] << endcolor << '\n';
            }
            else
            {
              for (unsigned i=0; i<values->size(); ++i)
              {
                stream << blue2light << "//     " << magentalight << (*values)[i] << endcolor << '\n';
              }
            }
          }
        }
        const Teuchos::any& v = entry.getAny(false);
        stream << bluelight << name << endcolor;
        unsigned l = name.length();
        for (int i=0; i<std::max<int>(31-l,0); ++i) stream << ' ';
        stream << ' ' << yellowlight << v << endcolor << '\n';
      }
    }
  }
}


/*----------------------------------------------------------------------*/
//! Print function to be called from C
/*----------------------------------------------------------------------*/
extern "C"
void PrintDefaultDatHeader()
{
  Teuchos::RCP<const Teuchos::ParameterList> list = DRT::INPUT::ValidParameters();
  DRT::INPUT::PrintDatHeader(std::cout,*list,"",true);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::PrintDefaultParameters(std::ostream& stream, const Teuchos::ParameterList& list)
{
  bool hasDefault = false;
  for (Teuchos::ParameterList::ConstIterator i = list.begin();
       i!=list.end();
       ++i)
  {
    const Teuchos::ParameterEntry& entry = list.entry(i);
    if (entry.isDefault())
    {
      if (not hasDefault)
      {
        hasDefault = true;
        stream << "default parameters in list '" << list.name() << "':\n";
      }
      const Teuchos::any& v = entry.getAny(false);
      int l = list.name(i).length();
      stream << "    " << list.name(i);
      for (int i=0; i<std::max<int>(31-l,0); ++i) stream << ' ';
      stream << ' ' << v << '\n';
    }
  }
  if (hasDefault)
    stream << "\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::IntParameter(std::string const &paramName,
                              int const value,
                              std::string const &docString,
                              Teuchos::ParameterList *paramList)
{
  Teuchos::AnyNumberParameterEntryValidator::AcceptedTypes validator(false);
  validator.allowInt(true);
  Teuchos::setIntParameter(paramName,value,
                           docString,
                           paramList,validator);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::DoubleParameter(std::string const &paramName,
                                 double const &value,
                                 std::string const &docString,
                                 Teuchos::ParameterList *paramList)
{
  Teuchos::AnyNumberParameterEntryValidator::AcceptedTypes validator(false);
  validator.allowDouble(true);
  validator.allowInt(true);
  Teuchos::setDoubleParameter(paramName,value,
                              docString,
                              paramList,validator);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Teuchos::ParameterList> DRT::INPUT::ValidParameters()
{
  using Teuchos::tuple;
  using Teuchos::setStringToIntegralParameter;

  Teuchos::Array<std::string> yesnotuple = tuple<std::string>("Yes","No","yes","no","YES","NO");
  Teuchos::Array<int> yesnovalue = tuple<int>(true,false,true,false,true,false);

  Teuchos::RCP<Teuchos::ParameterList> list = Teuchos::rcp(new Teuchos::ParameterList);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& discret = list->sublist("DISCRETISATION",false,"");

  IntParameter("NUMFLUIDDIS",1,"Number of meshes in fluid field",&discret);
  IntParameter("NUMSTRUCDIS",1,"Number of meshes in structural field",&discret);
  IntParameter("NUMALEDIS",1,"Number of meshes in ale field",&discret);
  IntParameter("NUMTHERMDIS",1,"Number of meshes in thermal field",&discret);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& size = list->sublist("PROBLEM SIZE",false,"");

  IntParameter("ELEMENTS",0,"Total number of elements",&size);
  IntParameter("NODES",0,"Total number of nodes",&size);
  IntParameter("DIM",3,"2d or 3d problem",&size);
  IntParameter("MATERIALS",0,"number of materials",&size);
  IntParameter("NUMDF",3,"maximum number of degrees of freedom",&size);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& type = list->sublist("PROBLEM TYP",false,"");

  setStringToIntegralParameter("PROBLEMTYP","Fluid_Structure_Interaction","",
                               tuple<std::string>(
                                 "Structure",
                                 "Fluid",
                                 "Fluid_XFEM",
                                 "Fluid_Ale",
                                 "Fluid_Freesurface",
                                 "Convection_Diffusion",
                                 "Fluid_Structure_Interaction",
                                 "Fluid_Structure_Interaction_XFEM",
                                 "Ale",
                                 "Thermal_Structure_Interaction")
                               .append("Structure_Multiscale")
                               .append("Electrochemistry")
                               .append("Combustion"),
                               tuple<PROBLEM_TYP>(
                                 prb_structure,
                                 prb_fluid,
                                 prb_fluid_xfem,
                                 prb_fluid_ale,
                                 prb_freesurf,
                                 prb_condif,
                                 prb_fsi,
                                 prb_fsi_xfem,
                                 prb_ale,
                                 prb_tsi)
                               .append(prb_struct_multi)
                               .append(prb_elch)
                               .append(prb_combust),
                               &type);
  IntParameter("NUMFIELD",1,"",&type);
  setStringToIntegralParameter("TIMETYP","Dynamic","",
                               tuple<std::string>("Static","Dynamic"),
                               tuple<TIME_TYP>(time_static,time_dynamic),
                               &type);
  //IntParameter("GRADERW",0,"",&type);
  IntParameter("MULTISC_STRUCT",0,"",&type);
  IntParameter("RESTART",0,"",&type);
  setStringToIntegralParameter("ALGEBRA","Trilinos","outdated",
                               tuple<std::string>("Trilinos","ccarat"),
                               tuple<int>(1,0),
                               &type);
  setStringToIntegralParameter("SHAPEFCT","Polynomial","Defines the function spaces for the spatial approximation",
                               tuple<std::string>("Polynomial","Nurbs"),
                               tuple<int>(1,0),
                               &type);
  
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& io = list->sublist("IO",false,"");

  // are these needed?
  setStringToIntegralParameter("OUTPUT_OUT","No","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("OUTPUT_GID","No","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("OUTPUT_BIN","No","",yesnotuple,yesnovalue,&io);

  setStringToIntegralParameter("STRUCT_DISP","Yes","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("STRUCT_STRESS","No","",
                               tuple<std::string>("No","no","NO",
                                                  "Yes","yes","YES",
                                                  "Cauchy","cauchy",
                                                  "2PK", "2pk"),
                               tuple<STRUCT_STRESS_TYP>(struct_stress_none,struct_stress_none,struct_stress_none,
                                                        struct_stress_pk,struct_stress_pk,struct_stress_pk,
                                                        struct_stress_cauchy,struct_stress_cauchy,
                                                        struct_stress_pk,struct_stress_pk),
                               &io);
  setStringToIntegralParameter("STRUCT_STRAIN","No","",
                               tuple<std::string>("No","no","NO",
                                                  "Yes","yes","YES",
                                                  "EA","ea",
                                                  "GL", "gl"),
                               tuple<STRUCT_STRAIN_TYP>(struct_strain_none,struct_strain_none,struct_strain_none,
                                                        struct_strain_gl,struct_strain_gl,struct_strain_gl,
                                                        struct_strain_ea,struct_strain_ea,
                                                        struct_strain_gl,struct_strain_gl),
                               &io);
  setStringToIntegralParameter("STRUCT_SM_DISP","No","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("FLUID_SOL","Yes","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("FLUID_STRESS","No","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("FLUID_VIS","No","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("ALE_DISP","No","",yesnotuple,yesnovalue,&io);

  setStringToIntegralParameter("THERM_TEMPERATURE","No","",yesnotuple,yesnovalue,&io);
  setStringToIntegralParameter("THERM_HEATFLUX","No","",yesnotuple,yesnovalue,&io);

  IntParameter("FILESTEPS",1000,"",&io);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& design = list->sublist("DESIGN DESCRIPTION",false,"number of nodal clouds");

  IntParameter("NDPOINT",0,"number of points",&design);
  IntParameter("NDLINE",0,"number of line clouds",&design);
  IntParameter("NDSURF",0,"number of surface clouds",&design);
  IntParameter("NDVOL",0,"number of volume clouds",&design);

  /*----------------------------------------------------------------------*/
  // An empty list. The actual list is arbitrary and not validated.
  //Teuchos::ParameterList& condition =
  list->sublist("CONDITION NAMES",false,
                "Names of conditions from exodus file.\n"
                "This section is not validated, any variable is allowed here.\n"
                "The names defined in this section can be used by all conditions instead of\n"
                "a design object number. This section assigns the respective numbers to\n"
                "the names.");

  //ParameterList& stat = list->sublist("STATIC",false,"");

  /*----------------------------------------------------------------------*/
  //Teuchos::ParameterList& eigen = list->sublist("EIGENVALUE ANALYSIS",false,"");

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& sdyn = list->sublist("STRUCTURAL DYNAMIC",false,"");

  setStringToIntegralParameter("DYNAMICTYP","Gen_Alfa",
                               "type of time integration control",
                               tuple<std::string>("Centr_Diff",
                                                  "Gen_EMM",
                                                  "Gen_Alfa",
                                                  "GenAlpha"
                                                  "OneStepTheta"),
                               tuple<int>(STRUCT_DYNAMIC::centr_diff,
                                          STRUCT_DYNAMIC::Gen_EMM,
                                          STRUCT_DYNAMIC::gen_alfa,
                                          STRUCT_DYNAMIC::genalpha,
                                          STRUCT_DYNAMIC::onesteptheta),
                               &sdyn);
  // Output type
  IntParameter("EIGEN",0,"EIGEN make eigenanalysis of the initial dynamic system",&sdyn);
  IntParameter("RESEVRYDISP",1,"save displacements and contact forces every RESEVRYDISP steps",&sdyn);
  IntParameter("RESEVRYSTRS",1,"save stresses every RESEVRYSTRS steps",&sdyn);
  IntParameter("RESTARTEVRY",1,"write restart possibility every RESTARTEVRY steps",&sdyn);
  // Time loop control
  DoubleParameter("TIMESTEP",0.05,"time step size",&sdyn);
  IntParameter("NUMSTEP",200,"maximum number of steps",&sdyn);
  DoubleParameter("MAXTIME",5.0,"maximum time",&sdyn);
  // Generalised-alpha parameters
  DoubleParameter("BETA",0.25,"generalized alpha factors, also used by explicit time integration",&sdyn);
  DoubleParameter("DELTA",0.25,"generalized alpha factors",&sdyn);
  DoubleParameter("GAMMA",0.5,"generalized alpha factors, also used by explicit time integration",&sdyn);
  DoubleParameter("ALPHA_M",0.5,"generalized alpha factors",&sdyn);
  DoubleParameter("ALPHA_F",0.5,"generalized alpha factors",&sdyn);
  // Rayleigh damping
  setStringToIntegralParameter("DAMPING","No",
                               "build raleigh damping matrix and use it from M_DAMP x M + K_DAMP x K",
                               yesnotuple,yesnovalue,
                               &sdyn);
  DoubleParameter("M_DAMP",0.5,"",&sdyn);
  DoubleParameter("K_DAMP",0.5,"",&sdyn);
  // Iteration
  setStringToIntegralParameter("ITERATION","full","unused",
                               tuple<std::string>("full","Full","FULL"),
                               tuple<int>(1,1,1),
                               &sdyn);
  setStringToIntegralParameter("CONV_CHECK","AbsRes_Or_AbsDis","type of convergence check",
                               tuple<std::string>(
                                 "AbsRes_Or_AbsDis",
                                 "AbsRes_And_AbsDis",
                                 "RelRes_Or_AbsDis",
                                 "RelRes_And_AbsDis",
                                 "RelRes_Or_RelDis",
                                 "RelRes_And_RelDis"),
                               tuple<int>(STRUCT_DYNAMIC::absres_or_absdis,
                                          STRUCT_DYNAMIC::absres_and_absdis,
                                          STRUCT_DYNAMIC::relres_or_absdis,
                                          STRUCT_DYNAMIC::relres_and_absdis,
                                          STRUCT_DYNAMIC::relres_or_reldis,
                                          STRUCT_DYNAMIC::relres_and_reldis),
                               &sdyn);

  DoubleParameter("TOLDISP",1.0E-10,
                  "tolerance in the displacement norm for the newton iteration",
                  &sdyn);
  DoubleParameter("TOLRES",1.0E-08,
                  "tolerance in the residual norm for the newton iteration",
                  &sdyn);
  DoubleParameter("TOLCONSTR",1.0E-08,
                    "tolerance in the constr error norm for the newton iteration",
                    &sdyn);
  IntParameter("MAXITER",50,
               "maximum number of iterations allowed for newton iteration before failure",
               &sdyn);

  setStringToIntegralParameter("NLNSOL","fullnewton","",
                               tuple<std::string>(
                                 "fullnewton",
                                 "lsnewton",
                                 "modnewton",
                                 "nlncg",
                                 "ptc"),
                               tuple<int>(
                                 STRUCT_DYNAMIC::fullnewton,
                                 STRUCT_DYNAMIC::lsnewton,
                                 STRUCT_DYNAMIC::modnewton,
                                 STRUCT_DYNAMIC::nlncg,
                                 STRUCT_DYNAMIC::ptc),
                               &sdyn);

  setStringToIntegralParameter("PREDICT","ConstDis","",
                               tuple<std::string>(
                                 "Vague",
                                 "ConstDis",
                                 "ConstDisVelAcc"),
                               tuple<int>(
                                 STRUCT_DYNAMIC::pred_vague,
                                 STRUCT_DYNAMIC::pred_constdis,
                                 STRUCT_DYNAMIC::pred_constdisvelacc),
                               &sdyn);

  // time adaptivity (old style)
  IntParameter("TIMEADAPT",0,"",&sdyn);
  IntParameter("ITWANT",0,"",&sdyn);
  DoubleParameter("MAXDT",0.0,"",&sdyn);
  DoubleParameter("RESULTDT",0.0,"",&sdyn);
  // Uzawa iteration for constraint systems
  DoubleParameter("UZAWAPARAM",1.0,"Parameter for Uzawa algorithm dealing with lagrange multipliers",&sdyn);
  IntParameter("UZAWAMAXITER",50,"maximum number of iterations allowed for uzawa algorithm before failure going to next newton step",&sdyn);
  setStringToIntegralParameter("UZAWAALGO","newtonlinuzawa","",
                                 tuple<std::string>(
                                   "newtonlinuzawa",
                                   "augmentedlagrange"),
                                 tuple<int>(
                                   STRUCT_DYNAMIC::linuzawa,
                                   STRUCT_DYNAMIC::nonlinuzawa),
                                 &sdyn);

  // convergence criteria adaptivity
  setStringToIntegralParameter("ADAPTCONV","No",
                               "Switch on adaptive control of linear solver tolerance for nonlinear solution",
                               yesnotuple,yesnovalue,&sdyn);
  DoubleParameter("ADAPTCONV_BETTER",0.1,"The linear solver shall be this much better than the current nonlinear residual in the nonlinear convergence limit",&sdyn);


  // Inverse Analysis
  setStringToIntegralParameter("INV_ANALYSIS","No",
                               "determines the material parameter for the hyperpolyconvex material",
                               yesnotuple,yesnovalue,
                               &sdyn);
  // Measured displacment during the experiments
  DoubleParameter("MEASURED_DISP0",0.0,"measured displacment of the tension testing",&sdyn);
  DoubleParameter("MEASURED_DISP1",0.0,"measured displacment of the tension testing",&sdyn);
  DoubleParameter("MEASURED_DISP2",0.0,"measured displacment of the tension testing",&sdyn);
  
  // mu for inv_analysis
  DoubleParameter("MU_MINUS",1.0,"parameter for inverse analysis",&sdyn);
  DoubleParameter("MU_PLUS",1.0,"parameter for inverse analysis",&sdyn);  
  
  // tolerance for inv_analysis
  DoubleParameter("INV_ANA_TOL",1.0,"tolerance for inverse analysis",&sdyn);  
    
  // Time adaptivity
  SetValidTimeAdaptivityParameters(sdyn);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& genalpha = sdyn.sublist("GENALPHA",false,"");

  setStringToIntegralParameter("GENAVG","ImrLike",
                               "mid-average type of internal forces",
                               tuple<std::string>("Vague",
                                                  "ImrLike",
                                                  "TrLike"),
                               tuple<int>(STRUCT_DYNAMIC::genavg_vague,
                                          STRUCT_DYNAMIC::genavg_imrlike,
                                          STRUCT_DYNAMIC::genavg_trlike),
                               &genalpha);
  DoubleParameter("BETA",0.25,"Generalised-alpha factor in (0,1/2]",&genalpha);
  DoubleParameter("GAMMA",0.5,"Generalised-alpha factor in (0,1]",&genalpha);
  DoubleParameter("ALPHA_M",0.5,"Generalised-alpha factor in [0,1)",&genalpha);
  DoubleParameter("ALPHA_F",0.5,"Generalised-alpha factor in [0,1)",&genalpha);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& onesteptheta = sdyn.sublist("ONESTEPTHETA",false,"");
  
  DoubleParameter("THETA",0.5,"One-step-theta factor in (0,1]",&onesteptheta);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& scontact = list->sublist("STRUCTURAL CONTACT",false,"");

  setStringToIntegralParameter("CONTACT","None","Type of structural contact",
                               tuple<std::string>("None","none",
                                                  "Normal","normal",
                                                  "Frictional","frictional",
                                                  "MeshTying","Meshtying","meshtying"),
                               tuple<int>(INPUTPARAMS::contact_none,INPUTPARAMS::contact_none,
                                          INPUTPARAMS::contact_normal,INPUTPARAMS::contact_normal,
                                          INPUTPARAMS::contact_frictional,INPUTPARAMS::contact_frictional,
                                          INPUTPARAMS::contact_meshtying,INPUTPARAMS::contact_meshtying,
                                          INPUTPARAMS::contact_meshtying),
                               &scontact);

  setStringToIntegralParameter("INIT_CONTACT","No","If chosen all slave nodes are set active at t=0",
                               yesnotuple,yesnovalue,&scontact);

  setStringToIntegralParameter("BASISTRAFO","No","If chosen basis transformation is applied to displacements",
                               yesnotuple,yesnovalue,&scontact);

  setStringToIntegralParameter("FRICTION","None","Type of friction law",
                                tuple<std::string>("None","none",
                                                   "Stick","stick",
                                                   "Tresca","tresca",
                                                   "Coulomb","coulomb"),
                                tuple<int>(INPUTPARAMS::friction_none,INPUTPARAMS::friction_none,
                                           INPUTPARAMS::friction_stick,INPUTPARAMS::friction_stick,
                                           INPUTPARAMS::friction_tresca,INPUTPARAMS::friction_tresca,
                                           INPUTPARAMS::friction_coulomb,INPUTPARAMS::friction_coulomb),
                                &scontact);

  DoubleParameter("FRBOUND",0.0,"Friction bound for Tresca friction",&scontact);
  DoubleParameter("FRCOEFF",0.0,"Friction coefficient for Coulomb friction",&scontact);

  setStringToIntegralParameter("FULL_LINEARIZATION","No","If chosen full linearization of contact is applied",
                               yesnotuple,yesnovalue,&scontact);
  
  setStringToIntegralParameter("SEMI_SMOOTH_NEWTON","No","If chosen semi-smooth Newton concept is applied",
                                 yesnotuple,yesnovalue,&scontact);
  
  DoubleParameter("SEMI_SMOOTH_CN",0.0,"Weighting factor cn for semi-smooth PDASS",&scontact);
  DoubleParameter("SEMI_SMOOTH_CT",0.0,"Weighting factor ct for semi-smooth PDASS",&scontact);
  
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn = list->sublist("FLUID DYNAMIC",false,"");

  setStringToIntegralParameter("DYNAMICTYP","Nlin_Time_Int",
                               "Nonlinear Time Integraton Scheme",
                               tuple<std::string>(
				 "Nlin_Time_Int",
                                 "Lin_Time_Int"
                                 ),
                               tuple<int>(
				dyntyp_nln_time_int,
                                dyntyp_lin_time_int
                                ),
                               &fdyn);

  setStringToIntegralParameter("TIMEINTEGR","One_Step_Theta",
                               "Time Integration Scheme",
                               tuple<std::string>(
                                 "Stationary",
                                 "Gen_Alfa",
                                 "Gen_Alpha",
                                 "One_Step_Theta",
                                 "BDF2",
                                 "Inc_Acc_Gen_Alpha",
                                 "Theta_Adamsbashforth"
                                 ),
                               tuple<FLUID_TIMEINTTYPE>(
                                 timeint_stationary,
                                 timeint_gen_alpha,
                                 timeint_gen_alpha,
                                 timeint_one_step_theta,
                                 timeint_bdf2,
                                 timeint_inc_acc_gen_alpha,
                                 timeint_theta_adamsbashforth
                                 ),
                               &fdyn);
  setStringToIntegralParameter("STARTINGALGO","One_Step_Theta","",
                               tuple<std::string>(
                                 "One_Step_Theta"
                                 ),
                               tuple<int>(
                                 timeint_one_step_theta
                                 ),
                               &fdyn);
  setStringToIntegralParameter("NONLINITER","fixed_point_like",
                               "Nonlinear iteration scheme",
                               tuple<std::string>(
                                 "fixed_point_like",
                                 "Newton"
                                 ),
                               tuple<int>(1,2),
                               &fdyn);

  setStringToIntegralParameter("CONVCHECK","L_2_norm",
                               "norm for convergence check",
                               tuple<std::string>(
                                 "No",
                                 "L_infinity_norm",
                                 "L_1_norm",
                                 "L_2_norm",
                                 "L_2_norm_without_residual_at_itemax"
                                 ),
                               tuple<std::string>(
                                 "do not check for convergence (ccarat)",
                                 "use max norm (ccarat)",
                                 "use abs. norm (ccarat)",
                                 "compute L2 errors of increments (reltive) and residuals (absolute)",
                                 "same as L_2_norm, only no residual norm is computed if itemax is reached (speedup for turbulence calculations, startup phase)"
                                 ),
                               tuple<int>(
                                 FLUID_DYNAMIC::fncc_no,
                                 FLUID_DYNAMIC::fncc_Linf,
                                 FLUID_DYNAMIC::fncc_L1,
                                 FLUID_DYNAMIC::fncc_L2,
                                 FLUID_DYNAMIC::fncc_L2_wo_res
                                 ),
                               &fdyn);
  setStringToIntegralParameter("STEADYCHECK","L_2_norm",
                               "Norm of steady state check",
                               tuple<std::string>(
                                 "No",
                                 "L_infinity_norm",
                                 "L_1_norm",
                                 "L_2_norm"
                                 ),
                               tuple<int>(
                                 FLUID_DYNAMIC::fncc_no,
                                 FLUID_DYNAMIC::fncc_Linf,
                                 FLUID_DYNAMIC::fncc_L1,
                                 FLUID_DYNAMIC::fncc_L2
                                 ),
                               &fdyn);
  setStringToIntegralParameter("INITIALFIELD","zero_field",
                               "Initial Starting Field",
                               tuple<std::string>(
                                 "zero_field",
                                 "field_from_file",
                                 "field_by_function",
                                 "disturbed_field_from_function",
                                 "SOLWAVE",
                                 "WAVEBREAKING",
                                 "BELTRAMI-FLOW",
                                 "KIM-MOIN-FLOW",
                                 "BREAKING-DAM"),
                               tuple<int>(0,1,2,3,6,7,8,9,10),
                               &fdyn);

  setStringToIntegralParameter("LIFTDRAG","No",
                               "Calculate lift and drag forces along specified boundary",
                               tuple<std::string>(
                                 "No",
                                 "no",
                                 "Yes",
                                 "yes",
                                 "Nodeforce",
                                 "NODEFORCE",
                                 "nodeforce"
                                 ),
                               tuple<int>(
                                 FLUID_DYNAMIC::ld_none,
                                 FLUID_DYNAMIC::ld_none,
                                 FLUID_DYNAMIC::ld_nodeforce,
                                 FLUID_DYNAMIC::ld_nodeforce,
                                 FLUID_DYNAMIC::ld_nodeforce,
                                 FLUID_DYNAMIC::ld_nodeforce,
                                 FLUID_DYNAMIC::ld_nodeforce
                                 ),
                               &fdyn);

  setStringToIntegralParameter("CD_VELOCITY","zero",
                               "type of velocity field used for convection-diffusion problems",
                               tuple<std::string>(
                                 "zero",
                                 "function",
                                 "Navier_Stokes"
                                 ),
                               tuple<int>(0,1,2),
                               &fdyn);

  IntParameter("CD_VELFUNCNO",-1,"function number for con-dif velocity field",&fdyn);

  setStringToIntegralParameter("CD_INITIALFIELD","zero_field",
                               "Initial Field for convection-diffusion",
                               tuple<std::string>(
                                 "zero_field",
                                 "field_by_function",
                                 "field_by_condition"),
                               tuple<int>(0,1,2),
                               &fdyn);

  IntParameter("CD_INITFUNCNO",-1,"function number for con-dif initial field",&fdyn);

  setStringToIntegralParameter("FSSUGRVISC","No","fine-scale subgrid diffusivity/viscosity",
                               tuple<std::string>(
                                 "No",
                                 "artificial_all",
                                 "artificial_small",
                                 "Smagorinsky_all",
                                 "Smagorinsky_small",
                                 "scale_similarity",
                                 "mixed_Smagorinsky_all",
                                 "mixed_Smagorinsky_small"
                                 ),
                               tuple<int>(0,1,2,3,4,5,6,7),
                               &fdyn);

  setStringToIntegralParameter("SIMPLER","no",
                               "Switch on SIMPLE family of solvers, needs additional FLUID PRESSURE SOLVER block!",
                               yesnotuple,yesnovalue,&fdyn);

  setStringToIntegralParameter("ADAPTCONV","yes",
                               "Switch on adaptive control of linear solver tolerance for nonlinear solution",
                               yesnotuple,yesnovalue,&fdyn);
  DoubleParameter("ADAPTCONV_BETTER",0.1,"The linear solver shall be this much better than the current nonlinear residual in the nonlinear convergence limit",&fdyn);

  IntParameter("UPPSS",1,"Increment for visualisation (unused)",&fdyn);
  IntParameter("UPOUT",1,"Increment for writing solution to output file",&fdyn);
  IntParameter("UPRES",1,"Increment for writing solution",&fdyn);
  IntParameter("RESSTEP",0,"Restart Step",&fdyn);
  IntParameter("RESTARTEVRY",20,"Increment for writing restart",&fdyn);
  IntParameter("NUMSTEP",1,"Total number of Timesteps",&fdyn);
  IntParameter("STEADYSTEP",-1,"steady state check every step",&fdyn);
  IntParameter("NUMSTASTEPS",0,"Number of Steps for Starting Scheme",&fdyn);
  IntParameter("STARTFUNCNO",-1,"Function for Initial Starting Field",&fdyn);
  IntParameter("ITEMAX",10,"max. number of nonlin. iterations",&fdyn);
  IntParameter("GRIDVEL",1,"order of accuracy of mesh velocity determination",&fdyn);
  DoubleParameter("TIMESTEP",0.01,"Time increment dt",&fdyn);
  DoubleParameter("MAXTIME",1000.0,"Total simulation time",&fdyn);
  DoubleParameter("ALPHA_M",1.0,"Time integration factor",&fdyn);
  DoubleParameter("ALPHA_F",1.0,"Time integration factor",&fdyn);
  DoubleParameter("GAMMA",1.0,"Time integration factor",&fdyn);

  DoubleParameter("THETA",0.66,"Time integration factor",&fdyn);

  DoubleParameter("CONVTOL",1e-6,"Tolerance for convergence check",&fdyn);
  DoubleParameter("STEADYTOL",1e-6,"Tolerance for steady state check",&fdyn);
  DoubleParameter("START_THETA",1.0,"Time integraton factor for starting scheme",&fdyn);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_stab = fdyn.sublist("STABILIZATION",false,"");

  // this parameter seperates stabilized from unstabilized methods
  setStringToIntegralParameter("STABTYPE",
                               "residual_based",
                               "Apply (un)stabilized fluid formulation",
                               tuple<std::string>(
                                 "no_stabilization",
                                 "inconsistent",
                                 "residual_based"),
                               tuple<std::string>(
                                 "Do not use any stabilization -> inf-sup stable elements required!",
                                 "Similar to residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
                                 "Use a residual-based stabilization or, more generally, a stabilization \nbased on the concept of the residual-based variational multiscale method...\nExpecting additional input")  ,
                               tuple<int>(0,1,2),
                               &fdyn_stab);

  // the following parameters are necessary only if a residual based stabilized method is applied
  setStringToIntegralParameter("TDS",
                               "quasistatic",
                               "Flag to allow time dependency of subscales for residual-based stabilization.",
                               tuple<std::string>(
                                 "quasistatic",
                                 "time_dependent"),
                               tuple<std::string>(
                                 "Use a quasi-static residual-based stabilization (standard case)",
                                 "Residual-based stabilization including time evolution equations for subscales"),
                               tuple<int>(0,1),
                               &fdyn_stab);

  setStringToIntegralParameter("TRANSIENT",
                               "no_transient",
                               "Specify how to treat the transient term.",
                               tuple<std::string>(
                                 "no_transient",
                                 "yes_transient"),
                               tuple<std::string>(
                                 "Do not use transient term (currently only opportunity for quasistatic stabilization)",
                                 "Use transient term (recommended for time dependent subscales)"),
                               tuple<int>(0,1),
                               &fdyn_stab);

  setStringToIntegralParameter("PSPG",
                               "yes_pspg",
                               "Flag to (de)activate PSPG.",
                               tuple<std::string>(
                                 "no_pspg",
                                 "yes_pspg"),
                               tuple<std::string>(
                                 "No PSPG -> inf-sup-stable elements mandatory",
                                 "Use PSPG -> allowing for equal-order interpolation"),
                               tuple<int>(0,1),
                               &fdyn_stab);

  setStringToIntegralParameter("SUPG",
                               "yes_supg",
                               "Flag to (de)activate SUPG.",
                               tuple<std::string>(
                                 "no_supg",
                                 "yes_supg"),
                               tuple<std::string>(
                                 "No SUPG",
                                 "Use SUPG."),
                               tuple<int>(0,1),
                               &fdyn_stab);

  setStringToIntegralParameter("VSTAB",
                               "no_vstab",
                               "Flag to (de)activate viscous term in residual-based stabilization.",
                               tuple<std::string>(
                                 "no_vstab",
                                 "vstab_gls",
                                 "vstab_gls_rhs",
                                 "vstab_usfem",
                                 "vstab_usfem_rhs"
                                 ),
                               tuple<std::string>(
                                 "No viscous term in stabilization",
                                 "Viscous stabilization of GLS type",
                                 "Viscous stabilization of GLS type, included only on the right hand side",
                                 "Viscous stabilization of USFEM type",
                                 "Viscous stabilization of USFEM type, included only on the right hand side"
                                 ),
                               tuple<int>(0,1,2,3,4),
                               &fdyn_stab);

  setStringToIntegralParameter("CSTAB",
                               "cstab_qs",
                               "Flag to (de)activate least-squares stabilization of continuity equation.",
                               tuple<std::string>(
                                 "no_cstab",
                                 "cstab_qs",
                                 "cstab_td"),
                               tuple<std::string>(
                                 "No continuity stabilization",
                                 "Quasistatic continuity stabilization",
                                 "Time-dependent generalization of continuity stabilization"),
                               tuple<int>(0,1,2),
                               &fdyn_stab);

  setStringToIntegralParameter("CROSS-STRESS",
                               "no_cross",
                               "Flag to (de)activate cross-stress term -> residual-based VMM.",
                               tuple<std::string>(
                                 "no_cross",
                                 "cross_complete",
                                 "cross_rhs"
                                 ),
                               tuple<std::string>(
                                 "No cross-stress term",
                                 "Include the cross-stress term with a linearization of the convective part",
                                 "Include cross-stress term, but only explicitly on right hand side"
                                 ),
                               tuple<int>(0,1,2),
                               &fdyn_stab);

  setStringToIntegralParameter("REYNOLDS-STRESS",
                               "no_reynolds",
                               "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
                               tuple<std::string>(
                                 "no_reynolds",
                                 "reynolds_rhs"
                                 ),
                               tuple<std::string>(
                                 "No Reynolds-stress term",
                                 "Include Reynolds-stress term explicitly on right hand side"
                                 ),
                               tuple<int>(0,1),
                               &fdyn_stab);

  // this parameter selects the tau definition applied
  setStringToIntegralParameter("DEFINITION_TAU",
                               "Barrenechea_Franca_Valentin_Wall",
                               "Definition of tau_M,C",
                               tuple<std::string>(
                                 "Barrenechea_Franca_Valentin_Wall",
                                 "Bazilevs",
                                 "Codina"),
                               tuple<std::string>(
                                 "tau_Mp: Barrenechea, Valentin; tau_M: Franca, Barrenechea; tau_C: Wall",
                                 "tau_M and tau_C (Bazilevs, based on G_ij and g_i)",
                                 "tau_Mp: Barrenechea, Valentin; tau_M: Franca, Barrenechea; tau_C: Codina")  ,
                               tuple<int>(0,1,2),
                               &fdyn_stab);

  setStringToIntegralParameter("OUTFLOW_STAB",
                               "no_outstab",
                               "Flag to (de)activate outflow stabilization term",
                               tuple<std::string>(
                                 "no_outstab",
                                 "yes_outstab"),
                               tuple<std::string>(
                                 "No outflow stabilization term",
                                 "Add outflow stabilization term."),
                               tuple<int>(0,1),
                               &fdyn_stab);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_turbu = fdyn.sublist("TURBULENCE MODEL",false,"");

  setStringToIntegralParameter("TURBULENCE_APPROACH",
                               "DNS_OR_RESVMM_LES",
                               "There are several options to deal with turbulent flows.",
                               tuple<std::string>(
                                 "DNS_OR_RESVMM_LES",
                                 "CLASSICAL_LES",
                                 "RANS"),
                               tuple<std::string>(
                                 "Try to solve flow as an underresolved DNS.\nMind that your stabilisation already acts as a kind of turbulence model!",
                                 "Perform a classical Large Eddy Simulation adding \naddititional turbulent viscosity. This may be based on various physical models.",
                                 "Solve Reynolds averaged Navier Stokes using an \nalgebraic, one- or two equation closure.\nNot implemented yet."),
                               tuple<int>(0,1,2),
                               &fdyn_turbu);

  setStringToIntegralParameter("PHYSICAL_MODEL",
                               "no_model",
                               "Classical LES approaches require an additional model for\nthe turbulent viscosity.",
                               tuple<std::string>(
                                 "no_model",
                                 "Smagorinsky",
                                 "Smagorinsky_with_van_Driest_damping",
                                 "Dynamic_Smagorinsky"),
                               tuple<std::string>(
                                 "If classical LES is our turbulence approach, this is a contradiction and should cause a dserror.",
                                 "Classical constant coefficient Smagorinsky model. Be careful if you \nhave a wall bounded flow domain!",
                                 "Use an exponential damping function for the turbulent viscosity \nclose to the wall. This is only implemented for a channel geometry of \nheight 2 in y direction. The viscous lengthscale l_tau is \nrequired as additional input.",
                                 "The solution is filtered and by comparison of the filtered \nvelocity field with the real solution, the Smagorinsky constant is \nestimated in each step --- mind that this procedure includes \nan averaging in the xz plane, hence this implementation will only work \nfor a channel flow."),
                               tuple<int>(0,1,2,3),
                               &fdyn_turbu);

  DoubleParameter("C_SMAGORINSKY",0.0,"Constant for the Smagorinsky model. Something between 0.1 to 0.24",&fdyn_turbu);

  setStringToIntegralParameter("CANONICAL_FLOW",
                               "no",
                               "Sampling is different for different canonical flows \n--- so specify what kind of flow you've got",
                               tuple<std::string>(
                                 "no",
                                 "channel_flow_of_height_2",
                                 "lid_driven_cavity",
                                 "square_cylinder"),
                               tuple<std::string>(
                                 "The flow is not further specified, so spatial averaging \nand hence the standard sampling procedure is not possible",
                                 "For this flow, all statistical data could be averaged in \nthe homogenous planes --- it is essentially a statistically one dimensional flow.",
                                 "For this flow, all statistical data are evaluated on the center lines of the xy-midplane, averaged only over time.",
                                 "For this flow, statistical data are evaluated on various lines of the xy-midplane, averaged only over time."),
                               tuple<int>(0,1,2,3),
                               &fdyn_turbu);

  setStringToIntegralParameter("CHANNEL_HOMPLANE",
                               "xz",
                               "Specify the homogenous plane in a channel flow",
                               tuple<std::string>(
                                 "xy",
                                 "xz",
                                 "yz"),
                               tuple<std::string>(
                                 "Wall normal direction is z",
                                 "Wall normal direction is y (the standard case)",
                                 "Wall normal direction is x"),
                               tuple<int>(0,1,2),
                               &fdyn_turbu);

  DoubleParameter("CHANNEL_L_TAU",0.0,"Used for normalisation of the wall normal distance in the Van \nDriest Damping function. May be taken from the output of \nthe apply_mesh_stretching.pl preprocessing script.",&fdyn_turbu);

  DoubleParameter("CHAN_AMPL_INIT_DIST",0.1,"Max. amplitude of the random disturbance in percent of the initial value in mean flow direction.",&fdyn_turbu);

  IntParameter("SAMPLING_START",1,"Time step after when sampling shall be started",&fdyn_turbu);
  IntParameter("SAMPLING_STOP",1,"Time step when sampling shall be stopped",&fdyn_turbu);
  IntParameter("DUMPING_PERIOD",1,"Period of time steps after which statistical data shall be dumped",&fdyn_turbu);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_combust = fdyn.sublist("COMBUSTION",false,"");  
  
  DoubleParameter("LAMINAR FLAMESPEED",1.0,"The laminar flamespeed incorporates all chemical kinetics into the problem for now",&fdyn_combust);
  DoubleParameter("MARKSTEIN LENGTH",0.0,"The Markstein length takes flame curvature into account",&fdyn_combust);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& adyn = list->sublist("ALE DYNAMIC",false,"");

  DoubleParameter("TIMESTEP",0.1,"",&adyn);
  IntParameter("NUMSTEP",41,"",&adyn);
  DoubleParameter("MAXTIME",4.0,"",&adyn);
  setStringToIntegralParameter("ALE_TYPE","classic_lin","ale mesh algorithm",
                               tuple<std::string>("classic_lin","springs"),
                               tuple<int>(ALE_DYNAMIC::classic_lin,
                                          ALE_DYNAMIC::springs),
                               &adyn);
  IntParameter("NUM_INITSTEP",0,"",&adyn);
  IntParameter("RESEVRYDISP",1,"",&adyn);

  setStringToIntegralParameter("QUALITY","none","unused",
                               tuple<std::string>("none","NONE"),
                               tuple<int>(
                                 ALE_DYNAMIC::no_quality,
                                 ALE_DYNAMIC::no_quality),
                               &adyn);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fsidyn = list->sublist(
    "FSI DYNAMIC",false,
    "Fluid Structure Interaction\n"
    "Partitioned FSI solver with various coupling methods"
    );

  setStringToIntegralParameter("COUPALGO","iter_stagg_AITKEN_rel_param",
                               "Iteration Scheme over the fields",
                               tuple<std::string>(
                                 "basic_sequ_stagg",
                                 //"sequ_stagg_pred",
                                 //"sequ_stagg_shift",
                                 "iter_stagg_fixed_rel_param",
                                 "iter_stagg_AITKEN_rel_param",
                                 "iter_stagg_steep_desc",
                                 "iter_stagg_NLCG",
                                 "iter_stagg_MFNK_FD",
                                 "iter_stagg_MFNK_FSI",
                                 "iter_stagg_MPE",
                                 "iter_stagg_RRE",
                                 "iter_monolithic")
                               .append("iter_monolithiclagrange")
                               .append("iter_monolithicstructuresplit")
                               .append("pseudo_structure"),
                               tuple<int>(
                                 fsi_basic_sequ_stagg,
                                 //fsi_sequ_stagg_pred,
                                 //fsi_sequ_stagg_shift,
                                 fsi_iter_stagg_fixed_rel_param,
                                 fsi_iter_stagg_AITKEN_rel_param,
                                 fsi_iter_stagg_steep_desc,
                                 fsi_iter_stagg_NLCG,
                                 fsi_iter_stagg_MFNK_FD,
                                 fsi_iter_stagg_MFNK_FSI,
                                 fsi_iter_stagg_MPE,
                                 fsi_iter_stagg_RRE,
                                 fsi_iter_monolithic)
                               .append(fsi_iter_monolithiclagrange)
                               .append(fsi_iter_monolithicstructuresplit)
                               .append(fsi_pseudo_structureale),
                               &fsidyn);

  setStringToIntegralParameter("PARTITIONED","DirichletNeumann",
                               "Coupling strategies for partitioned FSI solvers. Most of the time Dirichlet-Neumann is just right.",
                               tuple<std::string>(
                                 "DirichletNeumann",
                                 "RobinNeumann",
                                 "DirichletRobin",
                                 "RobinRobin"
                                 ),
                               tuple<INPUTPARAMS::FSIPartitionedCouplingMethod>(
                                 INPUTPARAMS::fsi_DirichletNeumann,
                                 INPUTPARAMS::fsi_RobinNeumann,
                                 INPUTPARAMS::fsi_DirichletRobin,
                                 INPUTPARAMS::fsi_RobinRobin
                                 ),
			       &fsidyn);

  DoubleParameter("ALPHA_F",-1.0,"Robin parameter fluid",&fsidyn);
  DoubleParameter("ALPHA_S",-1.0,"Robin parameter structure",&fsidyn);

  setStringToIntegralParameter("DEBUGOUTPUT","No",
                               "Output of unconverged interface values during partitioned FSI iteration.\n"
                               "There will be a new control file for each time step.\n"
                               "This might be helpful to understand the coupling iteration.",
                               yesnotuple,yesnovalue,&fsidyn);

  setStringToIntegralParameter("PREDICTOR","d(n)+dt*v(n)+0.5*dt^2*a(n)",
                               "Predictor for interface displacements",
                               tuple<std::string>(
                                 "d(n)",
                                 "d(n)+dt*(1.5*v(n)-0.5*v(n-1))",
                                 "d(n)+dt*v(n)",
                                 "d(n)+dt*v(n)+0.5*dt^2*a(n)"
                                 ),
                               tuple<int>(1,2,3,4),
                               &fsidyn);

  setStringToIntegralParameter("CONVCRIT","||g(i)||:sqrt(neq)",
                               "Convergence criterium for iteration over fields (unused)",
                               tuple<std::string>(
                                 "||g(i)||:sqrt(neq)",
                                 "||g(i)||:||g(0)||"
                                 ),
                               tuple<int>(1,2),
                               &fsidyn);

  setStringToIntegralParameter("COUPVARIABLE","Displacement",
                               "Coupling variable at the interface",
                               tuple<std::string>("Displacement","Force"),
                               tuple<int>(0,1),
                               &fsidyn);

  setStringToIntegralParameter("ENERGYCHECK","No",
                               "Energy check for iteration over fields",
                               yesnotuple,yesnovalue,&fsidyn);

  setStringToIntegralParameter("IALE","Pseudo_Structure",
                               "Treatment of ALE-field (outdated)",
                               tuple<std::string>(
                                 "Pseudo_Structure"
                                 ),
                               tuple<int>(1),
                               &fsidyn);

  setStringToIntegralParameter("COUPMETHOD","conforming",
                               "Coupling Method Mortar (mtr) or conforming nodes at interface",
                               tuple<std::string>(
                                 "MTR",
                                 "Mtr",
                                 "mtr",
                                 "conforming"
                                 ),
                               tuple<int>(0,0,0,1),
                               &fsidyn);

  setStringToIntegralParameter("COUPFORCE","nodeforce",
                               "Coupling force. Unused. We always couple with nodal forces.",
                               tuple<std::string>(
                                 "none",
                                 "stress",
                                 "nodeforce"
                                 ),
                               tuple<int>(
                                 FSI_DYNAMIC::cf_none,
                                 FSI_DYNAMIC::cf_stress,
                                 FSI_DYNAMIC::cf_nodeforce),
                               &fsidyn);

  setStringToIntegralParameter("SECONDORDER","No",
                               "Second order coupling at the interface.",
                               yesnotuple,yesnovalue,&fsidyn);

  setStringToIntegralParameter("MESHLINEARIZATION","No",
                               "Include linearization with respect to mesh movement in Navier Stokes equation.\n"
                               "Supported in monolithic FSI for now.",
                               yesnotuple,yesnovalue,&fsidyn);

  IntParameter("ITECHAPP",1,"",&fsidyn);
  IntParameter("ICHMAX",1,"",&fsidyn);
  IntParameter("ISDMAX",1,"not used up to now",&fsidyn);
  IntParameter("NUMSTEP",200,"Total number of Timesteps",&fsidyn);
  IntParameter("ITEMAX",100,"Maximum number of iterations over fields",&fsidyn);
  IntParameter("UPPSS",1,"Increment for visualisation",&fsidyn);
  IntParameter("UPRES",1,"Increment for writing solution",&fsidyn);
  IntParameter("RESTARTEVRY",1,"Increment for writing restart",&fsidyn);

  DoubleParameter("TIMESTEP",0.1,"Time increment dt",&fsidyn);
  DoubleParameter("MAXTIME",1000.0,"Total simulation time",&fsidyn);
  DoubleParameter("TOLENCHECK",1e-6,"Tolerance for energy check",&fsidyn);
  DoubleParameter("RELAX",1.0,"fixed relaxation parameter",&fsidyn);
  DoubleParameter("CONVTOL",1e-6,"Tolerance for iteration over fields",&fsidyn);
  DoubleParameter("MAXOMEGA",0.0,"largest omega allowed for Aitken relaxation (0.0 means no constraint)",&fsidyn);

  DoubleParameter("BASETOL",1e-3,
                  "Basic tolerance for adaptive convergence check in monolithic FSI.\n"
                  "This tolerance will be used for the linear solve of the FSI block system.\n"
                  "The linear convergence test will always use the relative residual norm (AZ_r0).\n"
                  "Not to be confused with the Newton tolerance (CONVTOL) that applies\n"
                  "to the nonlinear convergance test.",
                  &fsidyn);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& xfem_general = list->sublist("XFEM GENERAL",false,"");

  setStringToIntegralParameter("GMSH_DEBUG_OUT","Yes","Do you want to write extended Gmsh output for each timestep?",
                               yesnotuple,yesnovalue,&xfem_general);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fluidsolver = list->sublist("FLUID SOLVER",false,"");
  SetValidSolverParameters(fluidsolver);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fluidpsolver = list->sublist("FLUID PRESSURE SOLVER",false,"pressure solver parameters for SIMPLE preconditioning");
  SetValidSolverParameters(fluidpsolver);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& structsolver = list->sublist("STRUCT SOLVER",false,"");
  SetValidSolverParameters(structsolver);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& alesolver = list->sublist("ALE SOLVER",false,"");
  SetValidSolverParameters(alesolver);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& thermalsolver = list->sublist("THERMAL SOLVER",false,"");
  SetValidSolverParameters(thermalsolver);

  return list;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::SetValidSolverParameters(Teuchos::ParameterList& list)
{
  using Teuchos::tuple;
  using Teuchos::setStringToIntegralParameter;

  setStringToIntegralParameter("SOLVER","UMFPACK","",
                               tuple<std::string>(
                                 "Amesos_KLU_sym",
                                 "Amesos_KLU_nonsym",
                                 "Superlu",
                                 "vm3",
                                 "Aztec_MSR",
                                 "LAPACK_sym",
                                 "LAPACK_nonsym",
                                 "UMFPACK"
                                 ),
                               tuple<_SOLVER_TYP>(
                                 amesos_klu_sym,
                                 amesos_klu_nonsym,
                                 superlu,
                                 vm3,
                                 aztec_msr,
                                 lapack_sym,
                                 lapack_nonsym,
                                 umfpack),
                               &list);

  setStringToIntegralParameter(
    "AZSOLVE", "GMRES",
    "Type of linear solver algorithm to use.",
    tuple<std::string>("CG","GMRES","CGS","TFQMR","BiCGSTAB","LU"),
    tuple<_AZSOLVERTYP>(azsolv_CG,azsolv_GMRES,azsolv_CGS,azsolv_TFQMR,azsolv_BiCGSTAB,azsolv_LU),
    &list
    );
  setStringToIntegralParameter(
    "AZPREC", "ILU",
    "Type of internal preconditioner to use.\n"
    "Note! this preconditioner will only be used if the input operator\n"
    "supports the Epetra_RowMatrix interface and the client does not pass\n"
    "in an external preconditioner!",
    tuple<std::string>(
      "none",
      "ILU",
      "ILUT",
      "Jacobi",
      "SymmGaussSeidel",
      //"Polynomial",
      "Least_Squares",
      "Neumann",
      "ICC",
      "LU",
      "RILU").append("BILU").append("ML").append("MLFLUID").append("MLFLUID2").append("MLAPI").append("GaussSeidel").append("DownwindGaussSeidel"),
    //tuple<EAztecPreconditioner>(
    tuple<_AZPRECTYP>(
      azprec_none,
      azprec_ILU,
      azprec_ILUT,
      azprec_Jacobi,
      azprec_SymmGaussSeidel,
      //AZTEC_PREC_POLY,
      azprec_Least_Squares,
      azprec_Neumann,
      azprec_ICC,
      azprec_LU,
      azprec_RILU).append(azprec_BILU).append(azprec_ML).append(azprec_MLfluid).append(azprec_MLfluid2).append(azprec_MLAPI).append(azprec_GaussSeidel).append(azprec_DownwindGaussSeidel),
    &list
    );
  IntParameter(
    "AZOVERLAP", 0,
    "The amount of overlap used for the internal \"ilu\" and \"ilut\" preconditioners.",
    &list
    );
  IntParameter(
    "AZGFILL", 0,
    "The amount of fill allowed for the internal \"ilu\" preconditioner.",
    &list
    );
  DoubleParameter(
    "AZDROP", 0.0,
    "The tolerance below which an entry from the factors of an internal \"ilut\"\n"
    "preconditioner will be dropped.",
    &list
    );
  DoubleParameter(
    "AZFILL", 1.0,
    "The amount of fill allowed for an internal \"ilut\" preconditioner.",
    &list
    );
//   IntParameter(
//     Steps_name, 3,
//     "Number of steps taken for the \"Jacobi\" or the \"Symmetric Gauss-Seidel\"\n"
//     "internal preconditioners for each preconditioner application.",
//     &list
//     );
  IntParameter(
    "AZPOLY", 3,
    "The order for of the polynomials used for the \"Polynomial\" and\n"
    "\"Least-squares Polynomial\" internal preconditioners.",
    &list
    );
//   setStringToIntegralParameter(
//     RCMReordering_name, "Disabled",
//     "Determines if RCM reordering is used with the internal\n"
//     "\"ilu\" or \"ilut\" preconditioners.",
//     tuple<std::string>("Enabled","Disabled"),
//     tuple<int>(1,0),
//     &list
//     );
//   setStringToIntegralParameter(
//     Orthogonalization_name, "Classical",
//     "The type of orthogonalization to use with the \"GMRES\" solver.",
//     tuple<std::string>("Classical","Modified"),
//     tuple<int>(AZ_classic,AZ_modified),
//     &list
//     );
  IntParameter(
    "AZSUB", 300,
    "The maximum size of the Krylov subspace used with \"GMRES\" before\n"
    "a restart is performed.",
    &list
    );
  setStringToIntegralParameter(
    "AZCONV", "AZ_r0", // Same as "rhs" when x=0
    "The convergence test to use for terminating the iterative solver.",
    tuple<std::string>(
      "AZ_r0",
      "AZ_rhs",
      "AZ_Anorm",
      "AZ_noscaled",
      "AZ_sol",
      "AZ_weighted",
      "AZ_expected_values",
      "AZTECOO_conv_test",
      "AZ_inf_noscaled"
      ),
    tuple<int>(
      AZ_r0,
      AZ_rhs,
      AZ_Anorm,
      AZ_noscaled,
      AZ_sol,
      AZ_weighted,
      AZ_expected_values,
      AZTECOO_conv_test,
      AZ_inf_noscaled
      ),
    &list
    );
//   DoubleParameter(
//     IllConditioningThreshold_name, 1e+11,
//     "The threshold tolerance above which a system is considered\n"
//     "ill conditioned.",
//     &list
//     );
  IntParameter(
    "AZOUTPUT", 0, // By default, no output from Aztec!
    "The number of iterations between each output of the solver's progress.",
    &list
    );

  IntParameter("AZREUSE", 0, "how often to recompute some preconditioners", &list);
  IntParameter("AZITER", 1000, "max iterations", &list);
  IntParameter("AZGRAPH", 0, "unused", &list);
  IntParameter("AZBDIAG", 0, "", &list);

  DoubleParameter("AZTOL", 1e-8, "tolerance in (un)scaled residual", &list);
  DoubleParameter("AZOMEGA", 0.0, "damping for GaussSeidel and jacobi type methods", &list);
  DoubleParameter("DWINDTAU",1.5,"threshold tau for downwinding", &list);

  setStringToIntegralParameter(
    "AZSCAL","none","scaling of the system",
    tuple<std::string>("none","sym","infnorm"),
    tuple<int>(0,1,2),
    &list);

  // parameters of ML preconditioner

  IntParameter("ML_PRINT",0,
               "ML print-out level (0-10)",&list);
  IntParameter("ML_MAXCOARSESIZE",5000,
               "ML stop coarsening when coarse ndof smaller then this",&list);
  IntParameter("ML_MAXLEVEL",5,
               "ML max number of levels",&list);
  IntParameter("ML_AGG_SIZE",27,
               "objective size of an aggregate with METIS/VBMETIS, 2D: 9, 3D: 27",&list);

  DoubleParameter("ML_DAMPFINE",1.,"damping fine grid",&list);
  DoubleParameter("ML_DAMPMED",1.,"damping med grids",&list);
  DoubleParameter("ML_DAMPCOARSE",1.,"damping coarse grid",&list);
  DoubleParameter("ML_PROLONG_SMO",0.,"damping factor for prolongator smoother (usually 1.33 or 0.0)",&list);
  DoubleParameter("ML_PROLONG_THRES",0.,"threshold for prolongator smoother/aggregation",&list);

  setNumericStringParameter("ML_SMOTIMES","1 1 1 1 1 1",
                            "no. smoothing steps or polynomial order on each level (at least ML_MAXLEVEL numbers)",&list);

  setStringToIntegralParameter(
    "ML_COARSEN","UC","",
    tuple<std::string>("UC","METIS","VBMETIS","MIS"),
    tuple<int>(0,1,2,3),
    &list);

  setStringToIntegralParameter(
    "ML_SMOOTHERFINE","ILU","",
    tuple<std::string>("SGS","Jacobi","Chebychev","MLS","ILU","KLU","Superlu","GS","DGS"),
    tuple<int>(0,1,2,3,4,5,6,7,8),
    &list);

  setStringToIntegralParameter(
    "ML_SMOOTHERMED","ILU","",
    tuple<std::string>("SGS","Jacobi","Chebychev","MLS","ILU","KLU","Superlu","GS","DGS"),
    tuple<int>(0,1,2,3,4,5,6,7,8),
    &list);

  setStringToIntegralParameter(
    "ML_SMOOTHERCOARSE","KLU","",
    tuple<std::string>("SGS","Jacobi","Chebychev","MLS","ILU","KLU","Superlu","GS","DGS"),
    tuple<int>(0,1,2,3,4,5,6,7,8),
    &list);

  // unused
  setStringToIntegralParameter("PARTITION","Cut_Elements","unused",
                               tuple<std::string>("Cut_Elements"),
                               tuple<int>(0),
                               &list);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::SetValidTimeAdaptivityParameters(Teuchos::ParameterList& list)
{
  using Teuchos::tuple;
  using Teuchos::setStringToIntegralParameter;

  setStringToIntegralParameter(
    "TA_KIND","None","",
    tuple<std::string>("None","ZienkiewiczXie"),
    tuple<int>(
      TIMADA_DYNAMIC::timada_kind_none,
      TIMADA_DYNAMIC::timada_kind_zienxie),
    &list);

  DoubleParameter("TA_STEPSIZEMAX", 0.0, "", &list);
  DoubleParameter("TA_STEPSIZEMIN", 0.0, "", &list);
  DoubleParameter("TA_SIZERATIOMAX", 0.0, "", &list);
  DoubleParameter("TA_SIZERATIOMIN", 0.0, "", &list);
  DoubleParameter("TA_SIZERATIOSCALE", 0.0, "", &list);

  setStringToIntegralParameter(
    "TA_ERRNORM", "Vague", "",
    tuple<std::string>("Vague",
                       "L1",
                       "L2",
                       "Rms",
                       "Inf"),
    tuple<int>(TIMADA_DYNAMIC::timada_err_norm_vague,
               TIMADA_DYNAMIC::timada_err_norm_l1,
               TIMADA_DYNAMIC::timada_err_norm_l2,
               TIMADA_DYNAMIC::timada_err_norm_rms,
               TIMADA_DYNAMIC::timada_err_norm_inf),
    &list);

  DoubleParameter("TA_ERRTOL", 0.0, "", &list);
  IntParameter("TA_ADAPTSTEPMAX", 0, "", &list);
}


#endif
