// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_impedancecondition.hpp"

#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | Constructor (public)                                      Thon 07/16 |
 *----------------------------------------------------------------------*/
FLD::Utils::FluidImpedanceWrapper::FluidImpedanceWrapper(
    const std::shared_ptr<Core::FE::Discretization> actdis)
{
  std::vector<const Core::Conditions::Condition*> impedancecond;
  actdis->get_condition("ImpedanceCond", impedancecond);

  // the number of lines of impedance boundary conditions found in the input
  // note that some of these lines could belong to the same physical condition
  // which is then marked by the same 'ConditionID'
  // The corresponding resorting of result values has to be done later
  int numcondlines = impedancecond.size();


  if (numcondlines < 1)
    FOUR_C_THROW(
        "this function should just be called if there is a least one impedance condition.");

  // -------------------------------------------------------------------
  // get time period length of first condition, this should always be
  // the same!
  // -------------------------------------------------------------------
  double period = impedancecond[0]->parameters().get<double>("TIMEPERIOD");

  // -------------------------------------------------------------------
  // now care for the fact that there could be more than one input line
  // belonging to the same impedance boundary condition
  // -------------------------------------------------------------------
  for (int i = 0; i < numcondlines; i++)
  {
    int condid = impedancecond[i]->parameters().get<int>("ConditionID");

    double thisperiod = impedancecond[i]->parameters().get<double>("TIMEPERIOD");
    if (thisperiod != period)
      FOUR_C_THROW("all periods of impedance conditions in one problem have to be the same!!!");

    // -------------------------------------------------------------------
    // allocate the impedance bc class members for every case
    // -------------------------------------------------------------------
    std::shared_ptr<FluidImpedanceBc> impedancebc =
        std::make_shared<FluidImpedanceBc>(actdis, condid, impedancecond[i]);

    // -----------------------------------------------------------------
    // sort impedance bc's in map and test, if one condition ID appears
    // more than once. Currently this case is forbidden.
    // -----------------------------------------------------------------
    bool inserted = impmap_.insert(std::make_pair(condid, impedancebc)).second;
    if (!inserted)
      FOUR_C_THROW(
          "There are more than one impedance condition lines with the same ID. This can not yet be "
          "handled.");
  }  // end loop over condition lines from input
  return;
}  // end FluidImpedanceWrapper

/*----------------------------------------------------------------------*
 |  Split linearization matrix to a BlockSparseMatrixBase    Thon 07/16 |
 *----------------------------------------------------------------------*/

void FLD::Utils::FluidImpedanceWrapper::use_block_matrix(
    std::shared_ptr<std::set<int>> condelements, const Core::LinAlg::MultiMapExtractor& domainmaps,
    const Core::LinAlg::MultiMapExtractor& rangemaps, bool splitmatrix)
{
  std::map<const int, std::shared_ptr<class FluidImpedanceBc>>::iterator mapiter;

  for (mapiter = impmap_.begin(); mapiter != impmap_.end(); mapiter++)
  {
    mapiter->second->FluidImpedanceBc::use_block_matrix(
        condelements, domainmaps, rangemaps, splitmatrix);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Wrap update of residual                                  Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceWrapper::add_impedance_bc_to_residual_and_sysmat(const double dta,
    const double time, Core::LinAlg::Vector<double>& residual, Core::LinAlg::SparseOperator& sysmat)
{
  std::map<const int, std::shared_ptr<class FluidImpedanceBc>>::iterator mapiter;

  for (mapiter = impmap_.begin(); mapiter != impmap_.end(); mapiter++)
  {
    // calc flux
    mapiter->second->FluidImpedanceBc::flow_rate_calculation(mapiter->first);
    // calc pressure and traction vector and appliy to fluid residual and sysmat
    mapiter->second->FluidImpedanceBc::calculate_impedance_tractions_and_update_residual_and_sysmat(
        residual, sysmat, dta, time, mapiter->first);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Wrap for time update of impedance conditions             Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceWrapper::time_update_impedances(const double time)
{
  std::map<const int, std::shared_ptr<class FluidImpedanceBc>>::iterator mapiter;

  for (mapiter = impmap_.begin(); mapiter != impmap_.end(); mapiter++)
  {
    // update time step
    mapiter->second->FluidImpedanceBc::time_update_impedance(time, mapiter->first);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Wrap restart writing                                     Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceWrapper::write_restart(Core::IO::DiscretizationWriter& output)
{
  std::map<const int, std::shared_ptr<class FluidImpedanceBc>>::iterator mapiter;

  for (mapiter = impmap_.begin(); mapiter != impmap_.end(); mapiter++)
  {
    mapiter->second->FluidImpedanceBc::write_restart(output, mapiter->first);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Wrap restart reading                                     Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceWrapper::read_restart(Core::IO::DiscretizationReader& reader)
{
  std::map<const int, std::shared_ptr<class FluidImpedanceBc>>::iterator mapiter;

  for (mapiter = impmap_.begin(); mapiter != impmap_.end(); mapiter++)
    mapiter->second->FluidImpedanceBc::read_restart(reader, mapiter->first);

  return;
}

/*----------------------------------------------------------------------*
 |  Return relative vector of relative pressure errors      Thon 07/16 |
 *----------------------------------------------------------------------*/
std::vector<double> FLD::Utils::FluidImpedanceWrapper::get_w_krelerrors()
{
  std::vector<double> wk_rel_error;

  // get an iterator to my map
  std::map<const int, std::shared_ptr<class FLD::Utils::FluidImpedanceBc>>::iterator mapiter;

  for (mapiter = impmap_.begin(); mapiter != impmap_.end(); mapiter++)
  {
    wk_rel_error.push_back(mapiter->second->FLD::Utils::FluidImpedanceBc::get_w_krelerror());
  }

  return wk_rel_error;
}


/*----------------------------------------------------------------------*
 |  Constructor (public)                                     Thon 07/16 |
 *----------------------------------------------------------------------*/
FLD::Utils::FluidImpedanceBc::FluidImpedanceBc(
    const std::shared_ptr<Core::FE::Discretization> actdis, const int condid,
    const Core::Conditions::Condition* impedancecond)
    : discret_(actdis),
      myrank_(Core::Communication::my_mpi_rank(discret_->get_comm())),
      theta_(0.5),
      treetype_(impedancecond->parameters().get<std::string>("TYPE")),
      period_(impedancecond->parameters().get<double>("TIMEPERIOD")),
      r1_(impedancecond->parameters().get<double>("R1")),
      r2_(impedancecond->parameters().get<double>("R2")),
      c_(impedancecond->parameters().get<double>("C")),
      functnum_(impedancecond->parameters().get<int>("FUNCT"))
{
  if (myrank_ == 0)
  {
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Initializing impedance condition " << condid << std::endl;
  }

  // ---------------------------------------------------------------------
  // Initialize member variables
  // ---------------------------------------------------------------------
  p_np_ = 0.0;
  q_np_ = 0.0;

  // NOTE: one could think of using non-zero initial conditions...
  p_n_ = 0.0;
  q_n_ = 0.0;

  w_krelerror_ = 1.0;
  p_0_ = p_n_;

  // ---------------------------------------------------------------------
  // get theta for time integration
  // ---------------------------------------------------------------------
  // get theta of global time integration scheme to use it here
  // if global time integration scheme is not ONESTEPTHETA, theta is by default = 0.5
  auto dyntype = Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(
      Global::Problem::instance()->fluid_dynamic_params(), "TIMEINTEGR");

  if (dyntype == Inpar::FLUID::TimeIntegrationScheme::timeint_one_step_theta)
  {
    theta_ = Global::Problem::instance()->fluid_dynamic_params().get<double>("THETA");
  }

  // ---------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // ---------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
  impedancetbc_ = Core::LinAlg::create_vector(*dofrowmap, true);
  impedancetbcsysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 108, false, true);
  // NOTE: do not call impedancetbcsysmat_->Complete() before it is filled, since
  // this is our check if it has already been initialized

  // some safety check
  if (not(treetype_ == "windkessel" or treetype_ == "resistive" or
          treetype_ == "pressure_by_funct"))
    FOUR_C_THROW("TYPE {} not supported!", treetype_);

  if (myrank_ == 0)
  {
    std::cout << "Impedance type: " << treetype_.c_str() << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Split linearization matrix to a BlockSparseMatrixBase   Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceBc::use_block_matrix(std::shared_ptr<std::set<int>> condelements,
    const Core::LinAlg::MultiMapExtractor& domainmaps,
    const Core::LinAlg::MultiMapExtractor& rangemaps, bool splitmatrix)
{
  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>> mat;

  if (splitmatrix)
  {
    // (re)allocate system matrix
    mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>(
        domainmaps, rangemaps, 108, false, true);
    mat->set_cond_elements(condelements);
    impedancetbcsysmat_ = mat;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Flow rate calculation                                    Thon 07/16 |
 *----------------------------------------------------------------------*/
/*!
  modified by chfoe 04/08

  Calculate the flow rate across an impedance boundary surface

  Flow rates are
  (1) calculated for single element surfaces
  (2) added up over the elements of the single procs
  (3) communicated and added over the procs
  (4) and finally stored within the vector 'flowrates_'

  The vector of the flowrates holds the flow rate history of the
  very last cycle!

*/
void FLD::Utils::FluidImpedanceBc::flow_rate_calculation(const int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_flowrate);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // create vector (+ initialization with zeros)
  std::shared_ptr<Core::LinAlg::Vector<double>> flowrates =
      Core::LinAlg::create_vector(*dofrowmap, true);

  discret_->evaluate_condition(eleparams, flowrates, "ImpedanceCond", condid);

  double local_flowrate = 0.0;
  for (int i = 0; i < dofrowmap->num_my_elements(); i++)
  {
    local_flowrate += (flowrates->local_values_as_span()[i]);
  }

  double flowrate = 0.0;
  flowrate = Core::Communication::sum_all(local_flowrate, dofrowmap->get_comm());

  q_np_ = flowrate;

  //  if (myrank_ == 0)
  //    std::cout<<"Impedance condition Id: "<<condid<<", Current flux: "<<flowrate<<std::endl;

  return;
}  // FluidImplicitTimeInt::flow_rate_calculation



/*----------------------------------------------------------------------*
 |  Apply Impedance to outflow boundary                      Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceBc::calculate_impedance_tractions_and_update_residual_and_sysmat(
    Core::LinAlg::Vector<double>& residual, Core::LinAlg::SparseOperator& sysmat, const double dta,
    const double time, const int condid)
{
  // ---------------------------------------------------------------------//
  // ---------------------------------------------------------------------//
  // calculate the impedance tractions                                    //
  // ---------------------------------------------------------------------//
  // ---------------------------------------------------------------------//

  double pressure;
  double Q_np_fac;

  if (treetype_ == "resistive")
  {
    Q_np_fac = r1_ + r2_;
    pressure = Q_np_fac * q_np_;
  }
  else if (treetype_ == "windkessel")
  {
    const double fac = 1.0 / (c_ * r2_ + theta_ * dta);
    const double p_n_fac = fac * (c_ * r2_ - dta * (1 - theta_));
    const double Q_n_fac = fac * (dta * (1 - theta_) * (r1_ + r2_) - c_ * r1_ * r2_);
    Q_np_fac = fac * (c_ * r1_ * r2_ + dta * theta_ * (r1_ + r2_));

    pressure = p_n_fac * p_n_ + Q_np_fac * q_np_ + Q_n_fac * q_n_;
  }
  else if (treetype_ == "pressure_by_funct")
  {
    pressure = Global::Problem::instance()
                   ->function_by_id<Core::Utils::FunctionOfTime>(functnum_)
                   .evaluate(time);
    Q_np_fac = 0.0;
  }
  else
  {
    pressure = 0.0;
    Q_np_fac = 0.0;

    FOUR_C_THROW("Treetype {} not supported!", treetype_);
  }

  // save pressure value
  p_np_ = pressure;

  // call the element to apply the pressure
  Teuchos::ParameterList eleparams;
  // action for elements
  eleparams.set<FLD::BoundaryAction>("action", FLD::Outletimpedance);

  eleparams.set("WindkesselPressure", pressure);

  //  if (myrank_ == 0)
  //    std::cout<<"Impedance condition Id: "<<condid<<", Windkessel pressure:
  //    "<<pressure<<std::endl;


  impedancetbc_->put_scalar(0.0);

  discret_->evaluate_condition(eleparams, impedancetbc_, "ImpedanceCond", condid);


  // ---------------------------------------------------------------------//
  // ---------------------------------------------------------------------//
  // initialize the linearization matrix (iff not done already)
  // ---------------------------------------------------------------------//
  // ---------------------------------------------------------------------//

  // NOTE: this can not be done in the constructor since maybe use_block_matrix
  // is called afterwards and hence its contenten is reset :(
  if (not impedancetbcsysmat_->filled())
  {
    // calculate dQ/du = ( \phi o n )_Gamma
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
    std::shared_ptr<Core::LinAlg::Vector<double>> dQdu =
        Core::LinAlg::create_vector(*dofrowmap, true);

    Teuchos::ParameterList eleparams2;
    // action for elements
    eleparams2.set<FLD::BoundaryAction>("action", FLD::dQdu);

    discret_->evaluate_condition(eleparams2, dQdu, "ImpedanceCond", condid);

    // now move dQdu to one proc
    std::shared_ptr<Core::LinAlg::Map> dofrowmapred = Core::LinAlg::allreduce_e_map(*dofrowmap);
    Core::LinAlg::Vector<double> dQdu_full(*dofrowmapred, true);

    Core::LinAlg::export_to(*dQdu, dQdu_full);  //!!! add off proc components


    // calculate d wk/du = d/du ( (v,n)_gamma n,phi)_Gamma were (d wk/du)_i,j= timefacs*
    // (phi_i,n)_Gamma * (phi_j,n)_Gamma
    // Note: this derivative cannot be build on element level, hence we have to do it here!
    impedancetbcsysmat_->zero();

    const double tfaclhs = eleparams2.get<double>("tfaclhs", 0.0);

    for (int lid = 0; lid < dQdu->local_length(); lid++)
    {
      const int gid = dofrowmap->gid(lid);
      const double val = dQdu->local_values_as_span()[lid];
      if (abs(val) > 1e-15)
      {
        for (int lid2 = 0; lid2 < dQdu_full.local_length(); lid2++)
        {
          const int gid2 = dofrowmapred->gid(lid2);
          const double val2 = dQdu_full.local_values_as_span()[lid2];
          const double actmatentry = tfaclhs * val * val2;
          if (abs(actmatentry) > 1e-15) impedancetbcsysmat_->assemble(actmatentry, gid, gid2);
        }
      }
    }

    impedancetbcsysmat_->complete();
    //  std::cout<<__FILE__<<__LINE__<<*((std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(impedancetbcsysmat_))->EpetraMatrix())<<std::endl;

    // NOTE: since the building of the linearization is very expensive and since it only
    // changes due to the deformation of the BCs domain (the linearization is independent
    // of the velocities velaf_!), we simply scale the linearization by the change of the
    // domain area
  }


  // ---------------------------------------------------------------------//
  // ---------------------------------------------------------------------//
  // Add traction and linearisation to fluid                              //
  // ---------------------------------------------------------------------//
  // ---------------------------------------------------------------------//

  // Apply traction vector to fluid residual
  residual.update(1.0, *impedancetbc_, 1.0);

  // Add linearisation to fluid sysmat
  sysmat.add(*impedancetbcsysmat_, false, Q_np_fac, 1.0);

  return;
}  // FluidImplicitTimeInt::outflow_boundary

/*----------------------------------------------------------------------*
 |  Update flowrate and pressure vector                       Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceBc::time_update_impedance(const double time, const int condid)
{
  const double actpressure = p_np_;

  // NOTE:: time may be a huge number (e.g. in case of multiscale in time AC-FS3I).
  // Hence, the naive approach fmod(time+1e.8,period_)-1e-8 < 1e-8) is not suited.
  // This one is unintuitive but gives same results and is save!
  if (fabs((fmod(time + period_ / 2, period_) - period_ / 2)) / time <
      1e-12)  // iff we are at the beginning of a new period
  {
    w_krelerror_ = fabs((actpressure - p_0_) / actpressure);
    p_0_ = actpressure;

    if (myrank_ == 0)
    {
      std::cout << "Impedance condition Id: " << condid << ", Windkessel pressure: " << actpressure
                << std::endl;
      //      std::cout<<"Impedance condition Id: "<<condid<<", Relative windkessel error:
      //      "<<WKrelerror_<<std::endl;
    }
  }

  // shift time step
  p_n_ = p_np_;
  q_n_ = q_np_;

  return;
}  // FluidImplicitTimeInt::outflow_boundary

/*----------------------------------------------------------------------*
 |  Restart writing                                          Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceBc::write_restart(
    Core::IO::DiscretizationWriter& output, const int condnum)
{
  // condnum contains the number of the present condition
  // condition Id numbers must not change at restart!!!!

  std::stringstream stream1, stream2, stream3, stream4, stream5, stream6;

  stream1 << "P_np" << condnum;
  // write the pressure at time step n+1
  output.write_double(stream1.str(), p_np_);

  stream6 << "Q_np" << condnum;
  // write the flux at time step n+1
  output.write_double(stream6.str(), q_np_);

  stream2 << "WKrelerror" << condnum;
  output.write_double(stream2.str(), w_krelerror_);

  stream3 << "P_0" << condnum;
  output.write_double(stream3.str(), p_0_);

  stream4 << "P_n" << condnum;
  // write the input pressure at time step n
  output.write_double(stream4.str(), p_n_);

  stream5 << "Q_n" << condnum;
  // write the flux pressure at time step n
  output.write_double(stream5.str(), q_n_);

  return;
}

/*----------------------------------------------------------------------*
 |  Restart reading                                          Thon 07/16 |
 *----------------------------------------------------------------------*/
void FLD::Utils::FluidImpedanceBc::read_restart(
    Core::IO::DiscretizationReader& reader, const int condnum)
{
  std::stringstream stream1, stream2, stream3, stream4, stream5, stream6;

  stream1 << "P_np" << condnum;
  // read the pressure at time step n+1
  p_np_ = reader.read_double(stream1.str());

  stream6 << "Q_np" << condnum;
  // read the pressure at time step n+1
  q_np_ = reader.read_double(stream6.str());

  // read in pressure difference
  stream3 << "P_0" << condnum;
  p_0_ = reader.read_double(stream3.str());

  stream4 << "P_n" << condnum;
  // read the input pressure at time step n
  p_n_ = reader.read_double(stream4.str());

  stream5 << "Q_n" << condnum;
  // read the flux pressure at time step n
  q_n_ = reader.read_double(stream5.str());

  // get pressure difference and pressure of last period
  if (abs(w_krelerror_ - 1.0) <
      1e-14)  // if we just initialized the class (in context of AC-FS3I this is not guaranteed!)
  {
    // read in pressure difference
    stream2 << "WKrelerror" << condnum;
    w_krelerror_ = reader.read_double(stream2.str());
  }
  else
  {
    w_krelerror_ = 0.0;
  }

  return;
}

/*----------------------------------------------------------------------*
 | Area calculation                                          Thon 07/16 |
 *----------------------------------------------------------------------*/
double FLD::Utils::FluidImpedanceBc::area(const int condid)
{
  // fill in parameter list for subsequent element evaluation
  // there's no assembly required here
  Teuchos::ParameterList eleparams;
  eleparams.set<FLD::BoundaryAction>("action", FLD::calc_area);
  eleparams.set<double>("area", 0.0);
  // we have to set some dummy values here:
  eleparams.set<double>("viscosity", 0.0);
  eleparams.set<double>("density", 0.0);

  discret_->evaluate_condition(eleparams, "ImpedanceCond", condid);

  double actarea = eleparams.get<double>("area");

  // get total area in parallel case
  double pararea = 0.0;
  pararea = Core::Communication::sum_all(actarea, discret_->get_comm());

  //  if (myrank_ == 0)
  //    std::cout << "Impedance condition Id: " << condid << ", Area: " << pararea << std::endl;

  return pararea;
}  // FluidImplicitTimeInt::Area

FOUR_C_NAMESPACE_CLOSE
