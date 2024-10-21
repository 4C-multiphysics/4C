#ifndef FOUR_C_SO3_PRESTRESS_SERVICE_HPP
#define FOUR_C_SO3_PRESTRESS_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Prestress
{
  /*!
   * \brief Returns the type of the prestress algorithm stored in the parameters of structural
   * dynamics
   *
   * \return Inpar::Solid::PreStress
   */
  static inline Inpar::Solid::PreStress get_type()
  {
    static Inpar::Solid::PreStress pstype = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
        Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");

    return pstype;
  }

  /*!
   * \brief Returns the prestress time stored in the parameters of structural dynamics
   *
   * \return double
   */
  static inline double get_prestress_time()
  {
    static double pstime =
        Global::Problem::instance()->structural_dynamic_params().get<double>("PRESTRESSTIME");

    return pstime;
  }

  /*!
   * \brief Returns whether MULF is set for prestressing in the parameters of structural dynamics.
   * This method does not ensure that MULF is actually active
   *
   * \return true MULF is set in input file
   * \return false MULF is not set in input file
   */
  static inline bool is_mulf() { return get_type() == Inpar::Solid::PreStress::mulf; }

  /*!
   * \brief Returns whether material iterative prestressing is set in the parameters of structural
   * dynamics. This method does not ensure that prestressing is actually active
   *
   * \return true material iterative prestressing is set in input file
   * \return false material iterative prestressing is not set in input file
   */
  static inline bool is_material_iterative()
  {
    return get_type() == Inpar::Solid::PreStress::material_iterative;
  }

  /*!
   * \brief Returns whether MULF is set for prestressing as the given prestress type.
   * This method does not ensure that MULF is actually active
   *
   * \param pstype Prestress type that is used
   * \return true MULF is set in input file
   * \return false MULF is not set in input file
   */
  static inline bool is_mulf(Inpar::Solid::PreStress pstype)
  {
    return pstype == Inpar::Solid::PreStress::mulf;
  }

  /*!
   * \brief Returns whether material iterative prestressing is set as the given prestress type.
   * This method does not ensure that prestressing is actually active
   *
   * \param pstype Prestress type that is used
   * \return true material iterative prestressing is set in input file
   * \return false material iterative prestressing is not set in input file
   */
  static inline bool is_material_iterative(Inpar::Solid::PreStress pstype)
  {
    return pstype == Inpar::Solid::PreStress::material_iterative;
  }


  /*!
   * \brief Returns whether no prestressing is set in the parameters of
   * structural dynamics.
   *
   * \return true No prestressing is set in the input file
   * \return false Prestressing is set in the input file
   */
  static inline bool is_none() { return get_type() == Inpar::Solid::PreStress::none; }


  /*!
   * \brief Returns whether no prestressing is set in the given parameter.
   *
   * \param pstype Prestress type that is used
   * \return true No prestressing is set in the input parameter
   * \return false Prestressing is set in the input parameter
   */
  static inline bool is_none(Inpar::Solid::PreStress pstype)
  {
    return pstype == Inpar::Solid::PreStress::none;
  }

  /*!
   * \brief Returns whether any prestressing is set in the parameters of
   * structural dynamics.
   *
   * \return true Prestressing is set in the input file
   * \return false No prestressing is set in the input file
   */
  static inline bool is_any()
  {
    return Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
               Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS") !=
           Inpar::Solid::PreStress::none;
  }

  /*!
   * \brief Returns whether prestressing is set in the given parameter.
   *
   * \param pstype Prestress type that is used
   * \return true Prestressing is set in the input parameter
   * \return false No prestressing is set in the input parameter
   */
  static inline bool is_any(Inpar::Solid::PreStress pstype)
  {
    return pstype != Inpar::Solid::PreStress::none;
  }

  /*!
   * \brief Returns true if any prestressing method is currently active with the parameters of
   * strtuctural dynamics.
   *
   * \param currentTime Current time of the simulation
   * \return true Any prestressing method is active
   * \return false No prestressing method is active
   */
  static inline bool is_active(const double currentTime)
  {
    Inpar::Solid::PreStress pstype = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
        Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
    const double pstime =
        Global::Problem::instance()->structural_dynamic_params().get<double>("PRESTRESSTIME");
    return pstype != Inpar::Solid::PreStress::none && currentTime <= pstime + 1.0e-15;
  }

  /*!
   * \brief Returns true if any prestressing method is currently active with the given parameters.
   *
   * \param currentTimeCurrent time of the simulation
   * \param pstype Prestress type that is used
   * \param pstime Prestress time that is used
   * \return true Any prestressing method is active
   * \return false No prestressing method is active
   */
  static inline bool is_active(
      const double currentTime, Inpar::Solid::PreStress pstype, const double pstime)
  {
    return pstype != Inpar::Solid::PreStress::none && currentTime <= pstime + 1.0e-15;
  }

  /*!
   * \brief Returns true if MULF prestressing method is currently active with the parameters of
   * strtuctural dynamics.
   *
   * \param currentTimeCurrent time of the simulation
   * \return true MULF prestressing method is active
   * \return false MULF prestressing method is active
   */
  static inline bool is_mulf_active(const double currentTime)
  {
    return is_mulf() && currentTime <= get_prestress_time() + 1.0e-15;
  }

  /*!
   * \brief Returns true if MULF prestressing method is currently active with the given
   * parameters.
   *
   * \param currentTimeCurrent time of the simulation
   * \param pstype Prestress type that is used
   * \param pstime Prestress time that is used
   * \return true MULF prestressing method is active
   * \return false MULF prestressing method is active
   */
  static inline bool is_mulf_active(
      const double currentTime, Inpar::Solid::PreStress pstype, const double pstime)
  {
    return is_mulf(pstype) && currentTime <= pstime + 1.0e-15;
  }

}  // namespace Prestress

FOUR_C_NAMESPACE_CLOSE

#endif