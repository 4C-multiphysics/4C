// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEGACY_ENUM_DEFINITIONS_MATERIALS_HPP
#define FOUR_C_LEGACY_ENUM_DEFINITIONS_MATERIALS_HPP

#include "4C_config.hpp"

#include <string_view>

FOUR_C_NAMESPACE_OPEN

namespace Core::Materials
{
  /// type of material
  enum MaterialType
  {
    m_none,                                 ///< undefined
    m_0d_maxwell_acinus,                    ///< 0D acinar Maxwell material
    m_0d_maxwell_acinus_doubleexponential,  ///< 0D acinar Maxwell DoubleExponential material
    m_0d_maxwell_acinus_exponential,        ///< 0D acinar Maxwell Exponential material
    m_0d_maxwell_acinus_neohookean,         ///< 0D acinar Maxwell NeoHookean material
    m_0d_maxwell_acinus_ogden,              ///< 0D acinar Maxwell Ogden material
    m_aaaneohooke,                ///< quasi Neo-Hooke material for aneurysmatic artery wall
    m_beam_elast_hyper_generic,   ///< material law for a beam: hyperelastic stored energy function
    m_beam_reissner_elast_hyper,  ///< material parameters for a Simo-Reissner beam: hyperelastic
                                  ///< stored energy function
    m_beam_reissner_elast_hyper_bymodes,   ///< material parameters for a Simo-Reissner beam:
                                           ///< hyperelastic stored energy function, specified for
                                           ///< individual deformation modes
    m_beam_reissner_elast_plastic,         ///< material for a Simo-Reissner beam: elasto-plastic
    m_beam_kirchhoff_elast_hyper,          ///< material parameters for a Kirchhoff-Love beam:
                                           ///< hyperelastic stored energy function
    m_beam_kirchhoff_elast_hyper_bymodes,  ///< material parameters for a Kirchhoff-Love beam:
                                           ///< hyperelastic stored energy function, specified for
                                           ///< individual deformation modes
    m_beam_kirchhoff_torsionfree_elast_hyper,  ///< material parameters for a torsion-free,
                                               ///< isotropic Kirchhoff-Love beam: hyperelastic
                                               ///< stored energy function
    m_beam_kirchhoff_torsionfree_elast_hyper_bymodes,  ///< material parameters for a
                                                       ///< torsion-free, isotropic Kirchhoff-Love
                                                       ///< beam: hyperelastic stored energy
                                                       ///< function, specified for individual
                                                       ///< deformation modes
    m_shell_kirchhoff_love,                            ///< material for a Kirchhoff-Love shell
    m_carreauyasuda,      ///< fluid with nonlinear viscosity according to Carreau-Yasuda
    m_cnst_art,           ///< 1D_Artery constant material properties
    m_constraintmixture,  ///< growth and remodeling of arteries
    m_crosslinkermat,     ///< material for crosslinker in biopolymer networks
    m_crystplast,         ///< crystal plasticity
    m_elasthyper,         ///< collection of hyperelastic materials
    m_elchmat,            ///< material for porous separators
    m_elchphase,          ///< material for porous phase inside porous separator
    m_electrode,          ///< electrode material
    m_elpldamage,         ///< elasto-plastic material with von Mises plasticity and damage
    m_fluid,              ///< fluid
    m_fluid_linear_density_viscosity,  ///< linear law (pressure-dependent) for the density and
                                       ///< the viscosity
    m_fluid_murnaghantait,             ///< weakly compressible fluid according to Murnaghan-Tait
    m_fluid_weakly_compressible,       ///< weakly compressible fluid
    m_fluidporo,                       ///< darcy fluid for poroelasticity problems
    m_fluidporo_singlephase,    ///< single phase material for multiphase flow through porous medium
    m_fluidporo_singlevolfrac,  ///< single volume fraction material for multiphase flow through
                                ///< porous medium
    m_fluidporo_volfracpressure,  ///< volume fraction pressure material for multiphase flow
                                  ///< through porous medium
    m_fluidporo_singlereaction,   ///< single phase material for multiphase flow through porous
                                  ///< medium
    m_fluidporo_volfrac_pressure_blood_lung,  ///<  volume fraction material, with the pressure as
                                              ///<  primary variable and a closing relation defined
                                              ///<  for the vascular units in the lungs
    m_fluidporo_multiphase,  ///< collection of single phase materials for multiphase flow through
                             ///< porous medium
    m_fluidporo_multiphase_reactions,   ///< collection of single phase materials for reavcitve
                                        ///< multiphase flow through porous medium
    m_fluidporo_phasedof_pressure,      ///< pressure DOF for multiphase flow through porous medium
    m_fluidporo_phasedof_saturation,    ///< saturation DOF for multiphase flow through porous
                                        ///< medium
    m_fluidporo_phasedof_diffpressure,  ///< diffrenetial pressure DOF for multiphase flow through
                                        ///< porous medium
    m_fluidporo_phaselaw_linear,        ///< linear pressure-saturation relationship
    m_fluidporo_phaselaw_tangent,       ///< tangent pressure-saturation relationship
    m_fluidporo_phaselaw_constraint,    ///< the saturation constraint
    m_fluidporo_phaselaw_byfunction,    ///< pressure-saturation relationship defined by functions
                                        ///< in input file
    m_fluidporo_relpermeabilitylaw_constant,  ///< permeability law for constant permeability in
                                              ///< porous multiphase medium
    m_fluidporo_relpermeabilitylaw_exp,       ///< permeability law for permeability depending on
                                         ///< saturations^exponenent in porous multiphase medium
    m_fluidporo_viscositylaw_constant,  ///< viscosity law for constant viscosity in porous
                                        ///< multiphase medium
    m_fluidporo_viscositylaw_celladh,   ///< viscosity law modelling cell adherence
    m_sc_dep_interp,  ///< integration point based and scalar dependent interpolation between to
                      ///< materials
    m_membrane_elasthyper,       ///< collection of hyperelastic materials for membranes
    m_membrane_activestrain,     ///< active strain membrane material for gastric electromechanics
    m_growthremodel_elasthyper,  ///< growth and remodeling
    m_herschelbulkley,           ///< fluid with nonlinear viscosity according to Herschel-Bulkley
    m_ion,                       ///< properties of an ion species in an electrolyte solution
    m_linelast1D,                ///< linear elastic material in one direction
    m_linelast1D_growth,         ///< linear elastic material including growth in one direction
    m_lubrication,               ///< lubrication material
    m_lubrication_law_constant,  ///< lubrication material with constant viscosity
    m_lubrication_law_barus,     ///< lubrication material with Barus viscosity
    m_lubrication_law_roeland,   ///< lubrication material with Roeland viscosity
    m_matlist,             ///< collection of single materials (used for scalar transport problems)
    m_matlist_reactions,   ///< collection of single materials (used for scalar transport problems)
                           ///< and collection of reactions (used for advanced reaction elements)
    m_matlist_chemotaxis,  ///< collection of single materials (used for scalar transport
                           ///< problems) and collection of chemotactic pairs (used for chemotaxis
                           ///< elements)
    m_matlist_chemoreac,   ///< collection of single materials (used for scalar transport problems)
                           ///< and collection of chemotactic pairs AND reactions
    m_mixture,             ///< material for solid mixtures with homogenized stress response
    m_modpowerlaw,         ///< fluid with nonlinear viscosity according to a modified power law
    m_multiplicative_split_defgrad_elasthyper,  ///< deformation gradient is split
                                                ///< multiplicatively in elastic and inelastic
                                                ///< parts
    m_muscle_combo,                             ///< Combo generalized active strain muscle material
    m_muscle_giantesio,                         ///< Giantesio active strain muscle material
    m_muscle_weickenmeier,    ///< Weickenmeier generalized active strain muscle material
    m_myocard,                ///< anisotropic electrophysical model of heart tissue
    m_newman,                 ///< properties of an ion species in an electrolyte solution
    m_newman_multiscale,      ///< properties of an ion species in an electrolyte solution for
                              ///< multi-scale approach
    m_particle_sph_fluid,     ///< particle material for SPH fluid
    m_particle_sph_boundary,  ///< particle material for SPH boundary
    m_particle_dem,           ///< particle material for DEM
    m_particle_wall_dem,      ///< particle wall material for DEM
    m_particle_pd,            ///< particle material for PD
    m_pldruckprag,       ///< Plastic linear elastic St.Venant Kirchhoff / Drucker Prager plasticity
    m_plgtn,             ///< Plastic linear elastic St.Venant Kirchhoff / GTN plasticity
    m_plelasthyper,      ///< general hyperelastic material for finite strain von-Mises plasticity
                         ///< using a semi-smooth Newton strategy (only in combination with such
                         ///< elements!)
    m_plelasthyperVCU,   ///< general hyperelastic material for finite strain von-Mises plasticity
                         ///< using a variational constitutive update
    m_pllinelast,        ///< linear elasticity (St. Venant Kirchhoff) and von Mises plasticity
    m_plnlnlogneohooke,  ///< Neo-Hooke elasticity with logarithmic finite strain von Mises
                         ///< plasticity
    m_poro_law_constant,             ///< constant porosity
    m_poro_law_linear,               ///< linear law for porosity
    m_poro_law_logNeoHooke_Penalty,  ///< neo hookeian like law for porosity + penalty term
    m_poro_law_incompr_skeleton,     ///< porosity law for incompressible skeleton phase
    m_poro_law_linear_biot,          ///< porosity law for linear biot law
    m_poro_law_density_dependent,    ///< porosity law for density dependence
    m_poro_densitylaw_constant,  ///< density law for constant density in porous multiphase medium
    m_poro_densitylaw_exp,       ///< density law for pressure dependent exponential function
    m_iterative_prestress,  ///< A general material wrapper adding iterative prestressing to any
                            ///< material
    m_scatra,               ///< scalar transport material
    m_scatra_in_fluid_porofluid_pressure_based,    ///< scalar transport material for pressure-based
                                                   ///< porous flow (species in fluid)
    m_scatra_in_volfrac_porofluid_pressure_based,  ///< scalar transport material for pressure-based
                                                   ///< porous flow (species in volume fraction)
    m_scatra_in_solid_porofluid_pressure_based,    ///< scalar transport material for pressure-based
                                                   ///< porous flow (species in solid)
    m_scatra_as_temperature_porofluid_pressure_based,  ///< scalar transport material for
                                                       ///< pressure-based porous flow (temperature)
    m_scatra_multiscale,        ///< scalar transport material for multi-scale approach
    m_scatra_reaction_poroECM,  ///< reaction definition and parameters for reaction model in
                                ///< porous ECM
    m_scatra_reaction,          ///< reaction definition and parameters
    m_scatra_chemotaxis,        ///< chemotaxis definition parameters
    m_scl,                      ///< material for modeling space charge layers in solid electrolytes
    m_soret,   ///< material for heat transport due to Fourier-type thermal conduction and the
               ///< Soret effect
    m_spring,  ///< elastic spring (translational or rotational)
    m_struct_multiscale,      ///<  structural microscale approach
    m_structporo,             ///< wrapper material for poroelasticity (structure)
    m_structpororeaction,     ///< wrapper material for poroelasticity (structure)
    m_structpororeactionECM,  ///< wrapper material for poroelasticity (structure)
    m_superelast,             ///< Superelastic material behaviour of shape memory alloys
    m_stvenant,               ///< St.Venant Kirchhoff material
    m_orthostvenant,          ///< St.Venant Kirchhoff material with orthotropy
    m_sutherland,             ///< material with temperature dependence according to Sutherland law
    m_thermo_fourier,  ///< (Possibly anisotropic) linear implementation of Fourier's law of heat
                       ///< conduction
    m_thermoplhyperelast,   ///< Temperature-dependent hyperelasticity and von Mises plasticity
    m_thermopllinelast,     ///< Temperature-dependent Hooke elasticity and von Mises plasticity
    m_thermostvenant,       ///< St.Venant Kirchhoff material with temperature
    m_viscoanisotropic,     ///< Viscous Anisotropic Fiber Material
    m_viscoelasthyper,      ///< viscohyperelastic material
    m_visconeohooke,        ///< Viscous NeoHookean Material
    m_vp_no_yield_surface,  ///< visco-plastic finite strain material law without yield surface
    m_vp_robinson,          ///< Robinson's visco-plastic material
    mes_anisoactivestress_evolution,  ///< anisotropic single fiber summand with active stress
                                      ///< computed through a simplified Bestel-Clement-Sorine
                                      ///< model
    mes_coup1pow,       ///< general power hyperelastic potential summand for invariant I
    mes_coup2pow,       ///< general power hyperelastic potential summand for invariant II
    mes_coup3pow,       ///< general power hyperelastic potential summand for invariant III
    mes_coup13apow,     ///< hyperelastic potential summand for multiplicative coupled invariants I1
                        ///< and I3
    mes_coupanisoexpo,  ///< anisotropic exponential single fiber summand
    mes_coupanisoexposhear,         ///< summand for exponential shear behavior between two fibers
    mes_coupanisoexpoactive,        ///< anisotropic active fiber contribution
    mes_coupanisoexpotwocoup,       ///< anisotropic two fiber summand with
                                    ///< coupling between the fibers
    mes_coupanisoneohooke,          ///< anisotropic Neo-Hooke single fiber summand
    mes_coupanisoneohooke_varprop,  ///< anisotropic Neo-Hooke single fiber summand with
                                    ///< space-time variable coefficient
    mes_coupanisopow,               ///< anisotropic pow-like single fiber summand
    mes_couptransverselyisotropic,  ///< transversely isotropic hyperelastic summand
    mes_coupblatzko,                ///< Blatz and Ko  material as hyperelastic potential summand
    mes_coupexppol,                 ///< compressible, isotropic material law for soft tissue
    mes_couplogmixneohooke,  ///< logarithmic Neo-Hooke material as hyperelastic potential summand
    mes_couplogneohooke,     ///< logarithmic Neo-Hooke material as hyperelastic potential summand
    mes_coupmooneyrivlin,    ///< Mooney-Rivlin  material as hyperelastic potential summand
    mes_coupmyocard,         ///< isotropic viscous contribution of myocardial matrix
    mes_coupneohooke,        ///< Neo-Hooke material as hyperelastic potential summand
    mes_coupSVK,             ///< Saint-Venant-Kirchhoff material
    mes_coupsimopister,      ///< Simo-Pister type material
    mes_coupvarga,           ///< isotropic Varga material
    mes_genmax,              ///< viscous contribution according to SLS-Model,
    mes_generalizedgenmax,   ///< viscoelastic branches of the generalized Maxwell Model
    mes_fract,               ///< viscous contribution according to FSLS-Model,
    mes_viscopart,           ///< viscous part of generalized Maxwell Model
    mes_viscobranch,         ///< viscoelastic branch of generalized Maxwell Model
    mes_iso1pow,          ///< isochoric general power hyperelastic potential summand for modinv I
    mes_iso2pow,          ///< isochoric general power hyperelastic potential summand for modiinv II
    mes_isoanisoexpo,     ///< isochoric anisotropic single fiber material
    mes_isoexpopow,       ///< isochoric exponential hyperelastic potential summand
    mes_isomooneyrivlin,  ///< isochoric Mooney-Rivlin hyperelastic potential summand
    mes_isomuscleblemker,  ///< isochoric Blemker active stress muscle material
    mes_isoneohooke,       ///< isochoric Neo-Hooke hyperelastic potential summand
    mes_isoogden,          ///< isochoric one-term Ogden hyperelastic potential summand
    mes_isoratedep,        ///< isotropic isochoric frequency dependent viscous potential summand
    mes_isotestmaterial,   ///< material to test the elasthyper-toolbox
    mes_isovarga,          ///< isotropic isochoric Varga material
    mes_isoyeoh,           ///< isochoric Yeoh hyperelastic potential summand
    mes_remodelfiber,      ///< general fiber material for remodeling
    mes_vologden,          ///< Ogden volumetric part of the  hyperelastic potential summand
    mes_volpenalty,        ///< Penalty volumetric part of the  hyperelastic potential summand
    mes_volsussmanbathe,   ///< volumetric SussmanBathe hyperelastic potential summand
    mes_volpow,            ///< volumetric power law hyperelastic potential summand
    mes_structuraltensorstratgy,  ///< structural tensor in anisotropic materials
    mfi_lin_scalar_aniso,  ///< volume change due to (anisotropic) inelastic deformation gradient
                           ///< is a linear function of the scalar (in material configuration)
                           ///< that causes the volume change
    mfi_lin_scalar_iso,    ///< volume change due to (isotropic) inelastic deformation gradient is a
                           ///< linear function of the scalar (in material configuration) that
                           ///< causes the volume change
    mfi_lin_temp_iso,      ///< volume change due to isotropic inelastic deformation gradient is a
                           ///< linear function of the temperature
    mfi_no_growth,   ///< material with no volume change, i.e. inelastic deformation gradient is an
                     ///< identity tensor
    mfi_time_funct,  ///< growth evaluated by a function
    mfi_poly_intercal_frac_aniso,  ///< volume change due to (anisotropic) inelastic deformation
                                   ///< gradient is a polynomial function of the intercalation
                                   ///< fraction that causes the volume change
    mfi_poly_intercal_frac_iso,    ///< volume change due to (isotropic) inelastic deformation
                                   ///< gradient is a polynomial function of the intercalation
                                   ///< fraction that causes the volume change
    mfi_transv_isotrop_elast_viscoplast,  /// < versatile transversely-isotropic / isotropic
                                          /// material model for viscoplasticity, suitable for
                                          /// finite strain deformations
    mvl_reformulated_Johnson_Cook,        ///< reformulation of the Johnson-Cook
                                          ///< viscoplasticity law, as presented in
                                          ///< Mareau (Mech. Mat. 143, 2020)
    mix_growth_strategy_isotropic,        ///< Isotropic growth law for growth remodel mixture rule
    mix_growth_strategy_anisotropic,  ///< Anisotropic growth law for growth remodel mixture rule
    mix_growth_strategy_stiffness,    ///< Growth modeled as an expansion of the entire cell (growth
                                      ///< happens mainly in the direction of the smallest
                                      ///< stiffness)
    mix_rule_function,                ///< Function rule for the mixture model
    mix_rule_simple,                  ///< Simple rule for the mixture model
    mix_rule_growthremodel,           ///< Homogenized constrained mixture
    mix_prestress_strategy_cylinder,  ///< Prestress strategy for a cylinder
    mix_prestress_strategy_iterative,    ///< Iterative prestress strategy for any geometry
    mix_prestress_strategy_prescribed,   ///< Predefined prestretch field for any geometry
    mix_elasthyper,                      ///< Elast Hyper toolbox for constituents
    mix_elasthyper_damage,               ///< Elast hyper toolbox with temporal damage
    mix_elasthyper_elastin_membrane,     ///< Elast Hyper toolbox with temporal damage and 2D
                                         ///< membrane material
    mix_full_constrained_mixture_fiber,  ///< A quasi 1D fiber that growths with the full
                                         ///< constrained mixture model
    mix_remodelfiber_expl,  ///< Fiber that remodels with homogenized constrained mixture model
                            ///< with explicit solution of the growth and remodel equations
    mix_remodelfiber_impl,  ///< Fiber that remodels with homogenized constrained mixture model
                            ///< with implicit solution of the growth and remodel equations
    mix_remodelfiber_material_exponential,  ///< material for a remodel fiber with exponential
                                            ///< strain energy function
    mix_remodelfiber_material_exponential_active,  ///< material for a remodel fiber with
                                                   ///< exponential strain energy and an active
                                                   ///< contribution
    mix_solid_material                             ///< Solid material for constituents
  };


  std::string_view to_string(MaterialType material_type);
}  // namespace Core::Materials


FOUR_C_NAMESPACE_CLOSE

#endif
