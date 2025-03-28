#!/usr/bin/env wolframscript

# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

(* ::Package:: *)

(* ::Title:: *)
(*Create unit test cases for beam to solid volume mesh tying mortar pairs*)


<<FEMUtils`
<<NumericalDifferentialEquationAnalysis`


(* ::Section:: *)
(*Matrix to Core::LinAlg::TMatrix*)


matrixToFourC[var_,name_]:=Module[{allStringsWithZeros, stringList},
  allStringsWithZeros = Table[
    If[var[[i,j]]==0, 0,
        ToString[StringForm["``(``,``)=``;",name,i-1,j-1,NumberForm[N[var[[i,j]]],16,ScientificNotationThreshold->{-20,20}]]]
    ],
    {i,Dimensions[var][[1]]},{j,Dimensions[var][[2]]}];

  stringList=DeleteCases[Flatten[allStringsWithZeros],0];
  StringRiffle[stringList,"\n"]
]
vectorToFourC[var_,name_]:=Module[{allStringsWithZeros, stringList},
  allStringsWithZeros = Table[
    If[var[[i]]==0, 0,
        ToString[StringForm["``(``)=``;",name,i-1,NumberForm[N[var[[i]]],16,ScientificNotationThreshold->{-20,20}]]]
    ],
    {i,Length[var]}];

  stringList=DeleteCases[Flatten[allStringsWithZeros],0];
  StringRiffle[stringList,"\n"]
]


(* ::Section:: *)
(*Get FEM interpolation of field variables.*)


(* ::Text:: *)
(*Get solid position vector.*)


xSolid:={
	fieldFunction[solidType]/.posDof->posSx[n],
	fieldFunction[solidType]/.posDof->posSy[n],
	fieldFunction[solidType]/.posDof->posSz[n]
	};
qSolid:=Flatten[Transpose[
  {
    discreteValues[solidType]/.posDof->posSx[n],
    discreteValues[solidType]/.posDof->posSy[n],
    discreteValues[solidType]/.posDof->posSz[n]
  }]];

(*Add random offset to solid coordinates. This offset is alyays the same because of the seed.*)
replaceSolidRef:=(
  SeedRandom[666];
  temp=Flatten[nodalCoordiantes3D[solidType]];
  temp=Thread[qSolid->temp + RandomReal[{-0.1,0.1},Length[temp]]]/.n->0
  )


(* ::Text:: *)
(*Get beam position vector.*)


beamType="hermite2";
xBeam:={
	fieldFunction[beamType]/.posDof->posBx[n]/.tanDof->tanBx[n],
	fieldFunction[beamType]/.posDof->posBy[n]/.tanDof->tanBy[n],
	fieldFunction[beamType]/.posDof->posBz[n]/.tanDof->tanBz[n]
	}/.lenDof->beamLength;


(* ::Text:: *)
(*Reference position of beam.*)


(*The element length is taken from the 4C simulation*)
replaceBeamRef:={
    beamLength->0.6192043571449604711,
    posBx[0][1]->0.15,
    posBy[0][1]->0.2,
    posBz[0][1]->0.3,
    tanBx[0][1]->0.5773502691896255,
    tanBy[0][1]->0.5773502691896258,
    tanBz[0][1]->0.577350269189626,
    posBx[0][2]->0.65,
    posBy[0][2]->0.1,
    posBz[0][2]->0.1,
    tanBx[0][2]->0.801783725737273,
    tanBy[0][2]->-0.5345224838248488,
    tanBz[0][2]->0.2672612419124244};

qBeam:=Drop[replaceBeamRef[[All,1]],1]/.x_[0]:>x[n];


(* ::Text:: *)
(*Get interpolation of Lagrange multiplicator.*)


lambda:={
	fieldFunction[lambdaType]/.posDof->lambdax,
	fieldFunction[lambdaType]/.posDof->lambday,
	fieldFunction[lambdaType]/.posDof->lambdaz
	};
qLambda:=Flatten[Transpose[
  {
    discreteValues[lambdaType]/.posDof->lambdax,
    discreteValues[lambdaType]/.posDof->lambday,
    discreteValues[lambdaType]/.posDof->lambdaz
  }]];


(* ::Section:: *)
(*Project beam to solid*)


projectPointToSolid[point_]:=Module[{equations},
(*Set the point equal to a point in the solid in the reference configuration.*)
equations=Thread[point==xSolid/.n->0/.replaceSolidRef];

(*Solve the equations.*)
FindRoot[equations,{{xi1,0},{xi2,0},{xi3,0}},PrecisionGoal->20]
]


(* ::Section:: *)
(*Numerical integration along beam element.*)


nGP=6;
gaussSupportsWeights=GaussianQuadratureWeights[nGP,-1,1];
integrateLocalN[val_]:=Sum[(Norm[D[xBeam,xi]]/.n->0/.xi->item[[1]])item[[2]]val[item[[1]]],{item,gaussSupportsWeights}];


(* ::Section:: *)
(*Generate mesh tying potential*)


SetOptions[SelectedNotebook[], PrintPrecision -> 16]
gaussPointPotential[xii_]:=Module[{pointInSolid},
(*Current point on the beam in the solid.*)
pointInSolid=projectPointToSolid[xBeam/.xi->xii/.n->0/.replaceBeamRef];

(*Calcualte the meshtying potential.*)
lambda . (xBeam-xSolid)/.xi->xii/.pointInSolid
]

(*Calculate the element matrices.*)
\[CapitalPi]mt:=integrateLocalN[gaussPointPotential]/.replaceBeamRef;
gmt:=D[\[CapitalPi]mt,{qLambda}];


(* ::Section:: *)
(*Create the results for the test case*)


getTestString:=(
(*Calculate the matrices*)
DD=D[gmt,{qBeam}];
MM=-D[gmt,{qSolid}];

(*Calcualte the scaling matrix.*)
f\[Lambda][xii_]:=lambda/.xi->xii;
int\[Lambda]=integrateLocalN[f\[Lambda]]/.replaceBeamRef;
kappa=DeleteCases[Flatten[Transpose[D[int\[Lambda],{qLambda}]]],0];

(*Output the results.*)
StringRiffle[{
  "// Positional DOFs of the solid.",
  vectorToFourC[replaceSolidRef[[All,2]],"q_solid.element_position_"],
  "\n// Results for D.",
  matrixToFourC[DD,"result_local_D"],
  "\n// Results for M.",
  matrixToFourC[MM,"result_local_M"],
  "\n// Results for Kappa.",
  vectorToFourC[kappa,"result_local_kappa"]},"\n"]
  )


(* ::Section:: *)
(*Create the function for the test case*)


functionStringPre="  /**
   * \\brief Test a non straight beam in a mathematica_solid_type element, with mathematica_mortar_type mortar shape functions.
   */
  TEST_F(BeamToSolidVolumeMeshtyingPairMortarTest, TestBeamToSolidMeshtyingMortarHermite2mathematica_solid_Typemathematica_mortar_Type)
  {
    // Element types.
    typedef GEOMETRYPAIR::t_hermite beam_type;
    typedef GEOMETRYPAIR::t_mathematica_solid_type solid_type;
    typedef GEOMETRYPAIR::t_mathematica_mortar_type lambda_type;

    // Create the mesh tying mortar pair.
    BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type, lambda_type>
        contact_pair = BeamInteraction::BeamToSolidVolumeMeshtyingPairMortar<beam_type, solid_type,
            lambda_type>();

    // Definition of variables for this test case.
    GEOMETRYPAIR::ElementData<beam_type, double> q_beam;
    GEOMETRYPAIR::ElementData<solid_type, double> q_solid;
    Core::LinAlg::Matrix<9, 1> q_beam_rot;
    Core::LinAlg::SerialDenseMatrix local_D;
    Core::LinAlg::SerialDenseMatrix local_M;
    Core::LinAlg::SerialDenseVector local_kappa;

    // Matrices for the results.
    Core::LinAlg::Matrix<lambda_type::n_dof_, beam_type::n_dof_> result_local_D(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, solid_type::n_dof_> result_local_M(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<lambda_type::n_dof_, 1> result_local_kappa(Core::LinAlg::Initialization::zero);

    // Define the geometry of the two elements.
    q_beam.shape_function_data_.ref_length_ = 0.6192043571449604711;
    q_beam.element_position_(0) = 0.15;
    q_beam.element_position_(1) = 0.2;
    q_beam.element_position_(2) = 0.3;
    q_beam.element_position_(3) = 0.5773502691896255;
    q_beam.element_position_(4) = 0.5773502691896258;
    q_beam.element_position_(5) = 0.577350269189626;
    q_beam.element_position_(6) = 0.65;
    q_beam.element_position_(7) = 0.1;
    q_beam.element_position_(8) = 0.1;
    q_beam.element_position_(9) = 0.8017837257372733;
    q_beam.element_position_(10) = -0.5345224838248488;
    q_beam.element_position_(11) = 0.2672612419124244;

    q_beam_rot(0) = 1.674352746442651;
    q_beam_rot(1) = 0.1425949677148126;
    q_beam_rot(2) = 1.0831163124736984;
    q_beam_rot(3) = 1.4331999091513161;
    q_beam_rot(4) = -0.6560404572957742;
    q_beam_rot(5) = -0.2491376152457331;
    q_beam_rot(6) = 0.0;
    q_beam_rot(7) = 0.0;
    q_beam_rot(8) = 0.0;
";

functionStringPost="// Perform the unit tests.
    PerformMortarPairUnitTest(contact_pair, q_beam, q_beam_rot, q_solid, result_local_D,
        result_local_M, result_local_kappa);
  }";


SetDirectory[NotebookDirectory[]];

solidT:=Capitalize[solidType];
lambdaT:=Capitalize[lambdaType];
stringReplace:=StringReplace[
  functionStringPre,
  {
    "mathematica_solid_type"->solidType,
    "mathematica_mortar_type"->lambdaType,
    "mathematica_solid_Type"->solidT,
    "mathematica_mortar_Type"->lambdaT
  }
]<>"\n"<>getTestString<>"\n\n"<>functionStringPost;

solidType="hex8";
lambdaType="line2";
unitTestString=stringReplace;

solidType="hex8";
lambdaType="line3";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex8";
lambdaType="line4";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex20";
lambdaType="line2";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex20";
lambdaType="line3";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex20";
lambdaType="line4";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex27";
lambdaType="line2";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex27";
lambdaType="line3";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="hex27";
lambdaType="line4";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="tet4";
lambdaType="line2";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="tet4";
lambdaType="line3";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="tet4";
lambdaType="line4";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="tet10";
lambdaType="line2";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="tet10";
lambdaType="line3";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

solidType="tet10";
lambdaType="line4";
unitTestString=unitTestString<>"\n\n"<>stringReplace;

Export["unitTestCode.txt",unitTestString]



