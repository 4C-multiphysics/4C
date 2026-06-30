#!/usr/bin/env wolframscript

# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3 .0 or later .
#
# See the LICENSE . md file in the top-level for license information .
#
# SPDX-License-Identifier: LGPL-3.0-or-later

(* ::Package:: *)

(* ::Title:: *)
(*Create unit test values for beam-to-solid volume mesh tying mortar pair*)
(*Hermite beam, Hex8 solid, Dual Hermite LM*)


<<FEMUtils`
<<NumericalDifferentialEquationAnalysis`


(* ::Section:: *)
(*Matrix/vector to Core::LinAlg output format*)


matrixToFourC[var_, name_] := Module[{allStringsWithZeros, stringList},
  allStringsWithZeros = Table[
    If[Chop[var[[i, j]], 10^-14] == 0,
      0,
      ToString[
        StringForm[
          "    ``(``, ``) = ``;",
          name,
          i - 1,
          j - 1,
          NumberForm[
            N[var[[i, j]], 17],
            17,
            ScientificNotationThreshold -> {-20, 20}
          ]
        ]
      ]
    ],
    {i, Dimensions[var][[1]]},
    {j, Dimensions[var][[2]]}
  ];

  stringList = DeleteCases[Flatten[allStringsWithZeros], 0];
  StringRiffle[stringList, "\n"]
];


vectorToFourC[var_, name_] := Module[{allStringsWithZeros, stringList},
  allStringsWithZeros = Table[
    If[Chop[var[[i]], 10^-14] == 0,
      0,
      ToString[
        StringForm[
          "    ``(``) = ``;",
          name,
          i - 1,
          NumberForm[
            N[var[[i]], 17],
            17,
            ScientificNotationThreshold -> {-20, 20}
          ]
        ]
      ]
    ],
    {i, Length[var]}
  ];

  stringList = DeleteCases[Flatten[allStringsWithZeros], 0];
  StringRiffle[stringList, "\n"]
];


(* ::Section:: *)
(*Element types*)


solidType = "hex8";
beamType = "hermite2";


(* ::Section:: *)
(*Solid position interpolation*)


xSolid := {
  fieldFunction[solidType] /. posDof -> posSx[n],
  fieldFunction[solidType] /. posDof -> posSy[n],
  fieldFunction[solidType] /. posDof -> posSz[n]
};


qSolid := Flatten[
  Transpose[
    {
      discreteValues[solidType] /. posDof -> posSx[n],
      discreteValues[solidType] /. posDof -> posSy[n],
      discreteValues[solidType] /. posDof -> posSz[n]
    }
  ]
];


(* Clean HEX8 solid enclosing the beam.

   Node ordering:
   bottom face: 0,1,2,3
   top face:    4,5,6,7
*)
replaceSolidRef := Thread[
  qSolid -> {
    -0.2, -0.2, -0.2,
     1.0, -0.2, -0.2,
     1.0,  0.6, -0.2,
    -0.2,  0.6, -0.2,

    -0.2, -0.2,  0.6,
     1.0, -0.2,  0.6,
     1.0,  0.6,  0.6,
    -0.2,  0.6,  0.6
  }
] /. n -> 0;


(* ::Section:: *)
(*Beam position interpolation*)


xBeam := {
  fieldFunction[beamType] /. posDof -> posBx[n] /. tanDof -> tanBx[n],
  fieldFunction[beamType] /. posDof -> posBy[n] /. tanDof -> tanBy[n],
  fieldFunction[beamType] /. posDof -> posBz[n] /. tanDof -> tanBz[n]
} /. lenDof -> beamLength;


(* ::Section:: *)
(*Reference beam position*)


replaceBeamRef := {
  beamLength -> 0.4242640687119285,

  posBx[0][1] -> 0.0,
  posBy[0][1] -> 0.0,
  posBz[0][1] -> 0.0,

  tanBx[0][1] -> 0.94280904158,
  tanBy[0][1] -> 0.23570226039,
  tanBz[0][1] -> 0.23570226039,

  posBx[0][2] -> 0.4,
  posBy[0][2] -> 0.1,
  posBz[0][2] -> 0.1,

  tanBx[0][2] -> 0.94280904158,
  tanBy[0][2] -> 0.23570226039,
  tanBz[0][2] -> 0.23570226039
};


qBeam := Drop[replaceBeamRef[[All, 1]], 1] /. x_[0] :> x[n];


(* ::Section:: *)
(*Dual Hermite LM interpolation*)


(* Dual Hermite basis.  This is written directly here instead of using
   fieldFunction["hermitedual"], because only the basis polynomials are needed.

   IMPORTANT:
   Use beamLength here, because this is the same value set as
   q_beam.shape_function_data_.ref_length_ in the C++ test.
*)

lambdaShape := {
  -(35/4) xi^3 + (15/4) xi^2 + (15/4) xi - 3/4,

  (1/beamLength) (
    105 xi^3 - (45/2) xi^2 - 60 xi + 15/2
  ),

  (35/4) xi^3 + (15/4) xi^2 - (15/4) xi - 3/4,

  (1/beamLength) (
    105 xi^3 + (45/2) xi^2 - 60 xi - 15/2
  )
};


lambda := {
  lambdaShape[[1]] lambdaX1 +
  lambdaShape[[2]] lambdaX2 +
  lambdaShape[[3]] lambdaX3 +
  lambdaShape[[4]] lambdaX4,

  lambdaShape[[1]] lambdaY1 +
  lambdaShape[[2]] lambdaY2 +
  lambdaShape[[3]] lambdaY3 +
  lambdaShape[[4]] lambdaY4,

  lambdaShape[[1]] lambdaZ1 +
  lambdaShape[[2]] lambdaZ2 +
  lambdaShape[[3]] lambdaZ3 +
  lambdaShape[[4]] lambdaZ4
};


qLambda := {
  lambdaX1, lambdaY1, lambdaZ1,
  lambdaX2, lambdaY2, lambdaZ2,
  lambdaX3, lambdaY3, lambdaZ3,
  lambdaX4, lambdaY4, lambdaZ4
};


(* ::Section:: *)
(*Project beam point to solid reference coordinates*)


projectPointToSolid[point_] := Module[{equations},
  equations = Thread[point == xSolid /. n -> 0 /. replaceSolidRef];

  FindRoot[
    equations,
    {{xi1, 0}, {xi2, 0}, {xi3, 0}},
    PrecisionGoal -> 20
  ]
];


(* ::Section:: *)
(*Numerical integration along beam element*)


nGP = 6;
gaussSupportsWeights = GaussianQuadratureWeights[nGP, -1, 1];

(* ::Section:: *)
(*Debug: print beam and mortar shape functions*)

formatNumber[x_] := ToString[
  NumberForm[N[x, 17], 17, ScientificNotationThreshold -> {-20, 20}]
];

formatVector[v_] := "[" <> StringRiffle[formatNumber /@ N[v, 17], ", "] <> "]";

(* Hermite beam shape functions.
   Order: {pos node 1, tangent node 1, pos node 2, tangent node 2}. *)
beamShape[xii_] := Module[{xBeam0},
  xBeam0 = xBeam[[1]] /. n -> 0;

  {
    D[xBeam0, posBx[0][1]],
    D[xBeam0, tanBx[0][1]],
    D[xBeam0, posBx[0][2]],
    D[xBeam0, tanBx[0][2]]
  } /. xi -> xii /. beamLength -> 0.4242640687119285
];

(* Dual Hermite mortar shape functions.
   Order: {lambda node 1 value, lambda node 1 tangent,
           lambda node 2 value, lambda node 2 tangent}. *)
mortarShape[xii_] := (
  lambdaShape /. xi -> xii /. replaceBeamRef
);

printShapeFunctionDebug[] := Module[{eta, w, nm, nb},
  Print["\nShape-function debug at Gauss points:"];

  Do[
    eta = item[[1]];
    w = item[[2]];

    nm = mortarShape[eta];
    nb = beamShape[eta];

    Print["eta = ", formatNumber[eta], ", weight = ", formatNumber[w]];

    Print["N_mortar = Matrix<1,4>"];
    Print[formatVector[nm]];
    Print["sum N_mortar = ", formatNumber[Total[nm]]];

    Print["N_beam = Matrix<1,4>"];
    Print[formatVector[nb]];
    Print["sum N_beam = ", formatNumber[Total[nb]]];

    Print["beam Jacobian Norm[D[xBeam, xi]] = ",
      formatNumber[
        Norm[D[xBeam, xi]] /. n -> 0 /. xi -> eta /. replaceBeamRef
      ]
    ];

    Print["---"];
    ,
    {item, gaussSupportsWeights}
  ];
];


integrateLocalN[val_] := Sum[
  (
    Norm[D[xBeam, xi]] /. n -> 0 /. xi -> item[[1]]
  ) item[[2]] val[item[[1]]],
  {item, gaussSupportsWeights}
];


(* ::Section:: *)
(*Generate mesh tying potential*)


gaussPointPotential[xii_] := Module[{pointInSolid},
  pointInSolid = projectPointToSolid[
    xBeam /. xi -> xii /. n -> 0 /. replaceBeamRef
  ];

  lambda . (xBeam - xSolid) /. xi -> xii /. pointInSolid
];


(* Calculate the element residual with respect to LM DOFs. *)
PiMT := integrateLocalN[gaussPointPotential] /. replaceBeamRef;

gmt := D[PiMT, {qLambda}];


(* ::Section:: *)
(*Calculate result_local_D, result_local_M, and result_local_kappa*)


DD := D[gmt, {qBeam}];

MM := -D[gmt, {qSolid}];


fLambda[xii_] := lambda /. xi -> xii;

intLambda := integrateLocalN[fLambda] /. replaceBeamRef;

kappa := DeleteCases[
  Flatten[Transpose[D[intLambda, {qLambda}]]],
  0
];


(* ::Section:: *)
(*Output the results*)

printShapeFunctionDebug[];
output = StringRiffle[
  {
    "// Positional DOFs of the clean HEX8 solid.",
    vectorToFourC[replaceSolidRef[[All, 2]], "q_solid.element_position_"],

    "\n// Results for D.",
    matrixToFourC[DD, "result_local_D"],

    "\n// Results for M.",
    matrixToFourC[MM, "result_local_M"],

    "\n// Results for Kappa.",
    vectorToFourC[kappa, "result_local_kappa"]
  },
  "\n"
];


Print["Dimensions of result_local_D: ", Dimensions[DD]];
Print["Dimensions of result_local_M: ", Dimensions[MM]];
Print["Length of result_local_kappa: ", Length[kappa]];

Export["dualHermite_hex8_results_modified_geometry.txt", output];

Print["Written file: dualHermite_hex8_results_modified_geometry.txt"];