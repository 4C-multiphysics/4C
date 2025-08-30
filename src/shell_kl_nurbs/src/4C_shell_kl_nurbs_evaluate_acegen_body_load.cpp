// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_shell_kl_nurbs.hpp"

FOUR_C_NAMESPACE_OPEN



void Discret::Elements::KirchhoffLoveShellNurbs::evaluate_body_load_auto_generated(
    const Core::FE::IntegrationPoints1D& intpoints,
    const std::vector<Core::LinAlg::SerialDenseVector>& knots,
    const Core::LinAlg::Matrix<9, 1>& weights, const Core::LinAlg::Matrix<9, 3>& X,
    const std::function<Core::LinAlg::Matrix<3, 1>(const double*)>& bodyload,
    Core::LinAlg::SerialDenseVector& elementload)
{
  std::vector<double> v(392);
  double v01[3];
  int i87, i88, i89, i175;
  Core::LinAlg::Matrix<3, 1, double> force;
  Core::LinAlg::Matrix<2, 1, double> uv;
  Core::LinAlg::Matrix<9, 1, double> N;
  Core::LinAlg::Matrix<2, 9, double> dN;
  Core::LinAlg::Matrix<3, 9, double> ddN;
  v[6] = X(0, 0);
  v[7] = X(0, 1);
  v[8] = X(0, 2);
  v[9] = X(1, 0);
  v[10] = X(1, 1);
  v[11] = X(1, 2);
  v[12] = X(2, 0);
  v[13] = X(2, 1);
  v[14] = X(2, 2);
  v[15] = X(3, 0);
  v[16] = X(3, 1);
  v[17] = X(3, 2);
  v[18] = X(4, 0);
  v[19] = X(4, 1);
  v[20] = X(4, 2);
  v[21] = X(5, 0);
  v[22] = X(5, 1);
  v[23] = X(5, 2);
  v[24] = X(6, 0);
  v[25] = X(6, 1);
  v[26] = X(6, 2);
  v[27] = X(7, 0);
  v[28] = X(7, 1);
  v[29] = X(7, 2);
  v[30] = X(8, 0);
  v[31] = X(8, 1);
  v[32] = X(8, 2);
  i87 = intpoints.nquad;
  for (i88 = 1; i88 <= i87; i88++)
  {
    v[179] = intpoints.qwgt[i88 - 1];
    v[90] = intpoints.qxg[i88 - 1][0];
    for (i89 = 1; i89 <= i87; i89++)
    {
      v[92] = intpoints.qwgt[i89 - 1] * v[179];
      uv(0) = v[90];
      uv(1) = intpoints.qxg[i89 - 1][0];
      Core::FE::Nurbs::nurbs_get_2d_funct_deriv_deriv2(
          N, dN, ddN, uv, knots, weights, Core::FE::CellType::nurbs9);
      v[123] = dN(0, 0);
      v[124] = dN(0, 1);
      v[125] = dN(0, 2);
      v[126] = dN(0, 3);
      v[127] = dN(0, 4);
      v[128] = dN(0, 5);
      v[129] = dN(0, 6);
      v[130] = dN(0, 7);
      v[131] = dN(0, 8);
      v[166] = v[11] * v[124] + v[125] * v[14] + v[126] * v[17] + v[127] * v[20] + v[128] * v[23] +
               v[129] * v[26] + v[130] * v[29] + v[131] * v[32] + v[123] * v[8];
      v[164] = v[10] * v[124] + v[125] * v[13] + v[126] * v[16] + v[127] * v[19] + v[128] * v[22] +
               v[129] * v[25] + v[130] * v[28] + v[131] * v[31] + v[123] * v[7];
      v[162] = v[12] * v[125] + v[126] * v[15] + v[127] * v[18] + v[128] * v[21] + v[129] * v[24] +
               v[130] * v[27] + v[131] * v[30] + v[123] * v[6] + v[124] * v[9];
      v[132] = dN(1, 0);
      v[133] = dN(1, 1);
      v[134] = dN(1, 2);
      v[135] = dN(1, 3);
      v[136] = dN(1, 4);
      v[137] = dN(1, 5);
      v[138] = dN(1, 6);
      v[139] = dN(1, 7);
      v[140] = dN(1, 8);
      v[167] = v[11] * v[133] + v[134] * v[14] + v[135] * v[17] + v[136] * v[20] + v[137] * v[23] +
               v[138] * v[26] + v[139] * v[29] + v[140] * v[32] + v[132] * v[8];
      v[165] = v[10] * v[133] + v[13] * v[134] + v[135] * v[16] + v[136] * v[19] + v[137] * v[22] +
               v[138] * v[25] + v[139] * v[28] + v[140] * v[31] + v[132] * v[7];
      v[163] = v[12] * v[134] + v[135] * v[15] + v[136] * v[18] + v[137] * v[21] + v[138] * v[24] +
               v[139] * v[27] + v[140] * v[30] + v[132] * v[6] + v[133] * v[9];
      v[141] = N(0);
      v[142] = N(1);
      v[143] = N(2);
      v[144] = N(3);
      v[145] = N(4);
      v[146] = N(5);
      v[147] = N(6);
      v[148] = N(7);
      v[149] = N(8);
      v[168] = sqrt(std::pow(-(v[163] * v[164]) + v[162] * v[165], 2) +
                    std::pow(v[163] * v[166] - v[162] * v[167], 2) +
                    std::pow(-(v[165] * v[166]) + v[164] * v[167], 2));
      v01[0] = v[12] * v[143] + v[144] * v[15] + v[145] * v[18] + v[146] * v[21] + v[147] * v[24] +
               v[148] * v[27] + v[149] * v[30] + v[141] * v[6] + v[142] * v[9];
      v01[1] = v[10] * v[142] + v[13] * v[143] + v[144] * v[16] + v[145] * v[19] + v[146] * v[22] +
               v[147] * v[25] + v[148] * v[28] + v[149] * v[31] + v[141] * v[7];
      v01[2] = v[11] * v[142] + v[14] * v[143] + v[144] * v[17] + v[145] * v[20] + v[146] * v[23] +
               v[147] * v[26] + v[148] * v[29] + v[149] * v[32] + v[141] * v[8];
      force = bodyload(v01);
      v[171] = force(0, 0);
      v[172] = force(1, 0);
      v[173] = force(2, 0);
      v[266] = v[141] * v[171];
      v[267] = v[141] * v[172];
      v[268] = v[141] * v[173];
      v[269] = v[142] * v[171];
      v[270] = v[142] * v[172];
      v[271] = v[142] * v[173];
      v[272] = v[143] * v[171];
      v[273] = v[143] * v[172];
      v[274] = v[143] * v[173];
      v[275] = v[144] * v[171];
      v[276] = v[144] * v[172];
      v[277] = v[144] * v[173];
      v[278] = v[145] * v[171];
      v[279] = v[145] * v[172];
      v[280] = v[145] * v[173];
      v[281] = v[146] * v[171];
      v[282] = v[146] * v[172];
      v[283] = v[146] * v[173];
      v[284] = v[147] * v[171];
      v[285] = v[147] * v[172];
      v[286] = v[147] * v[173];
      v[287] = v[148] * v[171];
      v[288] = v[148] * v[172];
      v[289] = v[148] * v[173];
      v[290] = v[149] * v[171];
      v[291] = v[149] * v[172];
      v[292] = v[149] * v[173];
      for (i175 = 1; i175 <= 27; i175++)
      {
        elementload(i175 - 1) += v[168] * v[265 + i175] * v[92];
      };
    };
  };
};

FOUR_C_NAMESPACE_CLOSE