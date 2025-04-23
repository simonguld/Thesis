#ifndef LB_HPP_
#define LB_HPP_

#include "fields.hpp"
#include "random.hpp"
#include <array>

// =============================================================================
// LB model definition and tools

// This will be put into a `Model' type if one wants to implement different LB
// models...

/** LB weights corresponding to the different directions */
constexpr static double w[] = {4/9., 1/9., 1/9., 1/9., 1/9.,
                               1/36., 1/36., 1/36., 1/36.};
                               //

//older version
/* constexpr static double w[] = {0, 1/3., 1/3., 1/3., 1/3.,
                               1/12., 1/12., 1/12., 1/12.};
                               */

// =============================================================================
// More stuff

/** Parameter weights for compressible LB forces: Name is short for:   Force Order(2) Weights*/
constexpr static double FO2_wtXX[] = {-3.,  6.,  6., -3., -3.,  6.,  6.,  6.,  6.}; 
constexpr static double FO2_wtXY[] = { 0.,  0.,  0.,  0.,  0.,  9.,  9., -9., -9.}; 
constexpr static double FO2_wtYY[] = {-3., -3., -3.,  6.,  6.,  6.,  6.,  6.,  6.}; 

/** parameters for derivative stencil */
const double sB = 1/12., sD = 1/12.;
const double cs2 = 1./3.;
/** LB equilibrium distribution at a node */
inline LBNode GetEquilibriumDistribution(double vx, double vy, double nn)
{
  LBNode fe;

  const double v2 = vx*vx + vy*vy;
  fe[1] = nn*(1./9. + 1./3.*vx + 1./3.*v2 - 1./2.*vy*vy);
  fe[2] = nn*(1./9. - 1./3.*vx + 1./3.*v2 - 1./2.*vy*vy);
  fe[3] = nn*(1./9. + 1./3.*vy - 1./2.*vx*vx + 1./3.*v2);
  fe[4] = nn*(1./9. - 1./3.*vy - 1./2.*vx*vx + 1./3.*v2);
  fe[5] = nn*(1./36. + 1./12.*vx + 1./12.*vy + 1./12.*v2 + 1./4.*vx*vy);
  fe[6] = nn*(1./36. - 1./12.*vx - 1./12.*vy + 1./12.*v2 + 1./4.*vx*vy);
  fe[7] = nn*(1./36. - 1./12.*vx + 1./12.*vy + 1./12.*v2 - 1./4.*vx*vy);
  fe[8] = nn*(1./36. + 1./12.*vx - 1./12.*vy + 1./12.*v2 - 1./4.*vx*vy);
  // from mass conservation
  fe[0] = nn-fe[1]-fe[2]-fe[3]-fe[4]-fe[5]-fe[6]-fe[7]-fe[8];

  return fe;
}

/** Generates correlated noise with strength s on a node */
inline LBNode GenerateNoiseDistribution(double s)
{
  LBNode noise;

  const double xi4 = random_normal();
  const double xi5 = random_normal();
  const double xi6 = random_normal();
  const double xi7 = random_normal();
  const double xi8 = random_normal();
  const double xi9 = random_normal();
  constexpr double sqrt6 = sqrt(6);

  noise[0] = s*2./9.*(-2.*xi4 + xi9);
  noise[1] = s*(xi4 + 3.*xi5 - sqrt6*xi7 - 2.*xi9)/18.;
  noise[2] = s*(xi4 + 3.*xi5 + sqrt6*xi7 - 2.*xi9)/18.;
  noise[3] = s*(xi4 - 3.*xi5 - sqrt6*xi8 - 2.*xi9)/18.;
  noise[4] = s*(xi4 - 3.*xi5 + sqrt6*xi8 - 2.*xi9)/18.;
  noise[5] = s*(2.*xi4 + 3.*xi6 + sqrt6*xi7 + sqrt6*xi8 + 2.*xi9)/36.;
  noise[6] = s*(2.*xi4 + 3.*xi6 - sqrt6*xi7 - sqrt6*xi8 + 2.*xi9)/36.;
  noise[7] = s*(2.*xi4 - 3.*xi6 - sqrt6*xi7 + sqrt6*xi8 + 2.*xi9)/36.;
  noise[8] = s*(2.*xi4 - 3.*xi6 + sqrt6*xi7 - sqrt6*xi8 + 2.*xi9)/36.;

  return noise;
}

// =============================================================================
// Derivatives

/** Five-point finite difference derivative along the x direction */
inline double derivX(const ScalarField& arr,
                     const NeighboursList& d,
                     const double stencil)
{
  return .5*(1-4*stencil)*(arr[d[1]]-arr[d[2]]) + stencil*(arr[d[5]]-arr[d[6]]-arr[d[7]]+arr[d[8]]);
  //return .5*(arr[d[1]]-arr[d[2]])   +0*stencil;
}

/** Five-point finite difference derivative along the x direction */
inline double derivY(const ScalarField& arr,
                     const NeighboursList& d,
                     const double stencil)
{
  return .5*(1-4*stencil)*(arr[d[3]]-arr[d[4]]) + stencil*(arr[d[5]]-arr[d[6]]+arr[d[7]]-arr[d[8]]);
  //return .5*(arr[d[3]]-arr[d[4]])   +0*stencil;
}

/** Five-point finite difference second derivative along the x direction */
inline double derivXX(const ScalarField& arr,
          const NeighboursList& d,
          const double stencil)
{
  return (1-4*stencil)*(arr[d[1]]+arr[d[2]])
   - 4*stencil*(arr[d[3]]+arr[d[4]])
   + 2*stencil*(arr[d[5]]+arr[d[6]]+arr[d[7]]+arr[d[8]])
   - 2*(1-4*stencil)*arr[d[0]];
}

/** Five-point finite difference second derivative along the z direction */
inline double derivYY(const ScalarField& arr,
          const NeighboursList& d,
          const double stencil)
{
  return (1-4*stencil)*(arr[d[3]]+arr[d[4]])
   - 4*stencil*(arr[d[1]]+arr[d[2]])
   + 2*stencil*(arr[d[5]]+arr[d[6]]+arr[d[7]]+arr[d[8]])
   - 2*(1-4*stencil)*arr[d[0]];
}

/** Five-point finite difference derivative along x and z direction */
inline double derivXY(const ScalarField& arr,
          const NeighboursList& d,
          const double stencil)
{
  return 0.5*(1-8*stencil)*(arr[d[1]]+arr[d[2]]+arr[d[3]]+arr[d[4]])
   + 2*stencil*(arr[d[5]]+arr[d[6]])
   - 0.5*(1-4*stencil)*(arr[d[7]]+arr[d[8]])
   - (1-8*stencil)*arr[d[0]];
}


/** Five-point finite difference laplacian (const) */
inline double laplacian(const ScalarField& arr,
                        const NeighboursList& d,
                        const double stencil)
{
  return (1-4*stencil)*(arr[d[1]]+arr[d[2]]+arr[d[3]]+arr[d[4]])
         + 2*stencil*(arr[d[5]]+arr[d[6]]+arr[d[7]]+arr[d[8]])
         - 4*(1-2*stencil)*arr[d[0]];
  //return arr[d[1]]+arr[d[2]]+arr[d[3]]+arr[d[4]] - 4*arr[d[0]]   +0*stencil;

}

/** Five-point finite difference flux
 *
 * This is just dx(ux*field) + dy(uy*field) with the same definition for the derivative as
 * derivX and derivY.
 * */
inline double flux(const ScalarField& arr,
                   const ScalarField& uxs,
                   const ScalarField& uzs,
                   const NeighboursList& d,
                   const double stencil)
{
  //return + .5*(1-4*stencil)*(uxs[d[1]]*arr[d[1]]-uxs[d[2]]*arr[d[2]])
  //       + stencil*(uxs[d[5]]*arr[d[5]]-uxs[d[6]]*arr[d[6]]-uxs[d[7]]*arr[d[7]]+uxs[d[8]]*arr[d[8]])
  //       + .5*(1-4*stencil)*(uzs[d[3]]*arr[d[3]]-uzs[d[4]]*arr[d[4]])
  //       + stencil*(uzs[d[5]]*arr[d[5]]-uzs[d[6]]*arr[d[6]]+uzs[d[7]]*arr[d[7]]-uzs[d[8]]*arr[d[8]]);
  //return   + .5*(uxs[d[1]]*arr[d[1]]-uxs[d[2]]*arr[d[2]])
  //         + .5*(uzs[d[3]]*arr[d[3]]-uzs[d[4]]*arr[d[4]])   +0*stencil;
  //return   +  .5*(uxs[d[1]]-uxs[d[2]])*arr[d[0]] + .5*(arr[d[1]]-arr[d[2]])*uxs[d[0]]
  //         +  .5*(uzs[d[3]]-uzs[d[4]])*arr[d[0]] + .5*(arr[d[3]]-arr[d[4]])*uzs[d[0]]   +0*stencil;
  return derivX(arr,d,0)*uxs[d[0]] + derivX(uxs,d,0)*arr[d[0]] + derivY(arr,d,0)*uzs[d[0]] + derivY(uzs,d,0)*arr[d[0]] + 0*stencil;
}

#endif//LB_HPP_
