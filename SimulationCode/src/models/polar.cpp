#include "header.hpp"
#include "models/polar.hpp"
#include "error_msg.hpp"
#include "random.hpp"
#include "lb.hpp"
#include "tools.hpp"

using namespace std;
namespace opt = boost::program_options;

// from main.cpp:
extern unsigned nthreads, nsubsteps;
extern double time_step;

Polar::Polar(unsigned LX, unsigned LY, unsigned BC)
  : Model(LX, LY, BC, BC==0 ? GridType::Periodic : GridType::Layer)
{}
Polar::Polar(unsigned LX, unsigned LY, unsigned BC, GridType Type)
  : Model(LX, LY, BC, Type)
{}

void Polar::Initialize()
{
  // initialize variables
  angle = angle_deg*M_PI/180.;

  // allocate memory
  ff.SetSize(LX, LY, Type);
  fn.SetSize(LX, LY, Type);
  ff_tmp.SetSize(LX, LY, Type);
  fn_tmp.SetSize(LX, LY, Type);
  Px.SetSize(LX, LY, Type);
  Py.SetSize(LX, LY, Type);
  PNx.SetSize(LX, LY, Type);
  PNy.SetSize(LX, LY, Type);
  n.SetSize(LX, LY, Type);
  ux.SetSize(LX, LY, Type);
  uy.SetSize(LX, LY, Type);
  Hx.SetSize(LX, LY, Type);
  Hy.SetSize(LX, LY, Type);
  dxPx.SetSize(LX, LY, Type);
  dyPx.SetSize(LX, LY, Type);
  dxPy.SetSize(LX, LY, Type);
  dyPy.SetSize(LX, LY, Type);
  sigmaXX.SetSize(LX, LY, Type);
  sigmaYY.SetSize(LX, LY, Type);
  sigmaYX.SetSize(LX, LY, Type);
  sigmaXY.SetSize(LX, LY, Type);

  FFx.SetSize(LX, LY, Type);
  FFy.SetSize(LX, LY, Type);

  if(nsubsteps>1)
    throw error_msg("time stepping not implemented for this model"
                    ", please set nsubsteps=1.");
}

void Polar::ConfigureAtNode(unsigned k)
{
  double Order = init_order;
  double theta;

  theta   = angle + noise*M_PI*(random_real() - .5);
	
  Px[k] = Order*(cos(theta));
  Py[k] = Order*(sin(theta));
  // equilibrium dist
  ux[k] = uy[k] = 0;
  n[k]  = rho;
  ff[k] = GetEquilibriumDistribution(ux[k], uy[k], n[k]);
  // compute totals for later checks
  ftot  = accumulate(begin(ff[k]), end(ff[k]), ftot);
}
void Polar::Configure()
{
  for(unsigned k=0; k<DomainSize; ++k)
    ConfigureAtNode(k);

  //and do preinitialization
  cout << "Preinitialization started. ... ";
  preinit_flag = true;
  for (int i = 0; i< n_preinit; i++){
    Step();
  }
  preinit_flag = false;
  cout << "Preinitialization done. ... ";

}

void Polar::UpdatePolarQuantitiesAtNode(unsigned k)
{
  const auto& d = get_neighbours(k);  

  const double px = Px[k];
  const double py = Py[k]; 
  const double ps = px*px+py*py;
  const double p4 = px*px*px*px+py*py*py*py+2*px*px*py*py;

  const double dxpx = derivX(Px, d, sB);
  const double dypx = derivY(Px, d, sB);
  const double dxpy = derivX(Py, d, sB);
  const double dypy = derivY(Py, d, sB);

  const double term = 1. - ( nem_align? p4 : ps );
  const double hx = CC*term*( nem_align? 4*ps : 1. )*px + Kn*laplacian(Px, d, sD)
	  	  + Kp*( 1.*ps*laplacian(Px, d, sD)
         	  	+2.*px*(dxpx*dxpx+dypx*dypx-dxpy*dxpy-dypy*dypy)
         	  	+4.*py*(dxpx*dxpy+dypx*dypy));
  const double hy = CC*term*( nem_align? 4*ps : 1. )*py + Kn*laplacian(Py, d, sD)
	  	  + Kp*( 1.*ps*laplacian(Py, d, sD)
         	        +2.*py*(dxpy*dxpy+dypy*dypy-dxpx*dxpx-dypx*dypx)
         		+4.*px*(dxpx*dxpy+dypx*dypy));

  // backflow implemeted as of Giomi & Marchetti, Soft Matter (2012): https://doi.org/10.1039/C1SM06077E with \bar{lambda}=\lambda
  const double sigmaxx = -.5*zeta*(px*px-py*py) + (beta == 0 ? 1 : 0)*( ( backflow_on? -2.*xi*px*hx-xi*py*hy : 0) ) + (backflow_on? .5*CC*term*term : 0);
  const double sigmayy = -.5*zeta*(py*py-px*px) + (beta == 0 ? 1 : 0)*( ( backflow_on? -2.*xi*py*hy-xi*px*hx : 0) ) + (backflow_on? .5*CC*term*term : 0);
  const double sigmaxy = -zeta*px*py + (beta == 0 ? 1 : 0)*( ( backflow_on? .5*(1.-xi)*px*hy - .5*(1.+xi)*py*hx :0) );
  const double sigmayx = -zeta*py*px + (beta == 0 ? 1 : 0)*( ( backflow_on? .5*(1.-xi)*py*hx - .5*(1.+xi)*px*hy :0) );

  // transfer to arrays  
  Hx[k]    =  hx;
  Hy[k]    =  hy;
  dxPx[k]  =  dxpx;
  dxPy[k]  =  dxpy;
  dyPx[k]  =  dypx;
  dyPy[k]  =  dypy;
  sigmaXX[k] =  sigmaxx;
  sigmaYY[k] =  sigmayy;
  sigmaXY[k] =  sigmaxy;
  sigmaYX[k] =  sigmayx;
}
void Polar::UpdatePolarQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdatePolarQuantitiesAtNode(k);
}

void Polar::UpdateFluidQuantitiesAtNode(unsigned k)
{
  // array placeholders for current node
  const auto& d = get_neighbours(k);
  const auto& f = ff[k];

  const double dxSxx = derivX(sigmaXX, d, sB);
  const double dySxy = derivY(sigmaXY, d, sB);
  const double dxSyx = derivX(sigmaYX, d, sB);
  const double dySyy = derivY(sigmaYY, d, sB);
  const double Fx = dxSxx + dySxy;
  const double Fy = dxSyx + dySyy;
  
  FFx[k] = Fx;
  FFy[k] = Fy;

  const double AA_LBF = isGuo? 0.5 : tau;
  // compute velocities
  const double nn = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
  const double vx = (f[1] - f[2] + f[5] - f[6] - f[7] + f[8] + AA_LBF*FFx[k])/nn;
  const double vy = (f[3] - f[4] + f[5] - f[6] + f[7] - f[8] + AA_LBF*FFy[k])/nn;

  // transfer to arrays
  n[k]       =  nn;
  ux[k]      =  vx;
  uy[k]      =  vy;
}

void Polar::UpdateFluidQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k){
    UpdateFluidQuantitiesAtNode(k);
  }
}

void Polar::UpdatePolarFieldsAtNode(unsigned k, bool first)
{  
  const auto& d = get_neighbours(k);
  
  const double vx = ux[k];
  const double vy = uy[k];

  const double px = Px[k];
  const double py = Py[k];
  const double hx = Hx[k];
  const double hy = Hy[k];

  const double dxpx = dxPx[k];
  const double dypx = dyPx[k];
  const double dxpy = dxPy[k];
  const double dypy = dyPy[k];

  const double dxux = derivX(ux, d, sB);
  const double dyux = derivY(ux, d, sB);
  const double dxuy = derivX(uy, d, sB);
  const double dyuy = derivY(uy, d, sB);

  // corrections to the polarisation, the beta term is alignment to velocity after Maitra et al. PRL (2020) DOI: 10.1103/PhysRevLett.124.028002
  const double Dx = hx/gamma 
	  	    - (vx + Vp*px)*dxpx - (vy + Vp*py)*dypx
                    + (beta == 0 ? 1 : 0)*(xi*(dxux*px + .5*(dyux+dxuy)*py) + .5*(dyux-dxuy)*py )
		    + beta*vx;
  const double Dy = hy/gamma 
	  	    - (vx + Vp*px)*dxpy - (vy + Vp*py)*dypy
                    + (beta == 0 ? 1 : 0)*(xi*(dyuy*py + .5*(dxuy+dyux)*px) + .5*(dxuy-dyux)*px )
		    + beta*vy;

  if(first)
  {
    PNx[k] = Px[k] + .5*Dx;
    PNy[k] = Py[k] + .5*Dy;
    Px[k] = PNx[k] + .5*Dx;
    Py[k] = PNy[k] + .5*Dy;
  }
  else
  {
    Px[k] = PNx[k] + .5*Dx;
    Py[k] = PNy[k] + .5*Dy;
    
  }
}
void Polar::UpdatePolarFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdatePolarFieldsAtNode(k, first);
}

void Polar::UpdateFluidFieldsAtNode(unsigned k, bool first)
{  
  const double nn = n[k];
  const double vx = ux[k];
  const double vy = uy[k];
  const double px = Px[k];
  const double py = Py[k];
  const double hx = Hx[k];
  const double hy = Hy[k];

  const double Fx = FFx[k] - friction*vx + alpha*px - beta*hx;
  const double Fy = FFy[k] - friction*vy + alpha*py - beta*hy;
  
  // LB Step for the fluid field
  const auto fe = GetEquilibriumDistribution(vx, vy, nn);
  if(first)
  {   
    for(unsigned v=0; v<lbq; ++v)
    {
      const double Si = isGuo? w[v]*(1-1./(2.*tau))*(Fx*xdir(v) + Fy*ydir(v))/cs2 : 0;

      fn[k][v] = ff[k][v] + .5*(fe[v]-ff[k][v])/tau + 0.5*Si;
      ff[k][v] = ff[k][v] +    (fe[v]-ff[k][v])/tau +     Si;
        // LB Corrected as per Kruger pp 233
    }
  }
  else
  {
    for(unsigned v=0; v<lbq; ++v){
      const double Si = isGuo? w[v]*(1-1./(2.*tau))*(Fx*xdir(v) + Fy*ydir(v))/cs2 : 0;    
      ff[k][v] = fn[k][v] + .5*(fe[v]-ff[k][v])/tau + 0.5*Si;
    }
  }
}
void Polar::UpdateFluidFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateFluidFieldsAtNode(k, first);
}

void Polar::Move()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<TotalSize; ++k)
  {
    for(unsigned v=0; v<lbq; ++v)
    {
      // advect particles
      ff_tmp[next(k, v)][v] = ff[k][v];
      fn_tmp[next(k, v)][v] = fn[k][v];
    }
  }
  // swap temp variables
  swap(ff.get_data(), ff_tmp.get_data());
  swap(fn.get_data(), fn_tmp.get_data());
}

void Polar::BoundaryConditionsLB()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // free-slip channel
    case 1:
      ff.ApplyFreeSlipChannel();
      fn.ApplyFreeSlipChannel();
      break;
    // no-slip channel
    case 2:
      ff.ApplyNoSlipChannel();
      fn.ApplyNoSlipChannel();
      break;
    // free-slip box
    case 3:
      ff.ApplyFreeSlip();
      fn.ApplyFreeSlip();
      break;
    // no slip box
    case 4:
      ff.ApplyNoSlip();
      fn.ApplyNoSlip();
      break;
    // pbc with boundary layer
    default:
      ff.ApplyPBC();
      fn.ApplyPBC();
  }
}
void Polar::BoundaryConditionsFields()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // channel
    case 1:
    case 2:
      Px.ApplyNeumannChannel();
      Py.ApplyNeumannChannel();
      break;
    // box
    case 3:
    case 4:
      Px.ApplyNeumann();
      Py.ApplyNeumann();
      break;
    // pbc with bdry layer
    default:
      Px.ApplyPBC();
      Py.ApplyPBC();
  }
}
void Polar::BoundaryConditionsFields2()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // channel
    case 1:
    case 2:
      uy     .CopyDerivativeChannel();
      ux     .CopyDerivativeChannel();

      sigmaXX.ApplyNeumannChannel();
      sigmaYY.ApplyNeumannChannel();
      sigmaYX.ApplyNeumannChannel();
      sigmaXY.ApplyNeumannChannel();
      break;
    // box
    case 3:
    case 4:
      uy     .CopyDerivative();
      ux     .CopyDerivative();

      sigmaXX.ApplyNeumann();
      sigmaYY.ApplyNeumann();
      sigmaYX.ApplyNeumann();
      sigmaXY.ApplyNeumann();
      break;
    // pbc with bdry layer
    default:
      ux     .ApplyPBC();
      uy     .ApplyPBC();

      sigmaXX.ApplyPBC();
      sigmaYY.ApplyPBC();
      sigmaYX.ApplyPBC();
      sigmaXY.ApplyPBC();
  }
}

void Polar::Step()
{
  // boundary conditions for primary fields
  BoundaryConditionsFields();
  // predictor step
  UpdatePolarQuantities();
  UpdateFluidQuantities();
  
  BoundaryConditionsFields2();

  // LB Step
  this->UpdatePolarFields(true); 
  this->UpdateFluidFields(true);  
  BoundaryConditionsLB();
  Move();

  // corrector steps
  for(unsigned n=1; n<=npc; ++n)
  {    
    BoundaryConditionsFields();
    UpdatePolarQuantities();
    UpdateFluidQuantities();
    BoundaryConditionsFields2();
    this->UpdatePolarFields(true); 
    this->UpdateFluidFields(true);
  }
}

void Polar::RuntimeChecks()
{
  // check that the sum of f is constant
  {
    double fcheck = 0;
    for(unsigned k=0; k<DomainSize; ++k)
        fcheck = accumulate(begin(ff[k]), end(ff[k]), fcheck);
    cout << "fcheck: " << fcheck << "/" << ftot << '\n';
    if(abs(ftot-fcheck)>1)
      throw error_msg("f is not conserved (", ftot, "/", fcheck, ")");
  }
}

option_list Polar::GetOptions()
{
  // model specific options
  opt::options_description model_options("Model options");
  model_options.add_options()
    ("gamma", opt::value<double>(&gamma),
     "Q-tensor mobility")
    ("xi", opt::value<double>(&xi),
     "tumbling/aligning parameter")
    ("beta", opt::value<double>(&beta),
     "alignment to velocity parameter")
    ("tau", opt::value<double>(&tau),
     "polar viscosity")
    ("rho", opt::value<double>(&rho),
     "fluid density")
    ("friction", opt::value<double>(&friction),
     "friction from confinement")
    ("CC", opt::value<double>(&CC),
     "coupling constant")
    ("Kn", opt::value<double>(&Kn),
     "nematic elastic constant")
    ("Kp", opt::value<double>(&Kp),
     "polar elastic constant")
    ("zeta", opt::value<double>(&zeta),
     "activity parameter")
    ("Vp", opt::value<double>(&Vp),
     "self-advection parameter")
    ("alpha", opt::value<double>(&alpha),
     "monopole activity parameter")
    ("npc", opt::value<unsigned>(&npc),
     "number of correction steps for the predictor-corrector method")
    ("backflow_on", opt::value<bool>(&backflow_on),
     "Backflow flag")
    ("nem_align", opt::value<bool>(&nem_align),
     "nematic alignment flag")
    ("isGuo", opt::value<bool>(&isGuo),
     "LB forcing scheme")
    ("n_preinit", opt::value<int>(&n_preinit),
     "number of preinitialization steps");

  // init config options
  opt::options_description config_options("Initial configuration options");
  config_options.add_options()
    ("angle", opt::value<double>(&angle_deg),
     "initial angle to x direction (in degrees)")
    ("noise", opt::value<double>(&noise),
     "size of initial variations")
    ("initial-order", opt::value<double>(&init_order),
     "initial order of the polarisation field");

  return { model_options, config_options };
}
