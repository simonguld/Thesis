#include "header.hpp"
#include "models/drypolarlyotropic.hpp"
#include "error_msg.hpp"
#include "random.hpp"
#include "lb.hpp"
#include "tools.hpp"

using namespace std;
namespace opt = boost::program_options;

// from main.cpp:
extern unsigned nthreads, nsubsteps;
extern double time_step;

DryPolarLyotropic::DryPolarLyotropic(unsigned LX, unsigned LY, unsigned BC)
  : Model(LX, LY, BC, BC==0 ? GridType::Periodic : GridType::Layer)
{}
DryPolarLyotropic::DryPolarLyotropic(unsigned LX, unsigned LY, unsigned BC, GridType Type)
  : Model(LX, LY, BC, Type)
{}

void DryPolarLyotropic::Initialize()
{
  // initialize variables
  angle = angle_deg*M_PI/180.;

  // allocate memory
  Px.SetSize(LX, LY, Type);
  Py.SetSize(LX, LY, Type);
  PNx.SetSize(LX, LY, Type);
  PNy.SetSize(LX, LY, Type);
  phi.SetSize(LX, LY, Type);
  phi_tmp.SetSize(LX, LY, Type);
  phn.SetSize(LX, LY, Type);
  ux.SetSize(LX, LY, Type);
  uy.SetSize(LX, LY, Type);
  ux_phi.SetSize(LX, LY, Type);
  uy_phi.SetSize(LX, LY, Type);
  Hx.SetSize(LX, LY, Type);
  Hy.SetSize(LX, LY, Type);
  MU.SetSize(LX, LY, Type);
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

void DryPolarLyotropic::ConfigureAtNode(unsigned k)
{
  double Order = 0;
  double theta;
//  double xtemp,ytemp;
  const unsigned x = GetXPosition(k);
  const unsigned y = GetYPosition(k);
//  xtemp=x;
//  ytemp=y;
  if(init_config=="circle")
  {
    if(pow(diff(x, LX/2), 2) + pow(diff(y, LY/2), 2) <= radius*radius)
      Order = init_order;
  }
  else if(init_config=="square")
  {
    if (diff(LY/2, y) < level/2 && diff(LX/2, x) < level/2)
      Order = init_order;
  }
  else if(init_config=="stripe")
  {
    if(diff(LY/2, y) < level/2) Order = init_order;
  }
  else if(init_config=="half")
  {
    if(y < level) Order = init_order;
  }
  else if(init_config=="boxatwall")
  {
    if (BC==201 or BC==501 or BC==4)
    {
      if(x < level and diff(LY/2, y) < level/2) Order = init_order;
    }
    else
    {
      if(y < level and diff(LX/2, x) < level) Order = init_order;
    }
  }
  else if(init_config=="cuttingatwall")
  {
    if (BC==201 or BC==501 or BC==4)
    {
      if(pow(x, 2) + pow(wrap(diff(y, LY/2+int(radius*1.5)), LY), 2) <= radius*radius) Order = init_order;
      if(pow(x, 2) + pow(wrap(diff(y, LY/2-int(radius*1.5)), LY), 2) <= radius*radius) Order = init_order;
    }
    else
    {
      if(y < level and diff(LX/2, x) < level) Order = init_order;
    }
  }
  else if(init_config=="circleatwall")
  {
    if (BC==201 or BC==501)
    {
      if(pow(x, 2) + pow(wrap(diff(y, LY/2), LY), 2) <= radius*radius) Order = init_order;
    }
    else
    {
      if(pow(wrap(diff(x, LX/2), LX),2) + pow(diff(y, 0*LY), 2) <= radius*radius) Order = init_order;
    }
  }
    else if(init_config=="wettedwall")
  {
    if (BC==201 or BC==501)
    {
      if(x<level) Order = init_order;
    }
    else
    {
      if(y<level) Order = init_order;
    }
  }
  else
    throw error_msg("error: initial configuration '", init_config, "' unknown.");


  theta   = angle + noise*M_PI*(random_real() - .5);
	
  Px[k] = Order*(cos(theta));
  Py[k] = Order*(sin(theta));
  phi[k]  = Order*conc + noise*(random_real() - .5);;
  totalphi += phi[k];
  ux[k] = uy[k] = ux_phi[k] = uy_phi[k] = 0;
  // compute totals for later checks
  ptot += phi[k];
}
void DryPolarLyotropic::Configure()
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

void DryPolarLyotropic::UpdatePolarQuantitiesAtNode(unsigned k)
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

  const double p        = phi[k];
  const double dxPhi    = derivX   (phi, d, sB);
  const double dyPhi    = derivY   (phi, d, sB);
  const double del2p    = laplacian(phi, d, sD);

  const double term = p - p4;
  const double hx = CC*term*4*ps*px + Kn*laplacian(Px, d, sD)
	  	  + Kp*( 1.*ps*laplacian(Px, d, sD)
         	  	+2.*px*(dxpx*dxpx+dypx*dypx-dxpy*dxpy-dypy*dypy)
         	  	+4.*py*(dxpx*dxpy+dypx*dypy));
  const double hy = CC*term*4*ps*py + Kn*laplacian(Py, d, sD)
	  	  + Kp*( 1.*ps*laplacian(Py, d, sD)
         	        +2.*py*(dxpy*dxpy+dypy*dypy-dxpx*dxpx-dypx*dypx)
         		+4.*px*(dxpx*dxpy+dypx*dypy));

  const double mu = Aphi*p*(p-1.)*(p+1.) + CC*term - Kphi*del2p;// 

  // computation of sigma...
  const double sigmaB = .5*Aphi*p*p*(-1.+.5*p*p) - mu*p + (backflow_on? .5*CC*term*term : 0);
  const double sigmaF = .5*Kphi*(dyPhi*dyPhi-dxPhi*dxPhi);
  const double sigmaS =  - Kphi*dxPhi*dyPhi;

  // backflow implemeted as of Giomi & Marchetti, Soft Matter (2012): https://doi.org/10.1039/C1SM06077E with \bar{lambda}=\lambda
  const double sigmaxx = -.5*zeta*(px*px-py*py) + (beta == 0 ? 1 : 0)*( ( backflow_on? -2.*xi*px*hx-xi*py*hy : 0) ) + sigmaB + sigmaF;
  const double sigmayy = -.5*zeta*(py*py-px*px) + (beta == 0 ? 1 : 0)*( ( backflow_on? -2.*xi*py*hy-xi*px*hx : 0) ) + sigmaB - sigmaF;
  const double sigmaxy = -zeta*px*py + (beta == 0 ? 1 : 0)*( ( backflow_on? .5*(1.-xi)*px*hy - .5*(1.+xi)*py*hx :0) ) + sigmaS;
  const double sigmayx = -zeta*py*px + (beta == 0 ? 1 : 0)*( ( backflow_on? .5*(1.-xi)*py*hx - .5*(1.+xi)*px*hy :0) ) + sigmaS;

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
  MU[k]      =  mu;

}
void DryPolarLyotropic::UpdatePolarQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdatePolarQuantitiesAtNode(k);
}

void DryPolarLyotropic::UpdateFluidQuantitiesAtNode(unsigned k)
{
  // array placeholders for current node
  const auto& d = get_neighbours(k);

  const double dxSxx = derivX(sigmaXX, d, sB);
  const double dySxy = derivY(sigmaXY, d, sB);
  const double dxSyx = derivX(sigmaYX, d, sB);
  const double dySyy = derivY(sigmaYY, d, sB);
  const double Fx = dxSxx + dySxy;
  const double Fy = dxSyx + dySyy;
  
  FFx[k] = Fx;
  FFy[k] = Fy;

  const double p   = phi[k];
  const double px = Px[k];
  const double py = Py[k];
  const double hx = Hx[k];
  const double hy = Hy[k];

  // compute velocities
  const double vx = (FFx[k] + (p > 0.9 ? 1 : 0)*alpha*px - (p > 0.9 ? 1 : 0)*beta*hx)/friction;
  const double vy = (FFy[k] + (p > 0.9 ? 1 : 0)*alpha*py - (p > 0.9 ? 1 : 0)*beta*hy)/friction;

  // transfer to arrays
  ux[k]      =  vx;
  uy[k]      =  vy;
  ux_phi[k]  =  vx*p + Vphi*px*p;
  uy_phi[k]  =  vy*p + Vphi*py*p;
}
void DryPolarLyotropic::UpdateFluidQuantities()
{
  double sum = 0;

  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k){
    sum = sum + phi[k];
    UpdateFluidQuantitiesAtNode(k);
  }
  countphi = sum;
}

void DryPolarLyotropic::UpdatePolarFieldsAtNode(unsigned k, bool first)
{  
  const auto& d = get_neighbours(k);
  
  const double vx = ux[k];
  const double vy = uy[k];

  const double px = Px[k];
  const double py = Py[k];
  const double hx = Hx[k];
  const double hy = Hy[k];

  const double p   = phi[k];

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
		    + (p > 0.9 ? 1 : 0)*beta*vx;
  const double Dy = hy/gamma 
	  	    - (vx + Vp*px)*dxpy - (vy + Vp*py)*dypy
                    + (beta == 0 ? 1 : 0)*(xi*(dyuy*py + .5*(dxuy+dyux)*px) + .5*(dxuy-dyux)*px )
		    + (p > 0.9 ? 1 : 0)*beta*vy;

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
void DryPolarLyotropic::UpdatePolarFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdatePolarFieldsAtNode(k, first);
}

void DryPolarLyotropic::UpdateFluidFieldsAtNode(unsigned k, bool first)
{  
  const auto& d = get_neighbours(k);
  
  //const double vx = ux[k];
  //const double vy = uy[k];
  //const double px = Px[k];
  //const double py = Py[k];
  //const double hx = Hx[k];
  //const double hy = Hy[k];
  //const double p   = phi[k];

  const double del2mu = laplacian(MU, d, sD);
  const double pFlux = derivX(ux_phi, d, sB) + derivY(uy_phi, d, sB);
  const double Dp = GammaPhi*del2mu - pFlux - ( conserve_phi ? (countphi-totalphi)/DomainSize : 0 );


  if (first){
    phn[k]     = phi[k]  + .5*Dp;
    phi_tmp[k] = phi[k]  +    Dp;
  }
  else{
    phi_tmp[k] = phn[k]  + .5*Dp;
  }

}
void DryPolarLyotropic::UpdateFluidFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateFluidFieldsAtNode(k, first);

  swap(phi.get_data(), phi_tmp.get_data());
}

void DryPolarLyotropic::BoundaryConditionsFields()
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
      phi .ApplyNeumannChannel();
      break;
    // box
    case 3:
    case 4:
      Px.ApplyNeumann();
      Py.ApplyNeumann();
      phi .ApplyNeumann();
      break;
    // pbc with bdry layer
    default:
      Px.ApplyPBC();
      Py.ApplyPBC();
      phi .ApplyPBC();
  }
}
void DryPolarLyotropic::BoundaryConditionsFields2()
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
      uy_phi .ApplyDirichletChannel(0);
      ux_phi .ApplyDirichletChannel(0);

      MU     .ApplyNeumannChannel();
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
      uy_phi .ApplyDirichlet(0);
      ux_phi .ApplyDirichlet(0);

      MU     .ApplyNeumann();
      sigmaXX.ApplyNeumann();
      sigmaYY.ApplyNeumann();
      sigmaYX.ApplyNeumann();
      sigmaXY.ApplyNeumann();
      break;
    // pbc with bdry layer
    default:
      ux     .ApplyPBC();
      uy     .ApplyPBC();
      ux_phi .ApplyPBC();
      uy_phi .ApplyPBC();
      MU     .ApplyPBC();
      sigmaXX.ApplyPBC();
      sigmaYY.ApplyPBC();
      sigmaYX.ApplyPBC();
      sigmaXY.ApplyPBC();
  }
}

void DryPolarLyotropic::Step()
{
  // boundary conditions for primary fields
  BoundaryConditionsFields();
  // predictor step
  UpdatePolarQuantities();
  UpdateFluidQuantities();
  
  BoundaryConditionsFields2();

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

void DryPolarLyotropic::RuntimeChecks()
{
  // check that phi is conserved
  {
    double pcheck = 0;
    for(unsigned k=0; k<DomainSize; ++k)
        pcheck += phi[k];
    cout << "pcheck: " << pcheck << "/" << ptot << '\n';
    if(abs(ptot-pcheck)>1)
      throw error_msg("phi is not conserved (", ptot, "/", pcheck, ")");
  }
}

option_list DryPolarLyotropic::GetOptions()
{
  // model specific options
  opt::options_description model_options("Model options");
  model_options.add_options()
    ("gamma", opt::value<double>(&gamma),
     "Q-tensor mobility")
    ("GammaPhi", opt::value<double>(&GammaPhi),
     "binary mobility")
    ("xi", opt::value<double>(&xi),
     "tumbling/aligning parameter")
     ("beta", opt::value<double>(&beta),
     "alignment to velocity parameter")
    ("friction", opt::value<double>(&friction),
     "friction from confinement")
    ("Aphi", opt::value<double>(&Aphi),
     "binary fluid bulk constant")
    ("CC", opt::value<double>(&CC),
     "coupling constant")
    ("Kn", opt::value<double>(&Kn),
     "nematic elastic constant")
    ("Kp", opt::value<double>(&Kp),
     "polar elastic constant")
    ("Kphi", opt::value<double>(&Kphi),
     "binary gradient constant")
    ("zeta", opt::value<double>(&zeta),
     "activity parameter")
    ("Vphi", opt::value<double>(&Vphi),
     "self-advection parameter in phi")
    ("Vp", opt::value<double>(&Vp),
     "self-advection parameter")
    ("alpha", opt::value<double>(&alpha),
     "monopole activity parameter")
    ("npc", opt::value<unsigned>(&npc),
     "number of correction steps for the predictor-corrector method")
    ("backflow_on", opt::value<bool>(&backflow_on),
     "Backflow flag")
    ("n_preinit", opt::value<int>(&n_preinit),
     "number of preinitialization steps");

  // init config options
  opt::options_description config_options("Initial configuration options");
  config_options.add_options()
    ("config", opt::value<string>(&init_config),
     "initial configuration")
    ("level", opt::value<double>(&level),
     "starting thickness of the nematic region")
    ("conc", opt::value<double>(&conc),
     "starting phi concentration of nematic region")
    ("radius", opt::value<double>(&radius),
     "radius of the initial circle")
    ("angle", opt::value<double>(&angle_deg),
     "initial angle to x direction (in degrees)")
    ("noise", opt::value<double>(&noise),
     "size of initial variations")
    ("initial-order", opt::value<double>(&init_order),
     "initial order of the polarisation field");

  return { model_options, config_options };
}
