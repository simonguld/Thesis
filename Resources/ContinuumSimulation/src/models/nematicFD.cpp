#include "header.hpp"
#include "models/nematicFD.hpp"
#include "error_msg.hpp"
#include "random.hpp"
#include "lb.hpp"
#include "tools.hpp"

using namespace std;
namespace opt = boost::program_options;

// from main.cpp:
extern unsigned nthreads, nsubsteps;
extern double time_step;

NematicFD::NematicFD(unsigned LX, unsigned LY, unsigned BC)
  : Model(LX, LY, BC, BC==0 ? GridType::Periodic : GridType::Layer)
{}

void NematicFD::Initialize()
{
  // initialize variables
  angle = angle_deg*M_PI/180.;
  eta = 1./3.*(tau- .5)*rho;

  // allocate memory
  
  QQxx.SetSize(LX, LY, Type);
  QQyx.SetSize(LX, LY, Type);
  QNxx.SetSize(LX, LY, Type);
  QNyx.SetSize(LX, LY, Type);
  n.SetSize(LX, LY, Type);
  ux.SetSize(LX, LY, Type);
  uy.SetSize(LX, LY, Type);

  HHxx.SetSize(LX, LY, Type);
  HHyx.SetSize(LX, LY, Type);
  dxQQxx.SetSize(LX, LY, Type);
  dyQQxx.SetSize(LX, LY, Type);
  dxQQyx.SetSize(LX, LY, Type);
  dyQQyx.SetSize(LX, LY, Type);
  dxux.SetSize(LX, LY, Type);
  dyux.SetSize(LX, LY, Type);
  dxuy.SetSize(LX, LY, Type);
  dyuy.SetSize(LX, LY, Type);
  del2ux.SetSize(LX, LY, Type);
  del2uy.SetSize(LX, LY, Type);
  dxn.SetSize(LX, LY, Type);
  dyn.SetSize(LX, LY, Type);

  sigmaXX.SetSize(LX, LY, Type);
  sigmaYY.SetSize(LX, LY, Type);
  sigmaYX.SetSize(LX, LY, Type);
  sigmaXY.SetSize(LX, LY, Type);

  FFx.SetSize(LX, LY, Type);
  FFy.SetSize(LX, LY, Type);
  advX.SetSize(LX, LY, Type);
  advY.SetSize(LX, LY, Type);

  if(nsubsteps>1)
    throw error_msg("time stepping not implemented for this model"
                    ", please set nsubsteps=1.");

  Q_fluct = (Q_kBT!=0);
  u_fluct = (u_kBT!=0);

  //set_seed(121998);
}

void NematicFD::ConfigureAtNode(unsigned k)
{
  const double theta = angle + noise*M_PI*(random_real()-.5);
  QQxx[k] = cos(2*theta);
  QQyx[k] = sin(2*theta);

  ux[k] = 2e-3*(random_real()-0.5);
  uy[k] = 2e-3*(random_real()-0.5);
  n[k]  = rho;

  ntot = ntot + n[k];
  
}
void NematicFD::Configure()
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
  cout << " ... Preinitialization done. ";

  InsertPlusDefect(100, 100, 100, 0);
  //InsertMinusDefect(300, 200, 50, 0);
}

void NematicFD::UpdateNematicQuantitiesAtNode(unsigned k)
{
  const auto& d = get_neighbours(k);

  const double Qxx = QQxx[k];
  const double Qyx = QQyx[k];

  // compute derivatives etc.
  const double del2Qxx  = laplacian(QQxx,  d, sD);
  const double dxQxx    = derivX   (QQxx,  d, sB);
  const double dyQxx    = derivY   (QQxx,  d, sB);
  const double del2Qyx  = laplacian(QQyx,  d, sD);
  const double dxQyx    = derivX   (QQyx,  d, sB);
  const double dyQyx    = derivY   (QQyx,  d, sB);
 
  const double term = 1. - Qxx*Qxx - Qyx*Qyx;
  const double Hxx = CC*term*Qxx + LL*del2Qxx;
  const double Hyx = CC*term*Qyx + LL*del2Qyx;

  const double sigmaB = backflow_on? .5*CC*term*term : 0;
  const double sigmaF = (backflow_on? 2*xi*( (Qxx*Qxx-1.)*Hxx + Qxx*Qyx*Hyx ) : 0)
                        - (preinit_flag? 0 : zeta*Qxx )
                        + (backflow_on? LL*(dyQxx*dyQxx+dyQyx*dyQyx-dxQxx*dxQxx-dxQyx*dxQyx) : 0);
  const double sigmaS = (backflow_on? 2*xi*(Qyx*Qxx*Hxx + (Qyx*Qyx-1)*Hyx) : 0) 
                        - (preinit_flag? 0 : zeta*Qyx )
                        - (backflow_on? 2*LL*(dxQxx*dyQxx+dxQyx*dyQyx) : 0);
  const double sigmaA = backflow_on? 2*(Qxx*Hyx - Qyx*Hxx) : 0;

  HHxx[k]    =  Hxx;
  HHyx[k]    =  Hyx;
  dxQQxx[k]  =  dxQxx;
  dxQQyx[k]  =  dxQyx;
  dyQQxx[k]  =  dyQxx;
  dyQQyx[k]  =  dyQyx;
  sigmaXX[k] =  sigmaF + sigmaB;
  sigmaYY[k] = -sigmaF + sigmaB;
  sigmaXY[k] =  sigmaS + sigmaA;
  sigmaYX[k] =  sigmaS - sigmaA;
}
void NematicFD::UpdateFluidQuantitiesAtNode(unsigned k)
{
  const auto& d = get_neighbours(k);
  
  const double dxvx = derivX(ux, d, sB);
  const double dyvx = derivY(ux, d, sB);
  const double dxvy = derivX(uy, d, sB);
  const double dyvy = derivY(uy, d, sB);
  const double del2vx = laplacian(ux, d, sD);
  const double del2vy = laplacian(uy, d, sD);

  const double dxnn = derivX(n, d, sB);
  const double dynn = derivY(n, d, sB);

  dxux[k]    =  dxvx;
  dyux[k]    =  dyvx;
  dxuy[k]    =  dxvy;
  dyuy[k]    =  dyvy;  
  dxn[k]     =  dxnn;
  dyn[k]     =  dynn;

  del2ux[k] = del2vx;
  del2uy[k] = del2vy;
  
}

void NematicFD::UpdateNematicFieldsAtNode(unsigned k, bool first)
{
  const double vx = ux[k];
  const double vy = uy[k];
  const double Qxx = QQxx[k];
  const double Qyx = QQyx[k];
  const double Hxx = HHxx[k];
  const double Hyx = HHyx[k];
  const double dxQxx = dxQQxx[k];
  const double dyQxx = dyQQxx[k];
  const double dxQyx = dxQQyx[k];
  const double dyQyx = dyQQyx[k];

  const double dxvx = dxux[k];
  const double dyvx = dyux[k];
  const double dxvy = dxuy[k];
  const double dyvy = dyuy[k];
  
  const double expansion = dxvx + dyvy;
  const double shear     = .5*(dxvy + dyvx);
  const double vorticity = .5*(dxvy - dyvx);
  const double traceQL   = Qxx*(dxvx - dyvy) + 2*Qyx*shear;


  const double Dxx = Gamma*Hxx - vx*dxQxx - vy*dyQxx - 2*vorticity*Qyx
    + xi*((Qxx+1)*(2*dxvx-traceQL) +2*Qyx*shear -expansion);
  const double Dyx = Gamma*Hyx - vx*dxQyx - vy*dyQyx + 2*vorticity*Qxx
    + xi*( Qyx*(expansion-traceQL) + 2*shear); 

  if(first)
  {   
    QNxx[k] = QQxx[k] + .5*Dxx*time_step/nsteps_Q;
    QNyx[k] = QQyx[k] + .5*Dyx*time_step/nsteps_Q;
    QQxx[k] = QNxx[k] + Dxx*time_step/nsteps_Q;
    QQyx[k] = QNyx[k] + Dyx*time_step/nsteps_Q;
  }
  else
  {
    QQxx[k] = QNxx[k] + .5*Dxx*time_step/nsteps_Q;
    QQyx[k] = QNyx[k] + .5*Dyx*time_step/nsteps_Q;
  }
}
void NematicFD::UpdateFluidFieldsAtNode(unsigned k, bool first)
{
  const auto& d = get_neighbours(k);

  const double nn = n[k];
  const double vx = ux[k];
  const double vy = uy[k];

  const double dxvx = dxux[k];
  const double dyvx = dyux[k];
  const double dxvy = dxuy[k];
  const double dyvy = dyuy[k];

  const double dxnn = dxn[k];
  const double dynn = dyn[k];

  const double dxSxx = derivX(sigmaXX, d, sB);
  const double dySxy = derivY(sigmaXY, d, sB);
  const double dxSyx = derivX(sigmaYX, d, sB);
  const double dySyy = derivY(sigmaYY, d, sB);

  const double expansion = dxvx + dyvy;
  const double viscX = eta * del2ux[k];
  const double viscY = eta * del2uy[k];

  const double Fx = -0.3*dxnn+ dxSxx + dySxy - friction*vx + viscX;
  const double Fy = -0.3*dynn+ dxSyx + dySyy - friction*vy + viscY;

  const double Dn = -vx*dxnn - vy*dynn - nn*expansion;
  const double Dux = Fx/nn - vx*dxvx - vy*dyvx;
  const double Duy = Fy/nn - vx*dxvy - vy*dyvy;

  if(first)
  {
    n[k] = n[k] + time_step*Dn/nsteps_Fl;
    ux[k] = ux[k] + time_step*Dux/nsteps_Fl;
    uy[k] = uy[k] + time_step*Duy/nsteps_Fl; 
  }

  //NSE Equation debugging counters
  FFx[k] = Fx;
  FFy[k] = Fy;
  advX[k] = advX[k] + time_step*nn*(Dux+ vx*dxvx + vy*dyvx)/nsteps_Fl;
  advY[k] = advY[k] + time_step*nn*(Duy+ vx*dxvy + vy*dyvy)/nsteps_Fl;
}

void NematicFD::UpdateNematicQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateNematicQuantitiesAtNode(k);
}
void NematicFD::UpdateFluidQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateFluidQuantitiesAtNode(k);
}

void NematicFD::UpdateNematicFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateNematicFieldsAtNode(k, first);
}
void NematicFD::UpdateFluidFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateFluidFieldsAtNode(k, first);
}

void NematicFD::BoundaryConditionsFields()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // free-slip channel
    case 1:
    {
      auto apply_bc = [](ScalarField& field) {
        // pbc on the left and right walls
        field.ApplyPBC(PBCWall::LeftRight);
        // Neumann on the fron and back walls
        field.ApplyNeumann(Wall::Front);
        field.ApplyNeumann(Wall::Back);
        // corners
        field.ApplyNeumann(Corner::RightBack, Wall::Back);
        field.ApplyNeumann(Corner::RightFront, Wall::Front);
        field.ApplyNeumann(Corner::LeftBack, Wall::Back);
        field.ApplyNeumann(Corner::LeftFront, Wall::Front);
      };

      apply_bc(QQxx);
      apply_bc(QQyx);
      break;
    }
    // pbc with bdry layer
    default:
      QQxx.ApplyPBC();
      QQyx.ApplyPBC();
  }
}
void NematicFD::BoundaryConditionsFields2()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // free-slip channel
    case 1:
    {
      auto apply_bc = [](ScalarField& field) {
        // pbc on the left and right walls
        field.ApplyPBC(PBCWall::LeftRight);
        // Neumann on the fron and back walls
        field.ApplyNeumann(Wall::Front);
        field.ApplyNeumann(Wall::Back);
        // corners
        field.ApplyNeumann(Corner::RightBack, Wall::Back);
        field.ApplyNeumann(Corner::RightFront, Wall::Front);
        field.ApplyNeumann(Corner::LeftBack, Wall::Back);
        field.ApplyNeumann(Corner::LeftFront, Wall::Front);
      };

      apply_bc(ux);
      apply_bc(uy);
      apply_bc(sigmaXX);
      apply_bc(sigmaYY);
      apply_bc(sigmaYX);
      apply_bc(sigmaXY);
      break;
    }
    // pbc with bdry layer
    default:
      ux.ApplyPBC();
      uy.ApplyPBC();
      n.ApplyPBC();

      sigmaXX.ApplyPBC();
      sigmaYY.ApplyPBC();
      sigmaYX.ApplyPBC();
      sigmaXY.ApplyPBC();
  }
}

void NematicFD::Step()
{  
  BoundaryConditionsFields();
  BoundaryConditionsFields2();

  UpdateNematicQuantities();
  UpdateFluidQuantities();  
  
  for (int lcv = 1; lcv<=nsteps_Q; lcv++){
    this->UpdateNematicFields(true);
    BoundaryConditionsFields();
    UpdateNematicQuantities();    
  }
  
  ResetAdvectionCounters();
  for (int lcv = 1; lcv<=nsteps_Fl; lcv++){
    this->UpdateFluidFields(true);
    BoundaryConditionsFields2();
    UpdateFluidQuantities();
  }

  for(unsigned n=1; n<=npc; ++n)
  {
    for (int lcv = 1; lcv<=nsteps_Q; lcv++){
      this->UpdateNematicFields(false);
      BoundaryConditionsFields();
      UpdateNematicQuantities();    
    }
  }
}

//utilities and checks
void NematicFD::RuntimeChecks()
{
  // check that the sum of density is constant
  {
    double ncheck = 0;
    for(unsigned k=0; k<DomainSize; k++)
        ncheck = ncheck + n[k];
    if(abs(ntot-ncheck)>0.01*ntot)
      throw error_msg("n is not conserved (", ncheck, "/", ntot, ")");
    if (conserveDensity){
      for (unsigned lcv = 0; lcv<DomainSize; lcv++)
        n[lcv] += (ntot-ncheck)*1.0/DomainSize;
    }
  }

  // check that density is not negative
  {
    for(unsigned k=0; k<DomainSize; k++){
      if (n[k]<0)
        //cry
        throw error_msg("n is negative at (", GetXPosition(k), ",", GetYPosition(k), ")");
    
    }
  }
}
void NematicFD::ResetAdvectionCounters()
{
  for (unsigned k = 0; k< DomainSize; k++){
    advX[k] = 0;
    advY[k] = 0;
  }
}

option_list NematicFD::GetOptions()
{
  // model specific options
  opt::options_description model_options("Model options");
  model_options.add_options()
    ("Gamma", opt::value<double>(&Gamma),
     "Mobility")
    ("xi", opt::value<double>(&xi),
     "tumbling/aligning parameter")
    ("tau", opt::value<double>(&tau),
     "viscosity")
    ("rho", opt::value<double>(&rho),
     "fluid density")
    ("friction", opt::value<double>(&friction),
     "friction")
    ("CC", opt::value<double>(&CC),
     "coupling constant")
    ("LL", opt::value<double>(&LL),
     "elastic constant")
    ("zeta", opt::value<double>(&zeta),
     "activity parameter")      
    ("backflow_on", opt::value<bool>(&backflow_on),
     "Backflow flag")
    ("npc", opt::value<unsigned>(&npc),
     "number of correction steps for the predictor-corrector method")
    ("n_preinit", opt::value<int>(&n_preinit),
     "number of preinitialization steps")
    ("nsteps_Q", opt::value<int>(&nsteps_Q),
     "number of Q steps for every v step")
    ("nsteps_Fl", opt::value<int>(&nsteps_Fl),
     "number of v substeps for every v step");

  opt::options_description config_options("Initial configuration options");
  config_options.add_options()
    ("angle", opt::value<double>(&angle_deg),
     "initial angle to x direction (in degrees)")
    ("noise", opt::value<double>(&noise),
     "size of initial variations");

  return { model_options, config_options };
}


//other random functions: Defect Insertion
void NematicFD::InsertPlusDefect(unsigned x_def, unsigned y_def, int def_size, double angle){
  double theta = 0;
  for(unsigned k=0; k<DomainSize; ++k){
    unsigned x = GetXPosition(k);
    unsigned y = GetYPosition(k);
    double xtemp=(double) x- x_def;
    double ytemp=(double) y- y_def;  
    
    double rtemp = sqrt(xtemp*xtemp+ytemp*ytemp);
    if ((rtemp < def_size) && (rtemp > 1)){
      double phi_def = acos(xtemp/rtemp);
      if (ytemp < 0)  phi_def = -phi_def;
      theta = angle + phi_def/2.0;
      QQxx[k] = cos(2*theta);
      QQyx[k] = sin(2*theta);      
    } 
  }
  
}
void NematicFD::InsertMinusDefect(unsigned x_def, unsigned y_def, int def_size, double angle){
  double theta = 0;
  for(unsigned k=0; k<DomainSize; ++k){
    unsigned x = GetXPosition(k);
    unsigned y = GetYPosition(k);
    double xtemp=(double) x- x_def;
    double ytemp=(double) y- y_def;
    
    double rtemp = sqrt(xtemp*xtemp+ytemp*ytemp);

    if ((rtemp < def_size) && (rtemp > 1)){
      double phi_def = acos(xtemp/rtemp);
      if (ytemp < 0)  phi_def = -phi_def;
      theta = angle - phi_def/2.0;
      QQxx[k] = cos(2*theta);
      QQyx[k] = sin(2*theta);       
    } 
  }
  
}
void NematicFD::InsertMinus1Defect(unsigned x_def, unsigned y_def, int def_size, double angle){
  double theta = 0;
  for(unsigned k=0; k<DomainSize; ++k){
    unsigned x = GetXPosition(k);
    unsigned y = GetYPosition(k);
    double xtemp=(double) x- x_def;
    double ytemp=(double) y- y_def;
    
    double rtemp = sqrt(xtemp*xtemp+ytemp*ytemp);

    if ((rtemp < def_size) && (rtemp > 1)){
      double phi_def = acos(xtemp/rtemp);
      if (ytemp < 0)  phi_def = -phi_def;
      theta = angle - phi_def;
      QQxx[k] = cos(2*theta);
      QQyx[k] = sin(2*theta);       
    } 
  }
  
}
void NematicFD::InsertPlus1Defect(unsigned x_def, unsigned y_def, int def_size, double angle){
  double theta = 0;
  for(unsigned k=0; k<DomainSize; ++k){
    unsigned x = GetXPosition(k);
    unsigned y = GetYPosition(k);
    double xtemp=(double) x- x_def;
    double ytemp=(double) y- y_def;  
    
    double rtemp = sqrt(xtemp*xtemp+ytemp*ytemp);
    if ((rtemp < def_size) && (rtemp > 1)){
      double phi_def = acos(xtemp/rtemp);
      if (ytemp < 0)  phi_def = -phi_def;
      theta = angle + phi_def;
      QQxx[k] = cos(2*theta);
      QQyx[k] = sin(2*theta);      
    } 
  }
  
}

