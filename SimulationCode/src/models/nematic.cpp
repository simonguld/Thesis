#include "header.hpp"
#include "models/nematic.hpp"
#include "error_msg.hpp"
#include "random.hpp"
#include "lb.hpp"
#include "tools.hpp"

using namespace std;
namespace opt = boost::program_options;

// from main.cpp:
extern unsigned nthreads, nsubsteps;
extern double time_step;

Nematic::Nematic(unsigned LX, unsigned LY, unsigned BC)
  : Model(LX, LY, BC, BC==0 ? GridType::Periodic : GridType::Layer)
{}

void Nematic::Initialize()
{
  // initialize variables
  angle = angle_deg*M_PI/180.;

  // allocate memory
  ff.SetSize(LX, LY, Type);
  fn.SetSize(LX, LY, Type);
  ff_tmp.SetSize(LX, LY, Type);
  fn_tmp.SetSize(LX, LY, Type);
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
  sigmaXX.SetSize(LX, LY, Type);
  sigmaYY.SetSize(LX, LY, Type);
  sigmaYX.SetSize(LX, LY, Type);
  sigmaXY.SetSize(LX, LY, Type);

  FFx.SetSize(LX, LY, Type);
  FFy.SetSize(LX, LY, Type);

  E_kin.SetSize(LX, LY, Type);
  PE_LC.SetSize(LX, LY, Type);
  dE_kin.SetSize(LX, LY, Type);
  dPE_LC.SetSize(LX, LY, Type);

  if(nsubsteps>1)
    throw error_msg("time stepping not implemented for this model"
                    ", please set nsubsteps=1.");

  Q_fluct = (Q_kBT!=0);
  u_fluct = (u_kBT!=0);

  //set_seed(35256);
}

void Nematic::ConfigureAtNode(unsigned k)
{
  // add noise (for meta-stable configs)
  // theta is the angle of the director
  const double theta = angle + noise*M_PI*(random_real()-.5);
  QQxx[k] = cos(2*theta);
  QQyx[k] = sin(2*theta);
  // equilibrium dist
  ux[k] = uy[k] = 0;
  n[k]  = rho;
  ff[k] = GetEquilibriumDistribution(ux[k], uy[k], n[k]);
  // compute totals for later checks
  ftot  = accumulate(begin(ff[k]), end(ff[k]), ftot);

  FFx[k] = 0.0;
  FFy[k] = 0.0;
}
void Nematic::Configure()
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

  //InsertPlusDefect(100, 100, 100, 0);
  //InsertMinusDefect(300, 200, 50, 0);
}

void Nematic::UpdateNematicQuantitiesAtNode(unsigned k)
{
  // array placeholders for current node
  const auto& d = get_neighbours(k);  
  // Q-tensor
  const double Qxx = QQxx[k];
  const double Qyx = QQyx[k];
  
  // compute derivatives etc.
  const double del2Qxx  = laplacian(QQxx,  d, sD);
  const double dxQxx    = derivX   (QQxx,  d, sB);
  const double dyQxx    = derivY   (QQxx,  d, sB);
  const double del2Qyx  = laplacian(QQyx,  d, sD);
  const double dxQyx    = derivX   (QQyx,  d, sB);
  const double dyQyx    = derivY   (QQyx,  d, sB);

  // computation of the chemical potential and molecular field...
  // ...term that couples the binary phase to the degree of nematic order
  const double term = 1. - Qxx*Qxx - Qyx*Qyx;
  // ...molecular field
  const double Hxx = CC*term*Qxx + LL*del2Qxx;
  const double Hyx = CC*term*Qyx + LL*del2Qyx;

  // transfer to arrays  
  HHxx[k]    =  Hxx;
  HHyx[k]    =  Hyx;
  dxQQxx[k]  =  dxQxx;
  dxQQyx[k]  =  dxQyx;
  dyQQxx[k]  =  dyQxx;
  dyQQyx[k]  =  dyQyx;

  // computation of sigma...
  // ... on-diagonal stress components
  const double sigmaB = backflow_on? .5*CC*term*term : 0;
  const double sigmaF = (backflow_on? 2*xi*( (Qxx*Qxx-1.)*Hxx + Qxx*Qyx*Hyx ) : 0)
                        - (preinit_flag? 0 : zeta*Qxx )
                        + (backflow_on? LL*(dyQxx*dyQxx+dyQyx*dyQyx-dxQxx*dxQxx-dxQyx*dxQyx) : 0);
  // .. off-diagonal stress components
  const double sigmaS = (backflow_on? 2*xi*(Qyx*Qxx*Hxx + (Qyx*Qyx-1)*Hyx) : 0) 
                        - (preinit_flag? 0 : zeta*Qyx )
                        - (backflow_on? 2*LL*(dxQxx*dyQxx+dxQyx*dyQyx) : 0);
  const double sigmaA = backflow_on? 2*(Qxx*Hyx - Qyx*Hxx) : 0;


  sigmaXX[k] =  sigmaF + sigmaB;
  sigmaYY[k] = -sigmaF + sigmaB;
  sigmaXY[k] =  sigmaS + sigmaA;
  sigmaYX[k] =  sigmaS - sigmaA;

  if (outputEnergyCalc){
    const double PE_LC_k = 0.5*(CC*term*term + LL*dxQxx*dxQxx+dyQxx*dyQxx+dxQyx*dxQyx+dyQyx*dyQyx);
    const double d_PE_LC_k = PE_LC[k] - PE_LC_k;

    PE_LC[k] = PE_LC_k;
    dPE_LC[k] = d_PE_LC_k;
  }
}
void Nematic::UpdateNematicFieldsAtNode(unsigned k, bool first)
{
  // pointer to neighbours
  const auto& d = get_neighbours(k);

  // store data in arrays in local varibles
  //const double nn = n[k];
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

  const double dxux = derivX(ux, d, sB);
  const double dyux = derivY(ux, d, sB);
  const double dxuy = derivX(uy, d, sB);
  const double dyuy = derivY(uy, d, sB);

  const double expansion = dxux + dyuy;
  const double shear     = .5*(dxuy + dyux);
  const double vorticity = .5*(dxuy - dyux);
  const double traceQL   = Qxx*(dxux - dyuy) + 2*Qyx*shear;

  //components of the Beris-Edwards equation
  const double Dxx = Gamma*Hxx - vx*dxQxx - vy*dyQxx - 2*vorticity*Qyx
    + xi*((Qxx+1)*(2*dxux-traceQL) +2*Qyx*shear -expansion);

  const double Dyx = Gamma*Hyx - vx*dxQyx - vy*dyQyx + 2*vorticity*Qxx
    + xi*( Qyx*(expansion-traceQL) + 2*shear);
  
  
  if(first)
  {
    double Qxx_noise = 0., Qxy_noise = 0.;    

    if(Q_fluct)
    {
      static const double Q_stren = sqrt(Gamma*Q_kBT);
      Qxx_noise = Q_stren*random_real();
      Qxy_noise = Q_stren*random_real();
    }
    if(isComp){
      Qxx_noise = Qxx_noise - (dxux+dyuy)*Qxx;
      Qxy_noise = Qxy_noise - (dxux+dyuy)*Qyx;
    }

    QNxx[k] = QQxx[k] + .5*Dxx + Qxx_noise;
    QNyx[k] = QQyx[k] + .5*Dyx + Qxy_noise;
    QQxx[k] = QNxx[k] + .5*Dxx;
    QQyx[k] = QNyx[k] + .5*Dyx;
  }
  else
  {
    QQxx[k] = QNxx[k] + .5*Dxx;
    QQyx[k] = QNyx[k] + .5*Dyx;
    
  }
}

void Nematic::UpdateFluidQuantitiesAtNode(unsigned k)
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

  if (outputEnergyCalc){
    const double E_kin_k = 0.5*nn*(vx*vx+vy*vy);
    const double dE_kin_k = E_kin_k - E_kin[k];
    E_kin[k] = E_kin_k;
    dE_kin[k] = dE_kin_k;
  }

}
void Nematic::UpdateFluidFieldsAtNode(unsigned k, bool first)
{
  // pointer to neighbours
  //const auto& d = get_neighbours(k);
  
  const double nn = n[k];
  const double vx = ux[k];
  const double vy = uy[k];

  const double Fx = FFx[k] - friction*vx;
  const double Fy = FFy[k] - friction*vy;
  
  // calculate the equilibrium distribution fe
  const auto fe = GetEquilibriumDistribution(vx, vy, nn);

  if(first)
  {    
    LBNode ff_noise = {0.};

    if(u_fluct)
    {
      const auto ff_stren = sqrt(3.*nn*u_kBT*(2.*tau-1.)/tau/tau);
      ff_noise = GenerateNoiseDistribution(ff_stren);
    }

    for(unsigned v=0; v<lbq; ++v)
    {
      const double Si = isGuo? w[v]*(1-1./(2.*tau))*(Fx*xdir(v) + Fy*ydir(v))/cs2 : 0;

      fn[k][v] = ff[k][v] + .5*(fe[v]-ff[k][v])/tau + 0.5*Si + ff_noise[v];
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

void Nematic::UpdateNematicQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateNematicQuantitiesAtNode(k);
}
void Nematic::UpdateNematicFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateNematicFieldsAtNode(k, first);
}
void Nematic::UpdateFluidQuantities()
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateFluidQuantitiesAtNode(k);
}
void Nematic::UpdateFluidFields(bool first)
{
  #pragma omp parallel for num_threads(nthreads) if(nthreads)
  for(unsigned k=0; k<DomainSize; ++k)
    UpdateFluidFieldsAtNode(k, first);
}

void Nematic::Move()
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
  swap(ff, ff_tmp);
  swap(fn, fn_tmp);
}

void Nematic::BoundaryConditionsLB()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // free-slip channel
    case 1:
    {
      auto apply_bc = [](LBField& field) {
        // pbc on the left and right walls
        field.ApplyPBC(PBCWall::LeftRight);
        // Free-slip on the front and back walls
        field.ApplyFreeSlip(Wall::Front);
        field.ApplyFreeSlip(Wall::Back);
        // corners
        field.ApplyFreeSlip(Corner::RightBack, Wall::Back);
        field.ApplyFreeSlip(Corner::RightFront, Wall::Front);
        field.ApplyFreeSlip(Corner::LeftBack, Wall::Back);
        field.ApplyFreeSlip(Corner::LeftFront, Wall::Front);
      };

      apply_bc(ff);
      apply_bc(fn);

      break;
    }
    case 2:
    { //no-slip channel
      auto apply_bc = [](LBField& field) {
        // pbc on the left and right walls
        field.ApplyPBC(PBCWall::LeftRight);
        // Free-slip on the front and back walls
        field.ApplyNoSlip(Wall::Front);
        field.ApplyNoSlip(Wall::Back);
        // corners
        field.ApplyNoSlip(Corner::RightBack);
        field.ApplyNoSlip(Corner::RightFront);
        field.ApplyNoSlip(Corner::LeftBack);
        field.ApplyNoSlip(Corner::LeftFront);
      };

      apply_bc(ff);
      apply_bc(fn);

      break;
    }
    case 3:
      ff.ApplyFreeSlip();
      fn.ApplyFreeSlip();
      break;
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
void Nematic::BoundaryConditionsFields()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // free-slip channel
    case 1:
    case 2:
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
    case 3:
    case 4:
      QQxx.ApplyNeumann();
      QQyx.ApplyNeumann();
      break;
    // pbc with bdry layer
    default:
      QQxx.ApplyPBC();
      QQyx.ApplyPBC();
  }
}
void Nematic::BoundaryConditionsFields2()
{
  switch(BC)
  {
    // pbc without bdry layer (nothing to do)
    case 0:
      break;
    // free-slip channel
    case 1:
    case 2:
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

    case 3:
    case 4:
      sigmaXX.ApplyNeumann();
      sigmaYY.ApplyNeumann();
      sigmaYX.ApplyNeumann();
      sigmaXY.ApplyNeumann();
      ux.CopyDerivative();
      uy.CopyDerivative();
      break;

    // pbc with bdry layer
    default:
      ux.ApplyPBC();
      uy.ApplyPBC();
      sigmaXX.ApplyPBC();
      sigmaYY.ApplyPBC();
      sigmaYX.ApplyPBC();
      sigmaXY.ApplyPBC();
  }
}

void Nematic::Step()
{
  // boundary conditions for primary fields
  BoundaryConditionsFields();
  // predictor step
  UpdateNematicQuantities();
  UpdateFluidQuantities();
  // boundary conditions for the secondary
  // fields MU, u, sigma, and phi
  BoundaryConditionsFields2();
  // update all the fields
  this->UpdateNematicFields(true);
  this->UpdateFluidFields(true);
  // boundary conditions for
  // the flow before advection
  BoundaryConditionsLB();
  // move LB particles
  Move();
  
  if (outputEnergyCalc){ //note: this will break force balance evaluations for this step
    UpdateFluidQuantities();
  }
  // corrector steps
  for(unsigned n=1; n<=npc; ++n)
  {
    // same thing (no advection)
    BoundaryConditionsFields();
    UpdateNematicQuantities();
    UpdateFluidQuantities();
    BoundaryConditionsFields2();
    this->UpdateNematicFields(false);
    this->UpdateFluidFields(false);
  }
}

void Nematic::RuntimeChecks()
{
  // check that the sum of f is constant
  {
    double fcheck = 0;
    for(unsigned k=0; k<DomainSize; ++k)
        fcheck = accumulate(begin(ff[k]), end(ff[k]), fcheck);
    if(abs(ftot-fcheck)>1)
      throw error_msg("f is not conserved (", ftot, "/", fcheck, ")");
  }
}

option_list Nematic::GetOptions()
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
    ("Q_kBT", opt::value<double>(&Q_kBT),
     "hydrodynamic fluctations strength")
    ("u_kBT", opt::value<double>(&u_kBT),
     "nematic fluctuations strength")
    ("friction", opt::value<double>(&friction),
     "friction")
    ("CC", opt::value<double>(&CC),
     "coupling constant")
    ("LL", opt::value<double>(&LL),
     "elastic constant")
    ("zeta", opt::value<double>(&zeta),
     "activity parameter")
    ("isComp", opt::value<bool>(&isComp),
     "Compressibility (weakly) flag")
    ("backflow_on", opt::value<bool>(&backflow_on),
     "Backflow flag")
    ("isGuo", opt::value<bool>(&isGuo),
     "LB forcing scheme")
    ("outputEnergyCalc", opt::value<bool>(&outputEnergyCalc),
     "output Energy Calcs?")
    ("npc", opt::value<unsigned>(&npc),
     "number of correction steps for the predictor-corrector method")
    ("n_preinit", opt::value<int>(&n_preinit),
     "number of preinitialization steps");

  opt::options_description config_options("Initial configuration options");
  config_options.add_options()
    ("angle", opt::value<double>(&angle_deg),
     "initial angle to x direction (in degrees)")
    ("noise", opt::value<double>(&noise),
     "size of initial variations");

  return { model_options, config_options };
}


//other random functions: Defect Insertion
void Nematic::InsertPlusDefect(unsigned x_def, unsigned y_def, int def_size, double angle){
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
void Nematic::InsertMinusDefect(unsigned x_def, unsigned y_def, int def_size, double angle){
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
void Nematic::InsertMinus1Defect(unsigned x_def, unsigned y_def, int def_size, double angle){
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
void Nematic::InsertPlus1Defect(unsigned x_def, unsigned y_def, int def_size, double angle){
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

