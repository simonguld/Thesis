#ifndef MODELS_NematicFD_HPP_
#define MODELS_NematicFD_HPP_

#include "models.hpp"

class NematicFD : public Model
{
protected: 
  /** Q-Tensor */
  ScalarField QQxx, QNxx, QQyx, QNyx;
  
  /** Velocity */
  ScalarField ux, uy;
  /** Density */
  ScalarField n;

  /** Molecular field */
  ScalarField HHxx, HHyx;
  /** Derivatives */
  ScalarField dxQQxx, dyQQxx, dxQQyx, dyQQyx;
  ScalarField dxux, dyux, dxuy, dyuy, del2ux, del2uy;
  ScalarField dxn, dyn;
  /** Stress tensor */
  ScalarField sigmaXX, sigmaYY, sigmaYX, sigmaXY;

  ScalarField FFx, FFy; //debugging reasons
  ScalarField advX, advY; //debugging reasons
  
  /** Fluid density */
  double rho = 40.;
  /** Fluid parameters */
  double Gamma, xi, tau, friction, LL, CC, zeta, eta;
  

  /** Configuration flags*/
  bool Q_fluct = false, u_fluct = false, isComp = false;
  double Q_kBT = 0, u_kBT = 0; bool backflow_on = true;
  bool conserveDensity = true; double ntot = 0;


  /** Intial configuration */
  std::string init_config;
  /** Number of correction steps in the predictor/corrector scheme */
  unsigned npc = 1;

  
  /** Initial angle and noise */
  double angle_deg, angle, noise;
  int n_preinit = 1000; bool preinit_flag = false;
  int nsteps_Q = 1; int nsteps_Fl = 10;


  /** Update fields using FD algorithms */
  virtual void UpdateNematicFields(bool);
  virtual void UpdateFluidFields(bool);
  
  void ResetAdvectionCounters();

  virtual void UpdateNematicQuantities();
  virtual void UpdateFluidQuantities();

  void UpdateNematicFieldsAtNode(unsigned, bool);
  void UpdateFluidFieldsAtNode(unsigned, bool);
  void UpdateNematicQuantitiesAtNode(unsigned);
  void UpdateFluidQuantitiesAtNode(unsigned);
  
  virtual void BoundaryConditionsFields();
  virtual void BoundaryConditionsFields2();

  //other random functions
  void InsertPlusDefect(unsigned, unsigned, int, double);
  void InsertMinusDefect(unsigned, unsigned, int, double);
  void InsertMinus1Defect(unsigned, unsigned, int, double);
  void InsertPlus1Defect(unsigned, unsigned, int, double);



public:
  NematicFD(unsigned, unsigned, unsigned);

  /** Configure a single node */
  virtual void ConfigureAtNode(unsigned);

  // functions from base class Model
  virtual void Initialize();
  virtual void Step();
  virtual void Configure();
  virtual void RuntimeChecks();
  virtual option_list GetOptions();

  /** Serialization of parameters (do not change) */
  template<class Archive>
  void serialize_params(Archive& ar)
  {
    ar
       & auto_name(angle)
       & auto_name(noise)
       & auto_name(rho)
       & auto_name(Gamma)
       & auto_name(xi)
       & auto_name(zeta)
       & auto_name(tau)
       & auto_name(friction)
       & auto_name(LL)
       & auto_name(CC)
       & auto_name(nsteps_Fl)
       & auto_name(nsteps_Q);
  }

  /** Serialization of the current frame (time snapshot) */
  template<class Archive>
  void serialize_frame(Archive& ar)
  {
    ar & auto_name(ux)
       & auto_name(uy)
       & auto_name(n)
       & auto_name(QQxx)
       & auto_name(QQyx)
       & auto_name(FFx)
       & auto_name(FFy)
       & auto_name(advX)
       & auto_name(advY) ;
  }
};

#endif//MODELS_LYOTROPIC_HPP_
