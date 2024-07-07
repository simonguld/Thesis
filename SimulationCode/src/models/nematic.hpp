#ifndef MODELS_NEMATIC_HPP_
#define MODELS_NEMATIC_HPP_

#include "models.hpp"

class Nematic : public Model
{
protected:
  /** Lattice Boltzmann distribution
   *
   * Written as ff[k][v], where k is node and v is the direction.
   * */
  LBField ff, fn, ff_tmp, fn_tmp;
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
  /** Stress tensor */
  ScalarField sigmaXX, sigmaYY, sigmaYX, sigmaXY;

  ScalarField FFx, FFy; //needed to correct LB algorithm  

  ScalarField E_kin, PE_LC, dE_kin, dPE_LC; //to calculate the energies in the potential and kinetic modes of the system

  /** Initial angle and noise */
  double angle_deg, angle, noise;
  int n_preinit = 1000; bool preinit_flag = false;
  /** Fluid density */
  double rho = 40.;
  /** Fluid parameters */
  double Gamma, xi, tau, friction, LL, CC, zeta;  

  double Q_kBT = 0, u_kBT = 0;  
  bool Q_fluct = false, u_fluct = false, isComp = false;
  
  bool backflow_on = true;
  bool isGuo = true; //we do Guo and Shan-Chen forcing schemes only

  bool outputEnergyCalc = true;

  std::string init_config;  
  unsigned npc = 1;
  
  double ftot = 0;
  

  /** Update fields using predictor-corrector method
   *
   * Because of the way the predictor-corrector is implemented this function
   * can be called many times in a row in order to import the numerical
   * stability of the algorithm. Only the first call needs to have the parameter
   * set to true.
   * */
  virtual void UpdateNematicFields(bool);
  virtual void UpdateNematicQuantities();
  virtual void UpdateFluidFields(bool);
  virtual void UpdateFluidQuantities();
  
  void UpdateNematicFieldsAtNode(unsigned, bool);
  void UpdateNematicQuantitiesAtNode(unsigned);  
  void UpdateFluidFieldsAtNode(unsigned, bool);
  void UpdateFluidQuantitiesAtNode(unsigned);  

  virtual void BoundaryConditionsLB();
  virtual void BoundaryConditionsFields();
  virtual void BoundaryConditionsFields2();

  void Move();

  //other random functions
  void InsertPlusDefect(unsigned, unsigned, int, double);
  void InsertMinusDefect(unsigned, unsigned, int, double);
  void InsertMinus1Defect(unsigned, unsigned, int, double);
  void InsertPlus1Defect(unsigned, unsigned, int, double);

public:
  Nematic(unsigned, unsigned, unsigned);

  /** Configure a single node
   *
   * This allows to change the way the arrays are configured in derived
   * classes, see for example NematicFreeBoundary.
   * */
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
       & auto_name(backflow_on);
  }

  /** Serialization of the current frame (time snapshot) */
  template<class Archive>
  void serialize_frame(Archive& ar)
  {
    ar & auto_name(ff)
       & auto_name(QQxx)
       & auto_name(QQyx)
       & auto_name(FFx)
       & auto_name(FFy)
       & auto_name(dE_kin)
       & auto_name(dPE_LC);
  }
};

#endif//MODELS_LYOTROPIC_HPP_
