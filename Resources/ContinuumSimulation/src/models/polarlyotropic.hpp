#ifndef MODELS_POLARLYOTROPIC_HPP_
#define MODELS_POLARLYOTROPIC_HPP_

#include "models.hpp"

class PolarLyotropic : public Model
{
protected:
  /** Lattice Boltzmann distribution
   *
   * Written as ff[k][v], where k is node and v is the direction.
   * */
  LBField ff, fn, ff_tmp, fn_tmp;

  //Defining fields of the model
  ScalarField Px, PNx, Py, PNy;
  ScalarField phi, phn, phi_tmp;
  ScalarField ux, uy, ux_phi, uy_phi; 
  ScalarField n;

  //Derivatives, etc
  ScalarField Hx, Hy, MU;
  ScalarField FFx, FFy;
  ScalarField dxPx, dyPx, dxPy, dyPy;
  ScalarField sigmaXX, sigmaYY, sigmaYX, sigmaXY;

  /** Model parameters */
  double rho = 40.;
  double gamma, xi, tauPol, tauIso, friction, Kn, Kp, CC, zeta, Vp, alpha, beta;
  double GammaPhi, Kphi, Aphi, Vphi, Kw;

  //Initial Configuration options 
  double level, conc=1.0, angle_deg, angle, noise, radius, init_order;
  std::string init_config;

  
  /** Settings and checks */  
  unsigned npc = 1; 
  int n_preinit = 1000; bool preinit_flag = false;  
  bool backflow_on = true; bool isGuo = true; 
  bool conserve_phi = false; bool surftension_on = true; bool fixedphi = false;

  double ftot = 0; double ptot = 0;  
  double totalphi=0., countphi=0.;




  /** Update fields using predictor-corrector method
   *
   * Because of the way the predictor-corrector is implemented this function
   * can be called many times in a row in order to import the numerical
   * stability of the algorithm. Only the first call needs to have the parameter
   * set to true.
   * */
  virtual void UpdatePolarFields(bool);
  virtual void UpdateFluidFields(bool);
  /** Compute chemical potential, stress and derivatives */
  virtual void UpdatePolarQuantities();
  virtual void UpdateFluidQuantities();
  /** UpdateFields() implementation */
  void UpdatePolarFieldsAtNode(unsigned, bool);
  void UpdateFluidFieldsAtNode(unsigned, bool);
  /** UpdateQuantities() implementation */
  void UpdatePolarQuantitiesAtNode(unsigned);
  void UpdateFluidQuantitiesAtNode(unsigned);

  /** Boundary Conditions for the flow */
  virtual void BoundaryConditionsLB();
  /** Boundary Conditions for the fields */
  virtual void BoundaryConditionsFields();
  /** Boundary Conditions for the secondary fields */
  virtual void BoundaryConditionsFields2();
  /** Move the LB particles */
  void Move();

public:
  PolarLyotropic() = default;
  PolarLyotropic(unsigned, unsigned, unsigned);
  PolarLyotropic(unsigned, unsigned, unsigned, GridType);

  /** Configure a single node
   *
   * This allows to change the way the arrays are configured in derived
   * classes, see for example LyotropicFreeBoundary.
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
    ar & auto_name(level)
       & auto_name(conc)
       & auto_name(angle)
       & auto_name(noise)
       & auto_name(totalphi)
       & auto_name(rho)
       & auto_name(GammaPhi)
       & auto_name(gamma)
       & auto_name(xi)
       & auto_name(beta)
       & auto_name(zeta)
       & auto_name(Vphi)
       & auto_name(Vp)
       & auto_name(alpha)
       & auto_name(tauPol)
       & auto_name(tauIso)
       & auto_name(friction)
       & auto_name(Kn)
       & auto_name(Kp)
       & auto_name(Kphi)
        & auto_name(Kw)
       & auto_name(init_config)
       & auto_name(Aphi)
       & auto_name(CC);
  }

  /** Serialization of the current frame (time snapshot) */
  template<class Archive>
  void serialize_frame(Archive& ar)
  {
    ar & auto_name(ff)
       & auto_name(Px)
       & auto_name(Py)
       & auto_name(phi);
  }
};

#endif//MODELS_LYOTROPIC_HPP_
