#ifndef MODEL_HPP_
#define MODEL_HPP_

#include "fields.hpp"
#include "serialization.hpp"
#include "lb.hpp"

/** Shortcut, because we are no monsters */
using option_list = std::vector<boost::program_options::options_description>;

/** Base class for models.
  *
  * Should take care of:
  * * System size and dimension
  * * Parallelization (MPI)
  *
  * Order:
  * * InitializeBase()
  * * Initialize()
  * * Configure()
  * * Step()
  *
  * The boundary conditions is given from the command line as an unsigned. The
  * behaviour of the program is then: i) bc=0 always indicates periodic
  * boundary conditions and no boundary layer is created, this can be handled
  * independently from the model, (ii) any number bc>0 indicates that we fall
  * back to user defined boundary conditions that must be defined in the
  * appropriate model file through the virtual function BoundaryConditions().
  */
class Model : public Grid
{
protected:
  /** Boundary conditions */
  unsigned BC;

public:
  /** Default constructor
   *
   * Please call SetSize() to allocate memory and set BC*/
  Model() = default;
  /** Construct from dimension
    *
    * In general it is good practice to postpone all computations/allocations to
    * the Initialize() or InitializeBase() functions because dummy objects must
    * be created at program init.
    * */
  Model(unsigned, unsigned, unsigned, GridType);
  /** Virtual destructor
    *
    * The destructor is trivial here but is virtual in case a daughter class
    * needs it.
    * */
  virtual ~Model() {}
  /** Daughter classes specific intiatialization
    *
    * This function is called after creation and after option parsing. Typically
    * it will compute derived qties from the user input. Can throw in case some
    * parameters are not in their valid range.
    * */
  virtual void Initialize() = 0;
  /** Setup intial configuration
    *
    * This function is called after Initialize() and should setup the fields in
    * their initial configuration. Can throw in case some parameter is in an
    * invalid range.
    * */
  virtual void Configure() = 0;
  /** Get model specific parameters for option parsing
    *
    * This function should return a list of options that are specific to the
    * model, see minimal model for an example.
    * */
  virtual option_list GetOptions() = 0;
  /** Pre-run function
   *
   * This function is called just before the normal run and after PreRunStats().
   * It typically allows relaxation of the system in order to avoid numerical
   * difficulties related to the first time steps. Should not throw.
   * */
  virtual void Pre() {}
  /** Time step
   *
   * This is the time-stepping function and performs the main computation.
   * Should not throw.
   * */
  virtual void Step() = 0;
  /** Post-run function
   *
   * This function is called at the end of the normal run and allows to do some
   * numerical post processing. Should not throw.
   * */
  virtual void Post() {};
  /** Runtime checks
   *
   * This function is called periodically at every ninfo step and allows for
   * checks to be performed. Any failed check should throw an exception of the
   * type warning_msg or error_msg. An error will always interupt the program
   * while a warning will do so only if stop_at_warning=1. If you want to print
   * running information please use the function RuntimeStats() instead.
   * */
  virtual void RuntimeChecks() {}
  /** Print runtime statistics
   *
   * This is function is called periodically at every ninfo step and allows to
   * print some running information. This function can throw as RuntimeChecks
   * but is only called for verbose>2.
   * */
  virtual void RuntimeStats() {}
  /** Pre-run statistics
   *
   * This function is called before the main algorithm and allows one to perform
   * and print statistics. It should not be used to detect errors, for this
   * purpose please use the function Pre().
   * */
  virtual void PreRunStats() {}
};

// =============================================================================
// Models managment

/** Small ptr helper class for models
  *
  * We have to define this class because the serialize_params and
  * serialization_frame functions in daughter classes of Model are template and
  * hence can not be made virtual. We have to store the type information where
  * it is avaiable, i.e. at the model declaration. Using this trick it is
  * possible to call those functions on the derived classes through this object.
  * */
struct ModelPtr
{
  struct SerializerPtr
  {
    /** Ptr to the actual model */
    Model* ptr;
    /** Construct from existing pointer */
    SerializerPtr(Model* ptr_)
      : ptr(ptr_)
    {}
    /** Destructor frees memory */
    //~SerializerPtr() { delete ptr; }

    // forwarding to the serialization functions
    virtual void serialize_params(oarchive&) = 0;
    virtual void serialize_params(iarchive&) = 0;
    virtual void serialize_frame(oarchive&) = 0;
    virtual void serialize_frame(iarchive&) = 0;
  };

  template<class Derived>
  struct SerializerPtr_tmpl : public SerializerPtr
  {
    SerializerPtr_tmpl(Derived* ptr_)
      : SerializerPtr(ptr_)
    {}

    virtual void serialize_params(oarchive& ar)
    { dynamic_cast<Derived*>(ptr)->serialize_params(ar); }
    virtual void serialize_params(iarchive& ar)
    { dynamic_cast<Derived*>(ptr)->serialize_params(ar); }
    virtual void serialize_frame(oarchive& ar)
    { dynamic_cast<Derived*>(ptr)->serialize_frame(ar); }
    virtual void serialize_frame(iarchive& ar)
    { dynamic_cast<Derived*>(ptr)->serialize_frame(ar); }
  };

  SerializerPtr* ptr;

  /** Trivial constructor */
  ModelPtr()
    : ptr(nullptr)
  {}
  /** Template constructor from ptr */
  template<class Derived>
  ModelPtr(Derived* model_ptr)
  {
    ptr = new SerializerPtr_tmpl<Derived>(model_ptr);
  }
  /** Destructor frees memory */
  //~ModelPtr() { if(ptr!=nullptr) delete ptr; }

  /** Dereference operator access directly the object stored in SerlializerPtr
   * */
  Model* operator->() { return ptr->ptr; }
  /** Dereference operator access directly the object stored in SerlializerPtr
   * (const) */
  const Model* operator->() const { return ptr->ptr; }
};

/** Type of factory functions */
using Factory = std::function<ModelPtr(unsigned, unsigned, unsigned)>;

/** Add model (use declare_model() instead) */
void _add_model(std::string, std::string, Factory);

/** Declare model */
template<class T>
void declare_model(std::string name, std::string desc)
{
  // add model using simple lambda factory
  _add_model(name,
      desc,
      [](unsigned LX, unsigned LY, unsigned BC)
      { return ModelPtr(new T(LX, LY, BC)); }
      );
}

/** Return new model object from name string */
ModelPtr NewModel(std::string, unsigned, unsigned, unsigned);

/** Print models names and descriptions */
void ListModels();

/** Declares all the models of the program */
void DeclareModels();

#endif//MODEL_HPP_
