#include "models.hpp"

class Minimal : public Model
{
  // add your private variables here
  // ...

public:
  // the constructor needs to take the size of the grid as arguments
  Minimal(unsigned, unsigned, unsigned);

  // the following functions are pure virtual in Model and need to be overwritten (see minimal.cpp)
  virtual void Initialize();
  virtual void Configure();
  virtual void Step();
  virtual option_list GetOptions();

  /** Serialization of parameters */
  template<class Archive>
  void serialize_params(Archive&)
  {
    // add serialization of model parameters here
  }

  /** Serialization of the current frame */
  template<class Archive>
  void serialize_frame(Archive&)
  {
    // add serialization of frame (fields etc) here
  }
};
