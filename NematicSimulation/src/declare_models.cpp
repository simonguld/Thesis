// ======================================================================
// Model declaration (gets compiled in models.cpp)
// model headers and declare_model must be consistent!


// model headers
#include "models/minimal.hpp"
#include "models/lyotropic.hpp"
#include "models/nematic.hpp"
#include "models/polarlyotropic.hpp"
#include "models/drypolarlyotropic.hpp"

void DeclareModels()
{
  declare_model<Minimal>(
     "minimal",
      "This is just an example model showing a minimal implementation. "
     "This does exactly nothing (and not particularly fast)."
    );

  declare_model<Lyotropic>(
      "lyotropic",
      "Biphasic, lyotropic, nematic model as presented as described in "
      "10.1103/PhysRevLett.113.248303. We refer the user to this reference "
      "for further information."
      );

  declare_model<Nematic>(
      "nematic",
      "Pure nematic model with LdG free energy."
      );  

  declare_model<PolarLyotropic>(
      "polarlyotropic",
      "Polar model combined with the two-phase lyotropic formulation similar to the lyotropic model"
      );  

    declare_model<DryPolarLyotropic>(
      "drypolarlyotropic",
      "Polar model combined with the two-phase lyotropic formulation similar to the lyotropic model"
      );
  
}
