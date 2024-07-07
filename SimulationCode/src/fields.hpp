#ifndef FIELDS_HPP_
#define FIELDS_HPP_

#include "geometry.hpp"

/** Field
 *
 * A field is a grid with a value of arbitrary type at each node. Note that no
 * range check is performed.
 * */
template<class NodeType>
class Field : public Grid
{
protected:
  /** The underlying type of the field */
  using FieldType = std::vector<NodeType>;
  /** The actual field (i.e. one value at each node) */
  FieldType field;

public:
  /** Construct a field of zero size */
  Field() = default;
  /** Construct with given size */
  Field(unsigned LX, unsigned LY, GridType Type)
    : Grid(LX, LY, Type)
  { field.resize(TotalSize); }

  /** Resize field */
  virtual void SetSize(unsigned LX_, unsigned LY_, GridType Type)
  {
    Grid::SetSize(LX_, LY_, Type);
    field.resize(TotalSize);
  }

  /** Universal access operator
   *
   * This operator allows to access all components of the field (domain, walls,
   * and corners) using unsigned integers indices. In order to obtain the
   * correct index, the function GetIndex() (and related functions) should be
   * used. To access specific components, the function Get() should be used
   * instead.
   * */
  inline NodeType& operator[](unsigned n)
  { return field[n]; }
  /** Universal access operator (const) */
  inline const NodeType& operator[](unsigned n) const
  { return field[n]; }
  /** Access specific component */
  template<class T> inline NodeType& Get(T d, unsigned n=0)
  { return field[GetIndex(d, n)]; }
  /** Access specific component (const) */
  template<class T> inline const NodeType& Get(T d, unsigned n=0) const
  { return field[GetIndex(d, n)]; }

  /** Return const ref on underlying field object */
  inline FieldType& get_data()
  { return field; }
  /** Return const ref on underlying field object */
  inline const FieldType& get_data() const
  { return field; }

  /** Underlying type */
  using value_type = NodeType;

  /** Iterator
   *
   * This is a simple iterator implementation that holds an index together with
   * a reference to the underlying field object. All the operators are defined
   * on the index type, in particular we do not check if the iterators are
   * pointing to the same object.
   * */
  class DomainIterator
  {
  protected:
    /** Reference to the underlying field */
    Field& field;
    /** Position of the iterator */
    unsigned pos;

  public:
    /** Construct from underlying field and position */
    DomainIterator(Field& field, unsigned pos = 0)
      : field(field), pos(pos)
    {}
    /** Copy constructor */
    DomainIterator(const DomainIterator&) = default;

    inline DomainIterator& operator++()
    { ++pos; return *this; }
    inline DomainIterator operator++(int)
    { DomainIterator i { *this }; ++pos; return i; }
    inline DomainIterator& operator--()
    { --pos; return *this; }
    inline DomainIterator operator--(int)
    { DomainIterator i { *this }; --pos; return i; }
    inline NodeType& operator*()
    { return field[pos]; }
    inline const NodeType& operator*() const
    { return field[pos]; }
    inline bool operator==(const DomainIterator& i) const
    { return pos==i.pos; }
    inline bool operator!=(const DomainIterator& i) const
    { return pos!=i.pos; }
    inline bool operator<(const DomainIterator& i) const
    { return pos<i.pos; }
    inline bool operator>(const DomainIterator& i) const
    { return pos>i.pos; }
    inline bool operator<=(const DomainIterator& i) const
    { return pos<=i.pos; }
    inline bool operator>=(const DomainIterator& i) const
    { return pos>=i.pos; }
  };

  /** Const iterator
   *
   * This is the const version of DomainIterator().
   * */
  class ConstDomainIterator
  {
  protected:
    /** Const reference to the underlying field */
    const Field& field;
    /** Position of the iterator */
    unsigned pos;

  public:
    /** Construct from underlying field and position */
    ConstDomainIterator(const Field& field, unsigned pos = 0)
      : field(field), pos(pos)
    {}
    /** Copy constructor */
    ConstDomainIterator(const ConstDomainIterator&) = default;

    inline ConstDomainIterator& operator++()
    { ++pos; return *this; }
    inline ConstDomainIterator operator++(int)
    { DomainIterator i { *this }; ++pos; return i; }
    inline ConstDomainIterator& operator--()
    { --pos; return *this; }
    inline ConstDomainIterator operator--(int)
    { DomainIterator i { *this }; --pos; return i; }
    inline const NodeType& operator*() const
    { return field[pos]; }
    inline bool operator==(const ConstDomainIterator& i) const
    { return pos==i.pos; }
    inline bool operator!=(const ConstDomainIterator& i) const
    { return pos!=i.pos; }
    inline bool operator<(const ConstDomainIterator& i) const
    { return pos<i.pos; }
    inline bool operator>(const ConstDomainIterator& i) const
    { return pos>i.pos; }
    inline bool operator<=(const ConstDomainIterator& i) const
    { return pos<=i.pos; }
    inline bool operator>=(const ConstDomainIterator& i) const
    { return pos<=i.pos; }
  };

  DomainIterator begin()
  { return DomainIterator(*this, 0); }
  DomainIterator end()
  { return DomainIterator(*this, LX*LY); }
  ConstDomainIterator begin() const
  { return ConstDomainIterator(*this, 0); }
  ConstDomainIterator end() const
  { return ConstDomainIterator(*this, LX*LY); }

  //template<typename NodeType>
  //friend void swap(Field<NodeType>&, Field<NodeType>&);

  // ===========================================================================
  // Periodic boundary conditions

  /** Apply PBC on a given wall */
  inline void ApplyPBC(Wall w);

  /** Apply PBC on a given corner surrounded by two PBC walls */
  inline void ApplyPBC(Corner c);

  /** Apply PBC on a given corner surrounded by a single PBC wall
   *
   * The wall that is given as a parameter is the PBC wall.
   * */
  inline void ApplyPBC(Corner c, Wall w);

  /** Apply PBC on a wall and its opposite */
  inline void ApplyPBC(PBCWall w);

  /** Apply a constant offset PBC to a wall (for Scalar Field only) */
  inline void ApplyConstOffsetPBC(Wall w, double value);

  
  // ===========================================================================
  // Free slip (LBField only)

  /** Apply free-slip on a given wall (LBField only)
   *
   * This function applies free-slip boundary conditions on a given wall for
   * an unadvected LB field. If there is an adjacent PBC wall, then the PBC must
   * be enforced first. See Succi's book for details.
   * */
  inline void ApplyFreeSlip(Wall w);

  /** Apply free-slip on a given corner with a given free-slip wall (LBField
   * only)
   *
   * This applies free-slip boundary condition on a given corner adjacent to a
   * given free-slip wall. This function should NOT be called twice if both
   * walls are free-slip.
   * */
  inline void ApplyFreeSlip(Corner c, Wall w);

  /** Apply free-slip on a given corner when both surrounding walls are free-
   * slip
   *
   * N.b.: Same as for no-slip corner. */
  inline void ApplyFreeSlip(Corner c);

  // ===========================================================================
  // No slip (LBField only)

  /** Apply no-slip on a given wall (LBField only)
   *
   * This function applies no-slip boundary conditions on a given wall for
   * an unadvected LB field. See Succi's book for details.
   * */
  inline void ApplyNoSlip(Wall w);

  /** Apply no-slip on a given corner
   *
   * Because the no-slip condition is a simple bounce back, the same function
   * can be used if one or two walls are no-slip. It is also equivalent to the
   * case where there are two free slip walls. */
  inline void ApplyNoSlip(Corner c);

  // ===========================================================================
  // Neumann (ScalarField only)

  /** Apply zero-derivative boundary condition on a given wall (ScalarField
   * only)
   *
   * This corresponds to a Neumann boundary condition with zero derivative and
   * cannot be implemented independently of the stencil. Hence, this definition
   * relies on the peculiar form of derivX and derivY.
   * */
  inline void ApplyNeumann(Wall w);

  /** Apply zero-derivative boundary condition on a given corner with a given
   * impenetrable wall and a PBC wall (ScalarField only)
   *
   * See ApplyNeumann().
   * */
  inline void ApplyNeumann(Corner c, Wall w);

  /** Apply zero-derivative bdry conditions at a corner with two impenetrable
   * walls */
  inline void ApplyNeumann(Corner c);

  // ===========================================================================
  // Dirichlet (ScalarField only)

  /** Apply Dirichlet boundary condition on a given wall (ScalarField
   * only)
   *
   * This corresponds to setting the value of the field at the boundary along
   * the wall.
   * */
  inline void ApplyDirichlet(Wall w, double value);

  /** Apply Dirichlet boundary condition on a given corner with a given
   * impenetrable wall and a PBC wall (ScalarField only)
   *
   * Must be called after the PBC for the wall. See ApplyDirichlet().
   * */
  inline void ApplyDirichlet(Corner c, Wall w, double value);

  /** Apply Dirichlet boundary condition on a given corner surrounded by two
   * walls with (the same) Dirichlet bdry condition
   *
   * See ApplyDirichlet().
   * */
  inline void ApplyDirichlet(Corner c, double value);

  // ===========================================================================
  // Fixed Boundary Value (ScalarField only)

  /** Apply a given boundary value on the boundary nodes
   * Differs from Dirichlet boundary condition, because there is no constraint on the edge node of the lattice
   *
   * See SetBoundaryValue().
   * */
  inline void SetBoundaryValue(Corner c, double value);
  
  inline void SetBoundaryValue(Wall w, double value);


  // ===========================================================================
  // Derivative (ScalarField only)
  //
  // Derivative bdry conditions simply extrapolates the field such as
  // reproducing the given derivative using the usual stencil.

  /** Apply 'Derivative' boundary condition on a given wall (ScalarField
   * only) */
  inline void CopyDerivative(Wall w);

  /** Apply 'Derivative' boundary condition on a given corner with a given
   * impenetrable wall and a PBC wall (ScalarField only) */
  inline void CopyDerivative(Corner c, Wall w);

  /** Apply 'Derivative' boundary condition on a given corner with both walls
   * impenetrable */
  inline void CopyDerivative(Corner c);

  // ===========================================================================
  // Box

  /** Apply PBC on all walls and corners */
  inline void ApplyPBC();

  /** Apply Neumann on all walls (ScalarField only) */
  inline void ApplyNeumann();

  /** Apply Dirichlet (with the same value) on all walls (ScalarField only) */
  inline void ApplyDirichlet(double);

  /** Apply SetBoundaryValue (with the same value) on all walls (ScalarField only) */
  inline void SetBoundaryValue(double);

  /** Apply free slip on all walls (LBField only) */
  inline void ApplyFreeSlip();

  /** Apply no slip on all walls (LBField only) */
  inline void ApplyNoSlip();

  /** Apply 'Derivative' on all walls (ScalarField only) */
  inline void CopyDerivative();

  // ===========================================================================
  // Channel (pbc on left and right walls)

  /** Apply Neumann on front and back walls, while imposing PBC on left and
   * right (ScalarField only) */
  inline void ApplyNeumannChannel();

  /** Apply Dirichlet (with the same value) on front and back walls, while
   * imposing PBC on left and right (ScalarField only) */
  inline void ApplyDirichletChannel(double);

   /** Apply SetBoundaryValue (with the same value) on front and back walls, while
   * imposing PBC on left and right (ScalarField only) */
  inline void SetBoundaryValueChannel(double);

  /** Apply free slip on front and back walls, while imposing PBC on left and
   * right (LBField only) */
  inline void ApplyFreeSlipChannel();

  /** Apply no slip on front and back walls, while imposing PBC on left and
   * right (LBField only) */
  inline void ApplyNoSlipChannel();

  /** Apply 'Derivative' on front and back walls, while imposing PBC on left and
   * right (ScalarField only) */
  inline void CopyDerivativeChannel();

  // ===========================================================================
  // Free boundary conditions

  /** Apply free-slip boundary condition on an arbitrary wall (LBField only)
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyFreeSlipFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w);

  /** Apply no-slip boundary condition on an arbitrary wall (LBField only)
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyNoSlipFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w);

  /** Apply an outlet boundary condition on an arbitrary wall (LBField only)
   * See Succi chapter 6.3
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyOutletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double uin);

  /** Apply an inlet boundary condition on an arbitrary wall (LBField only) for a
   * homogeneous flow-profile
   * See Succi chapter 6.3
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyInletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double uin_x, double uin_y, Field<double>& nn);

  /** The corresponding stress boundary condition for an inlet boundary condition on
   * an arbitrary wall. Calculates the value to be consistent with the gradients of
   * phi and Q.
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyStressNeumannInletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, Field<double>& QQxx, Field<double>& QQyx,
      Field<double>& phi, double AA, double CC, double KK, double LL, double zeta,
      double xi, Grid::TensorComponent component);

  /** Apply the outlet boundary condition on an arbitrary wall (vy times phi only)
   * See Succi chapter 6.3
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyVyPhiOutletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double uin, Field< std::array<double, 9> >& ffield, Field<double>& phifield);

  /** Apply the outlet boundary condition on an arbitrary wall (vx times phi only)
   * See Succi chapter 6.3
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyVxPhiOutletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double uin, Field< std::array<double, 9> >& ffield, Field<double>& phifield);

  /** Apply the outlet boundary condition on an arbitrary wall (vy only)
   * See Succi chapter 6.3
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyVyOutletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double uin, Field< std::array<double, 9> >& ffield);

  /** Apply the outlet boundary condition on an arbitrary wall (vx only)
   * See Succi chapter 6.3
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */
  inline void ApplyVxOutletFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double uin, Field< std::array<double, 9> >& ffield);

  /** Apply no-slip boundary condition on an arbitrary wall that moves(LBField only)
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * The wall moves with velocity wallspeed along itself.
   * NEED TO DOUBLECHECK. Looks like wall moves only with about half 'wallspeed'.
   * Compare description in Succi and implementation in 3D code.
   * */
  inline void ApplyNoSlipMovingFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w, double wallspeed);

  /** Apply a constant-pressure boundary condition on an arbitrary wall (LBField only)
   * For a field the vorseponding condition would be a CopyDerivative one
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyFreeSlip().
   * */

inline void ApplyPressureOutletFreeWall(unsigned start_x, unsigned start_y,
    unsigned length, Wall w, Field<double>& nn, Field<double>& ux, Field<double>& uy);

  /** Apply free-slip on a given corner at position (corner_x,corner_y) with
   *  two free-slip walls. (LBField only)
   *
   * The ''concave'' means that the corner is such that the 90-degree angle
   * encloses the inside region, i.e. only one node neigbouring the corner is
   * inside.
   * */
  inline void ApplyFreeSlipFreeConcaveCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply free-slip on a given corner at position (corner_x,corner_y) with
   *  two free-slip walls. (LBField only)
   *
   * The ''convex'' means that the corner is such that the 90-degree angle
   * encloses the outside region, i.e. three nodes neigbouring the corner is inside
   * */
  inline void ApplyFreeSlipFreeConvexCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply no-slip on a given corner at position (corner_x,corner_y) with
   *  two no-slip walls. (LBField only)
   *
   * The ''concave'' means that the corner is such that the 90-degree angle
   * encloses the inside region, i.e. only one node neigbouring the corner is
   * inside.
   * */
  inline void ApplyNoSlipFreeConcaveCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply no-slip on a given corner at position (corner_x,corner_y) with
   *  two no-slip walls. (LBField only)
   *
   * The ''convex'' means that the corner is such that the 90-degree angle
   * encloses the outside region, i.e. three nodes neigbouring the corner is inside
   * */
  inline void ApplyNoSlipFreeConvexCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply zero-derivative boundary condition on an arbitrary wall
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyNeumann().
   * */
  inline void ApplyNeumannFreeWall(unsigned start_x,
      unsigned start_y, unsigned length, Wall w);

  /** For testing purposes
   *
   * Sums over normal derivative of the fields in the cells adjacent to the wall.
   * */
  inline double CheckNeumannFreeWall(unsigned start_x, unsigned start_y,
      unsigned length, Wall w);

  /** Apply zero-derivative boundary condition on a given corner at position
   * (corner_x,corner_y) between two walls w1, w2 (scalar field only)
   *
   * The ''concave'' means that the corner is such that the 90-degree angle
   * encloses the inside region, i.e. only one node neigbouring the corner is
   * inside.
   * */
  inline void ApplyNeumannFreeConcaveCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply zero-derivative boundary condition on a given corner at position
   * (corner_x,corner_y) between two walls w1, w2 (scalar field only)
   *
   * The ''convex'' means that the corner is such that the 90-degree angle
   * encloses the outside region, i.e. three nodes neigbouring the corner are
   * inside.
   * */
  inline void ApplyNeumannFreeConvexCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply Dirichlet boundary condition on an arbitrary wall
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyNeumann().
   * */
  inline void ApplyDirichletFreeWall(unsigned start_x,
      unsigned start_y, unsigned length, Wall w, double value);

   /** Apply Dirichlet boundary condition on a given corner at position
   * (corner_x,corner_y) between two walls w1, w2 (scalar field only)
   *
   * The ''concave'' means that the corner is such that the 90-degree angle
   * encloses the inside region, i.e. only one node neigbouring the corner is
   * inside.
   * */
  inline void ApplyDirichletFreeConcaveCorner(unsigned corner_x,
      unsigned corner_y, Corner c, double value);

  /** Apply Dirichlet boundary condition on a given corner at position
   * (corner_x,corner_y) between two walls w1, w2 (scalar field only)
   *
   * The ''convex'' means that the corner is such that the 90-degree angle
   * encloses the outside region, i.e. three nodes neigbouring the corner are
   * inside.
   * */
  inline void ApplyDirichletFreeConvexCorner(unsigned corner_x,
      unsigned corner_y, Corner c, double value);

  /** Apply 'Derivative' boundary condition on an arbitrary wall.
   * It simply extrapolates the field such that in the end the
   * derivative corresponds to the one-sided derivative.
   *
   * The wall starts at (start_x,start_y) and ends at (start_x,start_y)
   * + length * 'unit vector in direction parallel to wall'. See ApplyNeumann().
   * */
  inline void CopyDerivativeFreeWall(unsigned start_x,
      unsigned start_y, unsigned length, Wall w);

   /** Apply 'Derivative' boundary condition on a given corner at position
   * (corner_x,corner_y) between two walls w1, w2 (scalar field only)
   *
   * The ''concave'' means that the corner is such that the 90-degree angle
   * encloses the inside region, i.e. only one node neigbouring the corner is
   * inside.
   * */
  inline void CopyDerivativeFreeConcaveCorner(unsigned corner_x,
      unsigned corner_y, Corner c);

  /** Apply 'Derivative' boundary condition on a given corner at position
   * (corner_x,corner_y) between two walls w1, w2 (scalar field only)
   *
   * The ''convex'' means that the corner is such that the 90-degree angle
   * encloses the outside region, i.e. three nodes neigbouring the corner are
   * inside.
   * */
  inline void CopyDerivativeFreeConvexCorner(unsigned corner_x,
      unsigned corner_y, Corner c);
};

/** Scalar (i.e. single double) at each node, with boudnary layer */
using ScalarField     = Field<double>;
/** Type of a LB node (all directions) */ 
using LBNode = std::array<double, 9>;
/** Lattice-Boltzmann array at each node, with boundary layer */
using LBField     = Field<LBNode>;

/** LB equilibrium distribution at a node */
inline LBNode GetEqDistribution(double vx, double vy, double nn)
{
  LBNode fe;

  const double v2 = vx*vx + vy*vy;
  fe[1] = nn*(1./9. + 1./3.*vx + 1./3.*v2 - 1./2.*vy*vy);
  fe[2] = nn*(1./9. - 1./3.*vx + 1./3.*v2 - 1./2.*vy*vy);
  fe[3] = nn*(1./9. + 1./3.*vy - 1./2.*vx*vx + 1./3.*v2);
  fe[4] = nn*(1./9. - 1./3.*vy - 1./2.*vx*vx + 1./3.*v2);
  fe[5] = nn*(1./36. + 1./12.*vx + 1./12.*vy + 1./12.*v2 + 1./4.*vx*vy);
  fe[6] = nn*(1./36. - 1./12.*vx - 1./12.*vy + 1./12.*v2 + 1./4.*vx*vy);
  fe[7] = nn*(1./36. - 1./12.*vx + 1./12.*vy + 1./12.*v2 - 1./4.*vx*vy);
  fe[8] = nn*(1./36. + 1./12.*vx - 1./12.*vy + 1./12.*v2 - 1./4.*vx*vy);
  fe[0] = nn-fe[1]-fe[2]-fe[3]-fe[4]-fe[5]-fe[6]-fe[7]-fe[8];

  return fe;
}

// =============================================================================
// Implementation

/** Swap function for fields
 *
 * We need to overwrite the swap function for field objects because the way grid
 * manages the neighbour list. */
//template<typename NodeType>
//void swap(Field<NodeType>& f1, Field<NodeType>& f2)
//{
//  swap(f1.field, f2.field);
//}

template<>
inline void Field<double>::ApplyPBC(Wall w)
{
  for(unsigned k=0; k<GetSize(w); ++k)
    field[GetIndex(w, k)] = field[next(GetIndex(opp(w), k), w)];
}

template<>
inline void Field<double>::ApplyPBC(Corner c)
{
  field[GetIndex(c)] = field[next(GetIndex(opp(c)), c)];
}

template<>
inline void Field<double>::ApplyPBC(Corner c, Wall w)
{
  // this is the other corner along the non-PBC wall
  const auto o = GetIndex(refl(c, w));
  field[GetIndex(c)] = field[next(o, opp(c))];
}

template<>
inline void Field<double>::ApplyPBC(PBCWall w)
{
  ApplyPBC(static_cast<Wall>(w));
  ApplyPBC(opp(static_cast<Wall>(w)));
}

template<>
inline void Field<LBNode>::ApplyPBC(Wall w)
{
  for(unsigned k=0; k<GetSize(w); ++k)
  {    
    for(unsigned v=0; v<lbq; ++v)     
      field[GetIndex(w, k)][v] = field[next(GetIndex(opp(w), k), w)][v];

  }
  
}

template<>
inline void Field<LBNode>::ApplyPBC(Corner c)
{
  for(unsigned v=0; v<lbq; ++v)
    field[GetIndex(c)][v] = field[next(GetIndex(opp(c)), c)][v];
}

template<>
inline void Field<LBNode>::ApplyPBC(Corner c, Wall w)
{
  // this is the other corner along the non-PBC wall
  const auto o = GetIndex(refl(c, w));
  for(unsigned v=0; v<lbq; ++v)
    field[GetIndex(c)][v] = field[next(o, opp(c))][v];
}

template<>
inline void Field<LBNode>::ApplyPBC(PBCWall w)
{  
  ApplyPBC(static_cast<Wall>(w));
  ApplyPBC(opp(static_cast<Wall>(w)));
}

template<>
inline void Field<double>::ApplyConstOffsetPBC(Wall w, double value)
{
  int sign = (opp(w)>w)? 1:-1;
  for(unsigned k=0; k<GetSize(w); ++k){  
    field[GetIndex(w, k)] = field[next(GetIndex(opp(w), k), w)] - sign*value/2.0;
  }
}


template<>
inline void Field<LBNode>::ApplyFreeSlip(Wall w)
{
  for(unsigned i=0; i<GetSize(w); ++i)
  {
    const auto k = GetIndex(w, i);

    for(unsigned v=0; v<lbq; ++v)
      if(is_incoming(v, w))
      {
        const auto u = refl(v, w);
        field[k][v] = field[next(k, opp(w))][u];
      }
  }
}

template<>
inline void Field<LBNode>::ApplyNoSlip(Wall w)
{
  for(unsigned i=0; i<GetSize(w); ++i)
  {
    const auto k = GetIndex(w, i);

    for(unsigned v=0; v<lbq; ++v)
      if(is_incoming(v, w))
        field[k][v] = field[next(k, v)][opp(v)];
  }
}

template<>
inline void Field<LBNode>::ApplyFreeSlip(Corner c, Wall w)
{
  // the other wall
  const auto p = proj(c, w);
  // target
  const auto k = GetIndex(c);
  const auto v = opp(c);
  // source
  const auto l = next(GetIndex(refl(c, p)), p);

  field[k][v] = field[l][v];
}

template<>
inline void Field<LBNode>::ApplyNoSlip(Corner c)
{
  // simple bounce back
  field[GetIndex(c)][opp(c)] = field[next(GetIndex(c), opp(c))][c];
}

template<>
inline void Field<LBNode>::ApplyFreeSlip(Corner c)
{
  // simple bounce back
  field[GetIndex(c)][opp(c)] = field[next(GetIndex(c), opp(c))][c];
}

template<>
inline void Field<double>::ApplyNeumann(Wall w)
{
  for(unsigned k=0; k<GetSize(w); ++k)
  {
    const auto i = GetIndex(w, k);
    field[i] = field[next(i, opp(w))];
  }
}

template<>
inline void Field<double>::ApplyNeumann(Corner c, Wall w)
{
  const auto i = GetIndex(c);
  field[i] = field[next(i, opp(w))];
}

template<>
inline void Field<double>::ApplyNeumann(Corner c)
{
  const auto i = GetIndex(c);
  field[i] = field[next(i, opp(c))];
}

template<>
inline void Field<double>::ApplyDirichlet(Wall w, double value)
{
  for(unsigned k=0; k<GetSize(w); ++k)
  {
    const auto i = GetIndex(w, k);
    field[i] = 2*value - field[next(i, opp(w))];
  }
}

template<>
inline void Field<double>::ApplyDirichlet(Corner c, Wall w, double value)
{
  const auto i = GetIndex(c);
  field[i] = 2*value - field[next(i, opp(w))];
}

template<>
inline void Field<double>::ApplyDirichlet(Corner c, double value)
{
  const auto i = GetIndex(c);
  field[i] = 2*value - field[next(i, opp(c))];
}

template<>
inline void Field<double>::CopyDerivative(Wall w)
{
  for(unsigned k=0; k<GetSize(w); ++k)
  {
    const auto i = GetIndex(w, k);
    const auto j = next(i, opp(w));
    field[i] = 2*field[j]-field[next(j, opp(w))];
  }
}

template<>
inline void Field<double>::CopyDerivative(Corner c, Wall w)
{
  const auto i = GetIndex(c);
  const auto j = next(i, opp(w));
  field[i] = 2*field[j]-field[next(j, opp(w))];
}

template<>
inline void Field<double>::CopyDerivative(Corner c)
{
  const auto i = GetIndex(c);
  const auto j = next(i, opp(c));
  field[i] = 2*field[j]-field[next(j, opp(c))];
}

template<>
inline void Field<double>::SetBoundaryValue(Wall w, double value)
{
  for(unsigned k=0; k<GetSize(w); ++k)
  {
    const auto i = GetIndex(w, k);    
    field[i] = value;
  }
}

template<>
inline void Field<double>::SetBoundaryValue(Corner c, double value)
{
  const auto i = GetIndex(c);
  field[i] = value;
}


template<>
inline void Field<double>::ApplyNeumann()
{
  ApplyNeumann(Wall::Left);
  ApplyNeumann(Wall::Right);
  ApplyNeumann(Wall::Front);
  ApplyNeumann(Wall::Back);
  ApplyNeumann(Corner::LeftFront);
  ApplyNeumann(Corner::RightFront);
  ApplyNeumann(Corner::LeftBack);
  ApplyNeumann(Corner::RightBack);
}

template<>
inline void Field<double>::ApplyPBC()
{
  ApplyPBC(Wall::Left);
  ApplyPBC(Wall::Right);
  ApplyPBC(Wall::Front);
  ApplyPBC(Wall::Back);
  ApplyPBC(Corner::LeftFront);
  ApplyPBC(Corner::RightFront);
  ApplyPBC(Corner::LeftBack);
  ApplyPBC(Corner::RightBack);
}

/* //Old version, as is
template<class NodeType>
inline void Field<NodeType>::ApplyPBC()
{
  ApplyPBC(Wall::Left);
  ApplyPBC(Wall::Right);
  ApplyPBC(Wall::Front);
  ApplyPBC(Wall::Back);
  ApplyPBC(Corner::LeftFront);
  ApplyPBC(Corner::RightFront);
  ApplyPBC(Corner::LeftBack);
  ApplyPBC(Corner::RightBack);

  //Every corner and every wall
  ApplyPBC(Corner::RightBack, Wall::Right);
  ApplyPBC(Corner::RightBack, Wall::Back);
  ApplyPBC(Corner::LeftBack, Wall::Left);
  ApplyPBC(Corner::LeftBack, Wall::Back);
  ApplyPBC(Corner::RightFront, Wall::Right);
  ApplyPBC(Corner::RightFront, Wall::Front);
  ApplyPBC(Corner::LeftFront, Wall::Left);
  ApplyPBC(Corner::LeftFront, Wall::Front);

}
*/

template<>
inline void Field<LBNode>::ApplyPBC()
{
  ApplyPBC(Wall::Left);
  ApplyPBC(Wall::Right);
  ApplyPBC(Wall::Front);
  ApplyPBC(Wall::Back);
  ApplyPBC(Corner::LeftFront);
  ApplyPBC(Corner::RightFront);
  ApplyPBC(Corner::LeftBack);
  ApplyPBC(Corner::RightBack);

}

template<>
inline void Field<double>::ApplyDirichlet(double value)
{
  ApplyDirichlet(Wall::Left, value);
  ApplyDirichlet(Wall::Right, value);
  ApplyDirichlet(Wall::Front, value);
  ApplyDirichlet(Wall::Back, value);
  ApplyDirichlet(Corner::LeftFront, value);
  ApplyDirichlet(Corner::RightFront, value);
  ApplyDirichlet(Corner::LeftBack, value);
  ApplyDirichlet(Corner::RightBack, value);
}

template<>
inline void Field<double>::SetBoundaryValue(double value)
{
  SetBoundaryValue(Wall::Left, value);
  SetBoundaryValue(Wall::Right, value);
  SetBoundaryValue(Wall::Front, value);
  SetBoundaryValue(Wall::Back, value);
  SetBoundaryValue(Corner::LeftFront, value);
  SetBoundaryValue(Corner::RightFront, value);
  SetBoundaryValue(Corner::LeftBack, value);
  SetBoundaryValue(Corner::RightBack, value);
}


template<>
inline void Field<double>::CopyDerivative()
{
  CopyDerivative(Wall::Left);
  CopyDerivative(Wall::Right);
  CopyDerivative(Wall::Front);
  CopyDerivative(Wall::Back);
  CopyDerivative(Corner::LeftFront);
  CopyDerivative(Corner::RightFront);
  CopyDerivative(Corner::LeftBack);
  CopyDerivative(Corner::RightBack);
}

template<>
inline void Field<LBNode>::ApplyFreeSlip()
{
  ApplyFreeSlip(Wall::Left);
  ApplyFreeSlip(Wall::Right);
  ApplyFreeSlip(Wall::Front);
  ApplyFreeSlip(Wall::Back);
  ApplyFreeSlip(Corner::LeftFront);
  ApplyFreeSlip(Corner::RightFront);
  ApplyFreeSlip(Corner::LeftBack);
  ApplyFreeSlip(Corner::RightBack);
}

template<>
inline void Field<LBNode>::ApplyNoSlip()
{
  ApplyNoSlip(Wall::Left);
  ApplyNoSlip(Wall::Right);
  ApplyNoSlip(Wall::Front);
  ApplyNoSlip(Wall::Back);
  ApplyNoSlip(Corner::LeftFront);
  ApplyNoSlip(Corner::RightFront);
  ApplyNoSlip(Corner::LeftBack);
  ApplyNoSlip(Corner::RightBack);
}

template<>
inline void Field<double>::ApplyNeumannChannel()
{
  // pbc on the left and right walls
  ApplyPBC(PBCWall::LeftRight);
  // Neumann on the front and back walls
  ApplyNeumann(Wall::Front);
  ApplyNeumann(Wall::Back);
  // corners
  ApplyNeumann(Corner::RightBack, Wall::Back);
  ApplyNeumann(Corner::RightFront, Wall::Front);
  ApplyNeumann(Corner::LeftBack, Wall::Back);
  ApplyNeumann(Corner::LeftFront, Wall::Front);
}

template<>
inline void Field<double>::CopyDerivativeChannel()
{
  // pbc on the left and right walls
  ApplyPBC(PBCWall::LeftRight);
  // dirichlet on the front and back walls
  CopyDerivative(Wall::Front);
  CopyDerivative(Wall::Back);
  // corners
  CopyDerivative(Corner::RightBack, Wall::Back);
  CopyDerivative(Corner::RightFront, Wall::Front);
  CopyDerivative(Corner::LeftBack, Wall::Back);
  CopyDerivative(Corner::LeftFront, Wall::Front);
}

template<>
inline void Field<double>::ApplyDirichletChannel(double value)
{
  // pbc on the left and right walls
  ApplyPBC(PBCWall::LeftRight);
  // dirichlet on the front and back walls
  ApplyDirichlet(Wall::Front, value);
  ApplyDirichlet(Wall::Back, value);
  // corners
  ApplyDirichlet(Corner::RightBack, Wall::Back, value);
  ApplyDirichlet(Corner::RightFront, Wall::Front, value);
  ApplyDirichlet(Corner::LeftBack, Wall::Back, value);
  ApplyDirichlet(Corner::LeftFront, Wall::Front, value);
}

template<>
inline void Field<double>::SetBoundaryValueChannel(double value)
{
  // pbc on the left and right walls
  ApplyPBC(PBCWall::LeftRight);
  // dirichlet on the front and back walls
  SetBoundaryValue(Wall::Front, value);
  SetBoundaryValue(Wall::Back, value);
  // corners
  SetBoundaryValue(Corner::RightBack, value);
  SetBoundaryValue(Corner::RightFront, value);
  SetBoundaryValue(Corner::LeftBack, value);
  SetBoundaryValue(Corner::LeftFront, value);
}



template<>
inline void Field<LBNode>::ApplyFreeSlipChannel()
{
  // pbc on the left and right walls
  ApplyPBC(PBCWall::LeftRight);
  // Free-slip on the front and back walls
  ApplyFreeSlip(Wall::Front);
  ApplyFreeSlip(Wall::Back);
  // corners
  ApplyFreeSlip(Corner::RightBack, Wall::Back);
  ApplyFreeSlip(Corner::RightFront, Wall::Front);
  ApplyFreeSlip(Corner::LeftBack, Wall::Back);
  ApplyFreeSlip(Corner::LeftFront, Wall::Front);
}

template<>
inline void Field<LBNode>::ApplyNoSlipChannel()
{
  // pbc on the left and right walls
  ApplyPBC(PBCWall::LeftRight);
  // no-slip on the front and back walls
  ApplyNoSlip(Wall::Front);
  ApplyNoSlip(Wall::Back);
  // corners
  ApplyNoSlip(Corner::RightBack);
  ApplyNoSlip(Corner::RightFront);
  ApplyNoSlip(Corner::LeftBack);
  ApplyNoSlip(Corner::LeftFront);
}

template<>
inline void Field<LBNode>::ApplyFreeSlipFreeConcaveCorner(unsigned corner_x,
                                                          unsigned corner_y,
                                                          Corner c)
{
  //for safety reasons: kill contributions flowing away from the system
  const auto k = GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Left : Wall::Right);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Front : Wall::Back);

  for(unsigned v=0; v<lbq; ++v) field[k][v]=0;

  field[k][opp(c)] = field[next(k, opp(c))][c];
  field[next(k,opp(w1))][opp(c)] = field[next(next(k, opp(w2)), opp(c))][c];//correct also adjacent walls
  field[next(k,opp(w2))][opp(c)] = field[next(next(k, opp(w1)), opp(c))][c];//correct also adjacent walls
}

template<>
inline void Field<LBNode>::ApplyFreeSlipFreeConvexCorner(unsigned corner_x,
                                                         unsigned corner_y,
                                                         Corner c)
{
  const auto k = GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Right : Wall::Left);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Back : Wall::Front);

  //for safety reasons: kill contributions flowing away from the system
  field[k][opp(c)] = 0;
  field[k][w1] = 0;
  field[k][w2] = 0;

  //other flows
  field[k][opp(w1)]=field[next(k, opp(w1))][w1];
  field[k][opp(w2)]=field[next(k, opp(w2))][w2];

  field[k][refl(c,w1)]=field[next(k, opp(refl(c,w1)))][refl(c,w1)];
  field[k][refl(c,w2)]=field[next(k, opp(refl(c,w2)))][refl(c,w2)];

  field[k][c] = field[next(k, c)][opp(c)];

  //and also the Extra corner:
  unsigned k1=next(k,0);
  field[k1][opp(w1)] = field[next(k1, opp(w1))][w1];
  field[k1][opp(w2)] = field[next(k1, opp(w2))][w2];

  //Remark: Although it seems like the [opp(wi)] contributions are counted twice each,
  //this is not the case as one of the directions points to the
  //corner itself meaning it just copies the present value.
  //See implementation of Custom GridTypes in geometry.cpp.

}

template<>
inline void Field<LBNode>::ApplyNoSlipFreeConcaveCorner(unsigned corner_x,
                                                        unsigned corner_y,
                                                        Corner c)
{
  //for safety reasons: kill contributions flowing away from the system
  const auto k = GetDomainIndex(corner_x, corner_y);
  for(unsigned v=0; v<lbq; ++v) field[k][v]=0;

  field[k][opp(c)] = field[next(k, opp(c))][c];
}

template<>
inline void Field<LBNode>::ApplyNoSlipFreeConvexCorner(unsigned corner_x,
                                                       unsigned corner_y,
                                                       Corner c)
{
  const auto k = GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Right : Wall::Left);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Back : Wall::Front);

  //for safety reasons: kill contributions flowing away from the system
  field[k][opp(c)] = 0;
  field[k][w1] = 0;
  field[k][w2] = 0;

  //other flows
  field[k][opp(w1)]=field[next(k,opp(w1))][w1];
  field[k][opp(w2)]=field[next(k,opp(w2))][w2];

  field[k][refl(c, w1)]=field[next(k,refl(c, w1))][opp(refl(c, w1))];
  field[k][refl(c, w2)]=field[next(k,refl(c, w2))][opp(refl(c, w2))];

  field[k][c] = field[next(k, c)][opp(c)];

  //and also the Extra corner:
  unsigned k1=next(k,0);
  field[k1][opp(w1)] = field[next(k1, opp(w1))][w1];
  field[k1][opp(w2)] = field[next(k1, opp(w2))][w2];
}

template<>
inline void Field<double>::ApplyNeumannFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    field[k] = field[next(k, opp(w))];
    k = next(k, align(w));
  }
}

template<>
inline double Field<double>::CheckNeumannFreeWall(unsigned start_x,
                                                  unsigned start_y,
                                                  unsigned length,
                                                  Wall w)
{
  const auto beginindex = GetDomainIndex(start_x, start_y);
  double checksum=0, checkvalue=0;

  for(unsigned k = beginindex, i=0;
      i<length;
      k = next(k, align(w)), i++
      )
  {
    checkvalue = field[next(k,opp(w))]-field[k];
    checksum += checkvalue*checkvalue;
  }

  return checksum;
}

template<>
inline void Field<double>::ApplyNeumannFreeConcaveCorner(unsigned corner_x,
                                                         unsigned corner_y,
                                                         Corner c)
{
  const auto k=GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Left : Wall::Right);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Front : Wall::Back);

  field[k] = .5*( field[next(k, opp(w1))]
                 + field[next(k, opp(w2))] );
  //Never actually evaluated.
}

template<>
inline void Field<double>::ApplyNeumannFreeConvexCorner(unsigned corner_x,
                                                        unsigned corner_y,
                                                        Corner c)
{
  const unsigned k = GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Right : Wall::Left);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Back : Wall::Front);

  // seems strange but as for all convex croners all directions except
  // one point to the corner itself (and here we do not know which one)
  // this creates the desired result:
  field[k] = field[next(k, opp(w1))];
  field[k] = field[next(k, opp(w2))];
  //Now take care of fake-corner:
  unsigned k1=next(k,0);
  field[k1] = field[next(k1, opp(w1))];
  field[k1] = field[next(k1, opp(w2))];
}

template<>
inline void Field<double>::ApplyDirichletFreeWall(unsigned start_x,
                                                  unsigned start_y,
                                                  unsigned length,
                                                  Wall w, double value)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    field[k] = 2*value - field[next(k, opp(w))];
    k = next(k, align(w));
  }
}

template<>
inline void Field<double>::ApplyDirichletFreeConcaveCorner(unsigned corner_x,
                                                         unsigned corner_y,
                                                         Corner c, double value)
{
  const auto k=GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Left : Wall::Right);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Front : Wall::Back);

  field[k] = 2*value - .5*( field[next(k, opp(w1))]
                 + field[next(k, opp(w2))] );
  //never actually evaluated
}

template<>
inline void Field<double>::ApplyDirichletFreeConvexCorner(unsigned corner_x,
                                                        unsigned corner_y,
                                                        Corner c, double value)
{
  const unsigned k = GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Right : Wall::Left);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Back : Wall::Front);

  field[k] = field[next(k, opp(w1))];
  field[k] = field[next(k, opp(w2))];
  field[k] = 2*value - field[k];
  //Now take care of fake-corner:
  unsigned k1=next(k,0);
  field[k1] = field[next(k1, opp(w1))];
  field[k1] = field[next(k1, opp(w2))];
  field[k1] = 2*value - field[k1];
}

template<>
inline void Field<double>::CopyDerivativeFreeWall(unsigned start_x,
                                                  unsigned start_y,
                                                  unsigned length,
                                                  Wall w)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    const auto l = next(k, opp(w));
    field[k] = 2*field[l] - field[next(l, opp(w))];
    k = next(k, align(w));
  }
}

template<>
inline void Field<double>::CopyDerivativeFreeConcaveCorner(unsigned corner_x,
                                                           unsigned corner_y,
                                                           Corner c)
{
  const auto k=GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Left : Wall::Right);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Front : Wall::Back);

  const auto l1 = next(k, opp(w1));
  const auto l2 = next(k, opp(w2));

  field[k] = field[l1]+field[l2] - .5*( field[next(l1, opp(w1))]
                 + field[next(l2, opp(w2))] );
  //never actually evaluated and has to be redefined when stencil is changed.
}

template<>
inline void Field<double>::CopyDerivativeFreeConvexCorner(unsigned corner_x,
                                                          unsigned corner_y,
                                                          Corner c)
{
  const unsigned k = GetDomainIndex(corner_x, corner_y);
  const Wall w1 = ( c==Corner::LeftFront or c==Corner::LeftBack ?
                    Wall::Right : Wall::Left);
  const Wall w2 = ( c==Corner::LeftFront or c==Corner::RightFront ?
                    Wall::Back : Wall::Front);

  double aux_value;
  auto l1 = next(k, opp(w1));
  auto l2 = next(k, opp(w2));

  field[k] = field[next(l1, opp(w1))];
  aux_value=field[l1];
  field[k] = field[next(l2, opp(w2))];
  aux_value=field[l2];
  field[k] =   2*aux_value - field[k];

  //Now take care of fake-corner:
  unsigned k1=next(k,0);

  l1 = next(k1, opp(w1));
  l2 = next(k1, opp(w2));

  field[k1] = field[next(l1, opp(w1))];
  aux_value=field[l1];
  field[k1] = field[next(l2, opp(w2))];
  aux_value=field[l2];
  field[k1] =   2*aux_value - field[k1];
}


template<>
inline void Field<LBNode>::ApplyFreeSlipFreeWall(unsigned start_x,
                                                 unsigned start_y,
                                                 unsigned length,
                                                 Wall w)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    for(unsigned v=0; v<lbq; ++v)
    {
      // incoming directions
      if(xdir(w)*xdir(v)==-1 or ydir(w)*ydir(v)==-1)
      {
        const auto u = refl(static_cast<Direction>(v), w);
        field[k][v] = field[next(k, opp(u))][u];
      }
      else
        field[k][v]=0;
    }

    k = next(k, align(w));
  }
}

template<>
inline void Field<LBNode>::ApplyNoSlipFreeWall(unsigned start_x,
                                               unsigned start_y,
                                               unsigned length,
                                               Wall w)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    for(unsigned v=0; v<lbq; ++v)
    {
      // incoming directions
      if(xdir(w)*xdir(v)==-1 or ydir(w)*ydir(v)==-1)
        field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))];
      else
        field[k][v] = 0;
    }

    k = next(k, align(w));
  }
}

template<>
inline void Field<LBNode>::ApplyNoSlipMovingFreeWall(unsigned start_x,
                                               unsigned start_y,
                                               unsigned length,
                                               Wall w, double wallspeed)
{
  const auto beginindex = GetDomainIndex(start_x, start_y);

  for(unsigned k = beginindex, i=0;
      i<length;
      k = next(k, align(w)), i++
      )
  {
    for(unsigned v=0; v<lbq; ++v)
    {
      // incoming directions
      if(xdir(w)*xdir(v)==-1)
      {
        if(ydir(v)>0)
        {
          field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))] + wallspeed*10./3.;
        }
        else if(ydir(v)<0)
        {
          field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))] - wallspeed*10./3.;
        }
        else
        {
        field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))];
        }
      }
      else if(ydir(w)*ydir(v)==-1)
      {
        if(xdir(v)>0)
        {
          field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))] + wallspeed*10./3.;
        }
        else if(xdir(v)<0)
        {
          field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))] - wallspeed*10./3.;
        }
        else
        {
        field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))];
        }
      }
      else
        field[k][v]=0;
    }
  }
}

template<>
inline void Field<LBNode>::ApplyNeumannFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    field[k] = field[next(k, opp(w))];
    k = next(k, align(w));
  }
}

template<>
inline void Field<LBNode>::ApplyOutletFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w,
                                                double uin)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    // array placeholders for node next to wall
    const auto& f = field[next(k, opp(w))];
    // compute velocities
    const double nn = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    const double vx = (f[1] - f[2] + f[5] - f[6] - f[7] + f[8])/nn;
    const double vy = (f[3] - f[4] + f[5] - f[6] + f[7] - f[8])/nn;

    // calculate the equilibrium distribution fe
    const auto fe = GetEqDistribution(vx, vy, nn);

    switch(w)
    {
      case Wall::Left:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k][0]=0;
        field[k][1]=porosity*fe[1];
        field[k][2]=0;
        field[k][3]=0;
        field[k][4]=0;
        field[k][5]=porosity*fe[5];
        field[k][6]=0;
        field[k][7]=porosity*fe[7];
        field[k][8]=0;
      }
      break;

      case Wall::Right:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k][0]=0;
        field[k][1]=0;
        field[k][2]=porosity*fe[2];
        field[k][3]=0;
        field[k][4]=0;
        field[k][5]=0;
        field[k][6]=porosity*fe[6];
        field[k][7]=0;
        field[k][8]=porosity*fe[8];
      }
      break;

      case Wall::Front:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k][0]=0;
        field[k][1]=0;
        field[k][2]=0;
        field[k][3]=porosity*fe[3];
        field[k][4]=0;
        field[k][5]=porosity*fe[5];
        field[k][6]=0;
        field[k][7]=0;
        field[k][8]=porosity*fe[8];
      }
      break;

      case Wall::Back:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k][0]=0;
        field[k][1]=0;
        field[k][2]=0;
        field[k][3]=0;
        field[k][4]=porosity*fe[4];
        field[k][5]=0;
        field[k][6]=porosity*fe[6];
        field[k][7]=porosity*fe[7];
        field[k][8]=0;
      }
      break;
    }

    k = next(k, align(w));
  }
}

template<>
inline void Field<LBNode>::ApplyInletFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w,
                                                double uin_x,
                                                double uin_y,
                                                ScalarField& nn)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  /*switch(w)
  {
    case Wall::Left:
    case Wall::Right:
    {
      for(unsigned i=0; i<length; ++i)
      {
        field[k] = GetEqDistribution(uin, 0, rhoin);
        k = next(k, align(w));
      }
    }
    break;

    case Wall::Front:
    case Wall::Back:
    {
      for(unsigned i=0; i<length; ++i)
      {
        field[k] = GetEqDistribution(0, uin, rhoin);
        k = next(k, align(w));
      }
    }
    break;
  }*/

  for(unsigned i=0; i<length; ++i)
  {
    for(unsigned v=0; v<lbq; ++v)
    {
      // incoming directions
      if(xdir(w)*xdir(v)==-1 or ydir(w)*ydir(v)==-1)
      {
        // get densities and velocities
        const double rho = nn[next(k, v)];
        const double scalar_product=uin_x*xdir(v)+uin_y*ydir(v);
        double weight=1./9.;
        if ( xdir(v)*xdir(v)+ydir(v)*ydir(v)==2)
        {
          weight=1./36.;
        }
        field[k][v] = field[next(k, v)][opp(static_cast<Direction>(v))] + 6*weight*rho*scalar_product ;
      }
      else
        field[k][v] = 0;
    }

    k = next(k, align(w));
  }
}

template<>
inline void Field<double>::ApplyStressNeumannInletFreeWall(unsigned start_x,
                                                        unsigned start_y,
                                                        unsigned length,
                                                        Wall w,
                                                        ScalarField& QQxx,
                                                        ScalarField& QQyx,
                                                        ScalarField& phi,
                                                        double AA,
                                                        double CC,
                                                        double KK,
                                                        double LL,
                                                        double zeta,
                                                        double xi,
                                                        Grid::TensorComponent component)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    unsigned k1 = next(k, opp(w));

    const auto& d = get_neighbours(k1);
    // Q-tensor and binary phase order
    const double Qxx = QQxx[k1];
    const double Qyx = QQyx[k1];
    const double p   = phi[k1];

    // compute derivatives etc.
    double dQxx;
    double dQyx;
    double dPhi;
    double prefactor = ( w==Wall::Left or w==Wall::Front ? -1 : 1 );

    switch(w)
    {
      case Wall::Left:
      case Wall::Right:
      {
        dQxx    = .5*(QQxx[d[1]]-QQxx[d[2]]);
        dQyx    = .5*(QQyx[d[1]]-QQyx[d[2]]);
        dPhi    = .5*(phi[d[1]]-phi[d[2]]);
      }
      break;
      case Wall::Front:
      case Wall::Back:
      {
        dQxx    = .5*(QQxx[d[3]]-QQxx[d[4]]);
        dQyx    = .5*(QQyx[d[3]]-QQyx[d[4]]);
        dPhi    = .5*(phi[d[3]]-phi[d[4]]);
      }
      break;
    }

    const double del2Qxx  = QQxx[d[1]]+QQxx[d[2]]+QQxx[d[3]]+QQxx[d[4]] - 4*QQxx[d[0]];
    const double del2Qyx  = QQyx[d[1]]+QQyx[d[2]]+QQyx[d[3]]+QQyx[d[4]] - 4*QQyx[d[0]];

    // In general I assume dterm=0, dmu=0, and all higher order derivatives=0;
    const double term = p - Qxx*Qxx - Qyx*Qyx;
    const double Hxx = CC*term*Qxx + LL*del2Qxx;
    const double Hyx = CC*term*Qyx + LL*del2Qyx;
    const double dHxx = CC*term*dQxx;
    const double dHyx = CC*term*dQyx;

    switch(component)
    {
      // .. on-diagonal stress components
      case TensorComponent::XX:
      {
        const double del2p = phi[d[1]]+phi[d[2]]+phi[d[3]]+phi[d[4]] - 4*phi[d[0]];
        const double mu = AA*p*(1-p)*(1-2*p) + CC*term - KK*del2p;
        const double dsigmaB = AA*p*(2*p*p-3*p+1)*dPhi - mu*dPhi;
        const double dsigmaF = 2*xi*( 2*dQxx*Qxx*Hxx + (Qxx*Qxx-1)*dHxx
          + dQxx*Qyx*Hyx + Qxx*dQyx*Hyx + Qxx*Qyx*dHyx )
          - zeta*dQxx*p - zeta*Qxx*dPhi;
        field[k] = field[k1]+prefactor*(+dsigmaF+dsigmaB);
        std::cout<< prefactor*(+dsigmaF+dsigmaB) <<std::endl;
      }
      break;
      case TensorComponent::YY:
      {
        const double del2p = phi[d[1]]+phi[d[2]]+phi[d[3]]+phi[d[4]] - 4*phi[d[0]];
        const double mu = AA*p*(1-p)*(1-2*p) + CC*term - KK*del2p;
        const double dsigmaB = AA*p*(2*p*p-3*p+1)*dPhi - mu*dPhi;
        const double dsigmaF = 2*xi*( 2*dQxx*Qxx*Hxx + (Qxx*Qxx-1)*dHxx
          + dQxx*Qyx*Hyx + Qxx*dQyx*Hyx + Qxx*Qyx*dHyx )
          - zeta*dQxx*p - zeta*Qxx*dPhi;
        field[k] = field[k1]+prefactor*(-dsigmaF+dsigmaB);
      }
        break;
      // .. off-diagonal stress components
      case TensorComponent::XY:
      {
        const double dsigmaS = 2*xi*(dQyx*Qxx*Hxx + Qyx*dQxx*Hxx + Qyx*Qxx*dHxx
          + 2*dQyx*Qyx*Hyx + (Qyx*Qyx-1)*dHyx)
          - zeta*dQyx*p - zeta*Qyx*dPhi;
        const double dsigmaA = 2*(dQxx*Hyx+ Qxx*dHyx - dQyx*Hxx - Qyx*dHxx);
        field[k] = field[k1]+prefactor*(dsigmaS+dsigmaA);
      }
      break;
      case TensorComponent::YX:
      {
        const double dsigmaS = 2*xi*(dQyx*Qxx*Hxx + Qyx*dQxx*Hxx + Qyx*Qxx*dHxx
          + 2*dQyx*Qyx*Hyx + (Qyx*Qyx-1)*dHyx)
          - zeta*dQyx*p - zeta*Qyx*dPhi;
        const double dsigmaA = 2*(dQxx*Hyx+ Qxx*dHyx - dQyx*Hxx - Qyx*dHxx);
        field[k] = field[k1]+prefactor*(dsigmaS-dsigmaA);
      }
      break;
    }

    k = next(k, align(w));
  }
}

template<>
inline void Field<double>::ApplyVxOutletFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w,
                                                double uin,
                                                LBField& ffield)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    // array placeholders for node next to wall
    const auto& f = ffield[next(k, opp(w))];
    // compute velocities
    const double nn = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    const double vx = (f[1] - f[2] + f[5] - f[6] - f[7] + f[8])/nn;
    const double vy = (f[3] - f[4] + f[5] - f[6] + f[7] - f[8])/nn;

    // calculate the equilibrium distribution fe
    const auto fe = GetEqDistribution(vx, vy, nn);

    switch(w)
    {
      case Wall::Left:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (porosity*fe[1] - fe[2] + porosity*fe[5] - fe[6] - porosity*fe[7] + fe[8])/nn;
      }
      break;

      case Wall::Right:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (fe[1] - porosity*fe[2] + fe[5] - porosity*fe[6] - fe[7] + porosity*fe[8])/nn;
      }
      break;

      case Wall::Front:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (fe[1] - fe[2] + porosity*fe[5] - fe[6] - fe[7] + porosity*fe[8])/nn;
      }
      break;

      case Wall::Back:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (fe[1] - fe[2] + fe[5] - porosity*fe[6] - porosity*fe[7] + fe[8])/nn;
      }
      break;
    }
    k = next(k, align(w));
  }
}

template<>
inline void Field<double>::ApplyVyOutletFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w,
                                                double uin,
                                                LBField& ffield)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    // array placeholders for node next to wall
    const auto& f = ffield[next(k, opp(w))];
    // compute velocities
    const double nn = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    const double vx = (f[1] - f[2] + f[5] - f[6] - f[7] + f[8])/nn;
    const double vy = (f[3] - f[4] + f[5] - f[6] + f[7] - f[8])/nn;

    // calculate the equilibrium distribution fe
    const auto fe = GetEqDistribution(vx, vy, nn);

    switch(w)
    {
      case Wall::Left:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (fe[3] - fe[4] + porosity*fe[5] - fe[6] + porosity*fe[7] - fe[8])/nn;
      }
      break;

      case Wall::Right:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (fe[3] - fe[4] + fe[5] - porosity*fe[6] + fe[7] - porosity*fe[8])/nn;
      }
      break;

      case Wall::Front:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (porosity*fe[3] - fe[4] + porosity*fe[5] - fe[6] + fe[7] - porosity*fe[8])/nn;
      }
      break;

      case Wall::Back:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (fe[3] - porosity*fe[4] + fe[5] - porosity*fe[6] + porosity*fe[7] - fe[8])/nn;
      }
      break;
    }
    k = next(k, align(w));
  }
}


template<>
inline void Field<double>::ApplyVxPhiOutletFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w,
                                                double uin,
                                                LBField& ffield,
                                                ScalarField& phifield)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    // array placeholders for node next to wall
    const auto& f = ffield[next(k, opp(w))];
    // compute velocities
    const double nn = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    const double vx = (f[1] - f[2] + f[5] - f[6] - f[7] + f[8])/nn;
    const double vy = (f[3] - f[4] + f[5] - f[6] + f[7] - f[8])/nn;

    // calculate the equilibrium distribution fe
    const auto fe = GetEqDistribution(vx, vy, nn);

    switch(w)
    {
      case Wall::Left:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (porosity*fe[1] - fe[2] + porosity*fe[5] - fe[6] - porosity*fe[7] + fe[8])/nn;
      }
      break;

      case Wall::Right:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (fe[1] - porosity*fe[2] + fe[5] - porosity*fe[6] - fe[7] + porosity*fe[8])/nn;
      }
      break;

      case Wall::Front:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (fe[1] - fe[2] + porosity*fe[5] - fe[6] - fe[7] + porosity*fe[8])/nn;
      }
      break;

      case Wall::Back:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (fe[1] - fe[2] + fe[5] - porosity*fe[6] - porosity*fe[7] + fe[8])/nn;
      }
      break;
    }
    field[k]*=phifield[k];
    k = next(k, align(w));
  }
}


template<>
inline void Field<double>::ApplyVyPhiOutletFreeWall(unsigned start_x,
                                                unsigned start_y,
                                                unsigned length,
                                                Wall w,
                                                double uin,
                                                LBField& ffield,
                                                ScalarField& phifield)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    // array placeholders for node next to wall
    const auto& f = ffield[next(k, opp(w))];
    // compute velocities
    const double nn = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    const double vx = (f[1] - f[2] + f[5] - f[6] - f[7] + f[8])/nn;
    const double vy = (f[3] - f[4] + f[5] - f[6] + f[7] - f[8])/nn;

    // calculate the equilibrium distribution fe
    const auto fe = GetEqDistribution(vx, vy, nn);

    switch(w)
    {
      case Wall::Left:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (fe[3] - fe[4] + porosity*fe[5] - fe[6] + porosity*fe[7] - fe[8])/nn;
      }
      break;

      case Wall::Right:
      {
        double porosity=(1+2*vx-4*uin)/(1-2*vx);
        field[k] = (fe[3] - fe[4] + fe[5] - porosity*fe[6] + fe[7] - porosity*fe[8])/nn;
      }
      break;

      case Wall::Front:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (porosity*fe[3] - fe[4] + porosity*fe[5] - fe[6] + fe[7] - porosity*fe[8])/nn;
      }
      break;

      case Wall::Back:
      {
        double porosity=(1+2*vy-4*uin)/(1-2*vy);
        field[k] = (fe[3] - porosity*fe[4] + fe[5] - porosity*fe[6] + porosity*fe[7] - fe[8])/nn;
      }
      break;
    }
    field[k]*=phifield[k];
    k = next(k, align(w));
  }
}

template<>
inline void Field<LBNode>::ApplyPressureOutletFreeWall(unsigned start_x,
                                                    unsigned start_y,
                                                    unsigned length,
                                                    Wall w,
                                                    ScalarField& nn,
                                                    ScalarField& ux,
                                                    ScalarField& uy)
{
  unsigned k = GetDomainIndex(start_x, start_y);

  for(unsigned i=0; i<length; ++i)
  {
    for(unsigned v=0; v<lbq; ++v)
    {
      // incoming directions
      if(xdir(w)*xdir(v)==-1 or ydir(w)*ydir(v)==-1)
      {
        // get densities and velocities
        const double rho = nn[next(k, v)];
        const double vx = ux[next(k, v)];
        const double vy = uy[next(k, v)];

        const double vx2 = ux[next(next(k, v),opp(w))];
        const double vy2 = uy[next(next(k, v),opp(w))];

        const double wallspeed_x=1.5*vx-.5*vx2;
        const double wallspeed_y=1.5*vy-.5*vy2;

        //std::cout << "\n\n" << GetXPosition(k) << "," << GetYPosition(k) << "  ,  " << wallspeed_x << ","<<wallspeed_y;

        double weight=1./9.;
        double scalar_product=-wallspeed_x*xdir(v)-wallspeed_y*ydir(v);
        if ( xdir(v)*xdir(v)+ydir(v)*ydir(v)==2)
        {
          weight=1./36.;
        }

        field[k][v] = -field[next(k, v)][opp(static_cast<Direction>(v))]+rho*weight*(2+9*scalar_product*scalar_product-3*(wallspeed_x*wallspeed_x+wallspeed_y*wallspeed_y));
      }
      else
        field[k][v] = 0;
    }

    k = next(k, align(w));
  }
}


#endif//FIELDS_HPP_
