#ifndef GEOMETRY_HPP_
#define GEOMETRY_HPP_

#include "header.hpp"
#include <array>
#include <map>
#include <vector>
#include <memory>

/** Dimension */
constexpr static unsigned dim = 2;

/** Number of directions in the LB model */
constexpr static unsigned lbq = 9;

/** Coordinate (x, y) of neighbours / directions */
constexpr static std::array<int, 2> directions[] = {
  { 0, 0},
  { 1, 0},
  {-1, 0},
  { 0, 1},
  { 0,-1},
  { 1, 1},
  {-1,-1},
  {-1, 1},
  { 1,-1}
};

/** Reflection w.r.t. an arbitrary axis
 *
 * Defined as follows: directions_refl[i][j] gives the direction j reflected
 * about the plane with normal in direction i. For example, for bounce back
 * (i=0) we have
 *
 *  neighbours[i][j] == -neighbours[directions_refl[0][i]][j]
 *
 * The axis is the normal vector of the reflecting plane and defined in the same
 * order as the directions, apart from the first component which gives a simple
 * bounce back.
 * */
constexpr static std::array<unsigned, lbq> directions_refl[] = {
  // bounce back: (0,0)
  {0, 2, 1, 4, 3, 6, 5, 8, 7},
  // y-axis: (1,0)
  {0, 2, 1, 3, 4, 7, 8, 5, 6},
  // y-axis: (-1,0)
  {0, 2, 1, 3, 4, 7, 8, 5, 6},
  // x-axis: (0,1)
  {0, 1, 2, 4, 3, 8, 7, 6, 5},
  // x-axis: (0,-1)
  {0, 1, 2, 4, 3, 8, 7, 6, 5},
  // tilted plane with normal (1, 1)
  {0, 4, 3, 2, 1, 6, 5, 7, 8},
  // tilted plane with normal (-1, -1)
  {0, 4, 3, 2, 1, 6, 5, 7, 8},
  // tilted plane with normal (-1, 1)
  {0, 3, 4, 1, 2, 5, 6, 8, 7},
  // tilted plane with normal (1, -1)
  {0, 3, 4, 1, 2, 5, 6, 8, 7}
};

/** Projection into an arbitrary axis
 *
 * Defined as follows: directions_proj[i][j] gives the direction j projected
 * onto the plane with normal in direction i. See direction_refl for details.
 * */
constexpr static std::array<unsigned, lbq> directions_proj[] = {
  // (0,0)
  {0, 0, 0, 0, 0, 0, 0, 0, 0},
  // y-axis
  {0, 0, 0, 3, 4, 3, 4, 3, 4},
  // y-axis
  {0, 0, 0, 3, 4, 3, 4, 3, 4},
  // x-axis
  {0, 1, 2, 0, 0, 1, 2, 2, 1},
  // x-axis
  {0, 1, 2, 0, 0, 1, 2, 2, 1},
  // tilted plane with normal (1, 1)
  {0, 5, 6, 5, 6, 5, 6, 0, 0},
  // tilted plane with normal (-1, -1)
  {0, 1, 2, 4, 3, 8, 7, 6, 5},
  // tilted plane with normal (-1, 1)
  {0, 8, 7, 7, 8, 0, 0, 7, 8},
  // tilted plane with normal (1, -1)
  {0, 8, 7, 7, 8, 0, 0, 7, 8}
};

/** Names for the directions */
struct Direction_enum
{
  enum Direction {
    /** Center: (0, 0) */
    Center = 0,
    /** Right: (1, 0) */
    Right  = 1,
    /** Left: (-1, 0) */
    Left   = 2,
    /** Back: (0, 1) */
    Back   = 3,
    /** Front: (0,-1) */
    Front  = 4,
    /** Right/Back: (1, 1) */
    RightBack  = 5,
    /** Left/Front: (-1, -1) */
    LeftFront  = 6,
    /** Left/Back: (-1, 1) */
    LeftBack   = 7,
    /** Right/Front: (1, -1) */
    RightFront = 8,
  };
};
/** Implementation */
using Direction = Direction_enum::Direction;

/** Domain enum
 *
 * This enum is trivial but implemented for completeness.
 * */
enum Domain_enum
{
  /** Center is the only value */
  Domain = 0
};

/** Boundary layer
 *
 * This enum is trivial but implemented for completeness.
 * */
enum BoundaryLayer_enum
{
    /** Center is the only value */
    BoundaryLayer = 0
};

/** Supplementary enum
 *
 * This enum is trivial but implemented for completeness.
 * */
enum ExtraNode_enum
{
  /** Center is the only value */
  ExtraNode = 0
};

/** The different walls
 *
 * The numerical definition agrees with the definition of Direction().
 * */
struct Wall_enum
{
  enum Wall {
    /** Left wall, defined as x=0 */
    Left  = 2,
    /** Back wall, defined as x=LX-1 */
    Right = 1,
    /** Front wall, defined as y=0 */
    Front = 4,
    /** Back wall, defined as y=LY-1 */
    Back  = 3
  };
};
/** Implementation */
using Wall = Wall_enum::Wall;

/** The different walls when dealing with periodic boundary conditions
 *
 * In the case of pbc, the walls are identified with their opposite wall and
 * there is no need of differentiating them.
 * */
struct PBCWall_enum
{
  enum PBCWall
  {
    /** Left and/or right wall */
    LeftRight = 2,
    /** Front and/or back wall */
    FrontBack = 4
  };
};
/** Implementation */
using PBCWall = PBCWall_enum::PBCWall;

/** The different corners */
struct Corner_enum
{
  enum Corner
  {
    /** LeftFront corner, defined as x=0 and y=0 */
    LeftFront  = 6,
    /** RightFront corner, defined as x=LX and y=0 */
    RightFront = 8,
    /** LeftBack corner, defined as x=0 and y=LY */
    LeftBack   = 7,
    /** RightBack corner, defined as x=LX and y=LY */
    RightBack  = 5
  };
};
/** Implementation */
using Corner = Corner_enum::Corner;

// =============================================================================
// Helper functions

/** Return a direction vector */
inline std::array<int, 2> dir(unsigned d)
{
  return directions[d];
}

/** Return x-coord. of a direction */
inline int xdir(unsigned d)
{
  return directions[d][0];
}

/** Return y-coord. of a direction */
inline int ydir(unsigned d)
{
  return directions[d][1];
}

/** Return reflected direction */
inline unsigned refl(unsigned d, unsigned axis=0)
{
  return directions_refl[axis][d];
}

/** Return reflected direction */
inline Direction refl(Direction d, unsigned axis=0)
{
  return static_cast<Direction>(directions_refl[axis][d]);
}

/** Return reflected direction */
inline Wall refl(Wall w, unsigned axis=0)
{
  return static_cast<Wall>(directions_refl[axis][w]);
}

/** Return reflected direction */
inline Corner refl(Corner c, unsigned axis=0)
{
  return static_cast<Corner>(directions_refl[axis][c]);
}

/** Return projected direction */
inline unsigned proj(unsigned d, unsigned axis=0)
{
  return directions_proj[axis][d];
}

/** Return opposite direction */
inline Direction opp(Direction d)
{
  return refl(d, 0);
}

/** Return opposite wall */
inline Wall opp(Wall w)
{
  return refl(w, 0);
}

/** Return opposite corner */
inline Corner opp(Corner c)
{
  return refl(c, 0);
}

/** Return opposite direction (unsigned) */
inline Direction opp(unsigned d)
{
  return refl(static_cast<Direction>(d), 0);
}

/** Return the coord of a direction projected on the normal of a given wall. */
inline int dir(unsigned d, Wall w)
{
  switch(w)
  {
    case Wall::Left:  return opp(xdir(d));
    case Wall::Right: return xdir(d);
    case Wall::Front: return opp(ydir(d));
    case Wall::Back:  return ydir(d);
  }
  return 0;
}

/** Allows to iterate over cells parallel to a wall
 *
 * By definition, we want to iterate from smaller to larger indices.
 * */
inline Wall align(Wall w)
{
  switch(w)
  {
    case Wall::Left:  return Wall::Back;
    case Wall::Right: return Wall::Back;
    case Wall::Front: return Wall::Right;
    case Wall::Back:  return Wall::Right;
  }
  return Wall::Right;
}

/** Returns true if a direction outgoing w.r.t. a given wall */
inline bool is_outgoing(unsigned v, Wall w)
{
  // note that walls can not have both xdir and ydir non zero
  return xdir(w)*xdir(v)==1 or ydir(w)*ydir(v)==1;
}

/** Returns true if a direction incoming w.r.t. a given wall */
inline bool is_incoming(unsigned v, Wall w)
{
  // note that walls can not have both xdir and ydir non zero
  return xdir(w)*xdir(v)==-1 or ydir(w)*ydir(v)==-1;
}

// =============================================================================
// Grid class

/** Neighbours of a certain grid node */
using NeighboursList = std::array<unsigned, 9>;

/** Defines basic geometrical properties
 *
 * A grid only contains information about its size and dimension,  including a
 * boundary layer. We use this base class because this information is required
 * both in the Field() and Model() classes. This facilitates the implementation
 * of boundary conditions as it abstracts the underlying model for the domain
 * and the boundary layer. The total size of the grid is defined by its
 * dimensions and the thickness of the bdry layer.
 * */
class Grid
{
public:

  /** Defines the type of the grid
   *
   * This is used only in the constructor. In principle this could be replaced
   * by a bool but such an implementation avoids bugs comming from downcasting.
   * */
  enum class GridType {
    Periodic = 0,
    Layer    = 1,
    Custom1  = 2,
    Custom2  = 3,
    Custom3  = 4,
    Custom4  = 5,
    Chimney  = 6,
    Chimney4 = 7,
    Chimney7 = 8,
    Chimney10= 9,
    Chimney13= 10,
    Chimney16= 11,
    ChimneyExit=12,
    ChimneyExit4 = 13,
    ChimneyExit7 = 14,
    ChimneyExit10= 15,
    ChimneyExit13= 16,
    ChimneyExit16= 17,
    SquareAn= 18
  };

  /** This is used only in the derivatives for an inlet.
   * */
  enum class TensorComponent {
    XX  = 0,
    YY  = 1,
    YX  = 2,
    XY  = 3
  };

  /** Those parameters are just needed for the funnel geometry.
   * ...
   * */
  unsigned l=300; //parameter for "funnel"-geometry
  unsigned d=40; //parameter for "funnel"-geometry
  unsigned o=100; //parameter for "funnel"-geometry, for top-down symmetric system: o=(LY-l)/2
  unsigned r; //parameter for "funnel"-geometry, heigth of upper reservoir such that LY=o+l+r; r=o iff o=(LY-l)/2
  unsigned o2=o+l; //parameter for "funnel"-geometry, o2=o+l which makes it (LY+l)/2 in the case of o=(LY-l)/2

  unsigned SqX_left=20;
  unsigned SqY_front=20;
  unsigned SqHeight=20;
  unsigned SqLength=80;
  unsigned SqWallHeight=19; //SqHeight-1??
  unsigned SqWallLength=79;
  unsigned SqY_back=40;
  unsigned SqX_right=100;

private:
  /** Ptr to actual array containing the neighbour list */
  std::shared_ptr<std::vector<NeighboursList>> neighbours_ptr;

  /** Update the list of neighbours and pointer ptr */
  void UpdateNeighboursList();

protected:
  /** Dimensions */
  unsigned LX, LY;
//  unsigned    SqX_left, SqY_front, SqHeight,  SqLength;
//  unsigned    SqWallHeight, SqWallLength, SqY_back,  SqX_right;
  /** Is the grid periodic? (no ghost cells) */
  GridType Type = GridType::Periodic;
  /** Total number of nodes in the main domain, without the boundary layers */
  unsigned DomainSize;
  /** Total number of nodes including the boundary layer */
  unsigned TotalSize;

public:
  /** Default empty constructor */
  Grid() = default;
  /** Construct from a given size */
  Grid(unsigned LX_, unsigned LY_, GridType Type_)
  { SetSize(LX_, LY_, Type_); }
  /** Virtual destructor */
  virtual ~Grid();

  /** Set size */
  virtual void SetSize(unsigned LX_, unsigned LY_, GridType Type_);

  /** Get the domain x position of a node from its absolute index */
  inline unsigned GetXPosition(unsigned k) const
  { return k/LY; }
  /** Get the domain y position of a node from its absolute index */
  inline unsigned GetYPosition(unsigned k) const
  { return k%LY; }
  /** Return domain index from position */
  inline unsigned GetDomainIndex(unsigned x, unsigned y) const
  { return y + LY*x; }

  /** Return absolute index from domain index
   *
   * Return absolute index from an index within the main domain < DomainSize. At
   * the moment this function is trivial because we store the domain nodes
   * before the buffer nodes. But this might change if we change our data model
   * in the future.
   * */
  inline unsigned GetIndex(Domain_enum, unsigned k=0) const
  { return k; }

  /** Return boudary layer index
   *
   * Returns the absolute index of all boudary cells in order. This function is useful
   * e.g. to set all ghost cells to a certain value.
   * */
  inline unsigned GetIndex(BoundaryLayer_enum, unsigned k=0) const
  { return DomainSize+k; }

  /** Return index of extra nodes
   *
   * Returns the absolute index of all auxilary cells in order. This function is useful
   * e.g. to set all those additional cells to a certain value.
   * */
  inline unsigned GetIndex(ExtraNode_enum, unsigned k=0) const
  { return DomainSize+k; }

  /** Return absolute index from index on wall
   *
   * This function provides an effective way to iterate over the walls,
   * independently of the data model used internally. For example, iterating
   * over the Left wall (excluding corners) can be simply acheived by:
   *
   *  for(unsigned k=0; k<Field.GetSize(Wall::Left); ++k)
   *  {
   *    const unsigned i = Field.GetIndex(Wall::Left, k);
   *
   *    // do smth there
   *    Field[i] = ...
   *  }
   *
   * Note that no bound checking is performed.
   * */
  inline unsigned GetIndex(Wall w, unsigned k) const
  {
    // any decent compiler will optimise this out
    switch(w)
    {
      case Wall::Left:  return DomainSize + k;
      case Wall::Right: return DomainSize + LY + k;
      case Wall::Front: return DomainSize + 2*LY + k;
      case Wall::Back:  return DomainSize + 2*LY + LX + k;
    }
    // zis can't be!
    return -1;
  }
  /** Return absolute index of corner
   *
   * The corners are stored after everything else (domain and walls) in the same
   * order as in the definition of Corner. Similar to GetWallIndex().
   * */
  inline unsigned GetIndex(Corner c, unsigned=0) const
  {
    switch(c)
    {
      case Corner::LeftFront:  return TotalSize - 4;
      case Corner::RightFront: return TotalSize - 3;
      case Corner::LeftBack:   return TotalSize - 2;
      case Corner::RightBack:  return TotalSize - 1;
    }
    return -1;
  }

  /** Return size of the domain */
  inline unsigned GetSize(Domain_enum=Domain) const
  { return DomainSize; }
  /** Return size of the full boundary layer */
  inline unsigned GetSize(BoundaryLayer_enum) const
  { return TotalSize-DomainSize; }
  /** Return the size of a wall */
  inline unsigned GetSize(Wall w) const
  {
    switch(w)
    {
      case Wall::Left:  return LY;
      case Wall::Right: return LY;
      case Wall::Front: return LX;
      case Wall::Back:  return LX;
    }
    return -1;
  }
  /** Specialization for PBCWall() */
  inline unsigned GetSize(PBCWall w) const
  {
    switch(w)
    {
    case PBCWall::LeftRight: return GetSize(Wall::Left);
    case PBCWall::FrontBack: return GetSize(Wall::Front);
    }
    return -1;
  }
  /** Specialization for Corners */
  inline unsigned GetSize(Corner) const
  { return 1; }
  /** Specialization for extra nodes */
  inline unsigned GetSize(ExtraNode_enum) const
  { return 1; }

  /** Returns index of node i in direction d */
  inline unsigned next(unsigned i, unsigned d) const
  { return neighbours_ptr->at(i)[d]; }
  /** Return list of neighbours of a given node */
  inline const NeighboursList& get_neighbours(unsigned i) const
  { return neighbours_ptr->at(i); }
  /** Return whole neighbours list */
  inline auto get_all_neighbours() const
  { return neighbours_ptr; }

  /** initialize an extra corner */
  void init_extra_corner(unsigned x,unsigned y, unsigned extra_node_index, Corner c)
  {
    unsigned k1; //neighbouring field that points towards fake corner
    unsigned k2=GetIndex(ExtraNode_enum::ExtraNode, extra_node_index); //fake corner that is connected only to this field
    unsigned k3=GetDomainIndex(x, y); //the "real" corner
    switch(c)
    {
      case Corner::LeftFront:
      case Corner::RightFront:
      {
        k1=GetDomainIndex(x, y-1);
        neighbours_ptr->at(k1)[3] = k2;
        neighbours_ptr->at(k2)[4] = k1;
        neighbours_ptr->at(k3)[4] = k3;
        neighbours_ptr->at(k2)[0] = k3;
        neighbours_ptr->at(k3)[0] = k2;
        break;
      }
      case Corner::LeftBack:
      case Corner::RightBack:
      {
        k1=GetDomainIndex(x, y+1);
        neighbours_ptr->at(k1)[4] = k2;
        neighbours_ptr->at(k2)[3] = k1;
        neighbours_ptr->at(k3)[3] = k3;
        neighbours_ptr->at(k2)[0] = k3;
        neighbours_ptr->at(k3)[0] = k2;
        break;
      }
    }
  }
};

#endif//GEOMETRY_HPP_
