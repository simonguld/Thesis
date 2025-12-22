#include "header.hpp"
#include <map>
#include <tuple>
#include "geometry.hpp"
#include "tools.hpp"

using namespace std;

/** Precomputed neighbours list */
static std::map<
    std::tuple<unsigned, unsigned, Grid::GridType>,
    std::weak_ptr<std::vector<NeighboursList>>
  > neighbours_pre;

void Grid::SetSize(unsigned LX_, unsigned LY_, GridType Type_)
{
  // store values
  LX = LX_;
  LY = LY_;
  Type = Type_;
  DomainSize = LX*LY;

  switch(Type)
  {
    case GridType::Periodic:
      TotalSize = DomainSize;
      break;
    case GridType::Layer:
      TotalSize = (LX+2)*(LY+2);
      break;
    case GridType::Custom1:
      TotalSize = DomainSize+4;
      break;
    case GridType::Custom2:
      TotalSize = DomainSize+2;
      break;
    case GridType::Custom3:
      TotalSize = DomainSize+20;
      break;
    case GridType::Custom4:
      TotalSize = DomainSize+40;
      break;
    case GridType::Chimney:
      TotalSize = DomainSize+2;
      break;
    case GridType::Chimney4:
      TotalSize = DomainSize+8;
      break;
    case GridType::Chimney7:
      TotalSize = DomainSize+10;
      break;
    case GridType::Chimney10:
      TotalSize = DomainSize+14;
      break;
    case GridType::Chimney13:
      TotalSize = DomainSize+18;
      break;
    case GridType::Chimney16:
      TotalSize = DomainSize+20;
      break;
    case GridType::ChimneyExit:
      TotalSize = DomainSize+2;
      break;
    case GridType::ChimneyExit4:
      TotalSize = DomainSize+8;
      break;
    case GridType::ChimneyExit7:
      TotalSize = DomainSize+10;
      break;
    case GridType::ChimneyExit10:
      TotalSize = DomainSize+14;
      break;
    case GridType::ChimneyExit13:
      TotalSize = DomainSize+18;
      break;
    case GridType::ChimneyExit16:
      TotalSize = DomainSize+20;
      break;
    case GridType::SquareAn:
      TotalSize = DomainSize+4;
      break;
  }
  // update neighbours
  UpdateNeighboursList();
}

void Grid::UpdateNeighboursList()
{
  // if neighbours map has not been pre-computed, then pre-compute it!
  if(neighbours_pre.count(make_tuple(LX, LY, Type))==0)
  {
    // create new shared neighbours_ptr
    neighbours_ptr = make_shared<std::vector<NeighboursList>>();
    // register
    neighbours_pre[make_tuple(LX, LY, Type)] = neighbours_ptr;
    // resize underlying array
    neighbours_ptr->resize(TotalSize);

    // periodic bdry conditions
    if(Type==GridType::Periodic)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<GetSize(); ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }
    }

    else if(Type==GridType::SquareAn)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }

      init_extra_corner(SqX_left, SqY_back , 0, Corner::LeftBack);
      init_extra_corner(SqX_right, SqY_back , 1, Corner::RightBack);
      init_extra_corner(SqX_left, SqY_front , 2, Corner::LeftFront);
      init_extra_corner(SqX_right, SqY_front , 3, Corner::RightFront);

    }
    // Layer boundary condition requires a boundary layer (ghost cells)
    else if(Type==GridType::Layer)
    {
      // the general strategy to generate the list of neighbours in the presence
      // of the boundary layer is to use the same algorithm as for the pbc but
      // with a conversion function that converts indices on the bigger grid of
      // size (LX+2)*(LY+2) to the actual index where the boundaries are stored
      // (after the main domain).

      // conversion function
      auto conv_index = [this](unsigned x, unsigned y) -> unsigned
      {
        // ...left-front corner
        if(x==0 and y==0) return GetIndex(Corner::LeftFront);
        // ...right-front corner
        if(x==LX+1 and y==0) return GetIndex(Corner::RightFront);
        // ...left-back corner
        if(x==0 and y==LY+1) return GetIndex(Corner::LeftBack);
        // ...right-back corner
        if(x==LX+1 and y==LY+1) return GetIndex(Corner::RightBack);
        // walls...
        // ...left wall
        if(x==0) return GetIndex(Wall::Left, y-1);
        // ...right wall
        if(x==LX+1) return GetIndex(Wall::Right, y-1);
        // ...front wall
        if(y==0) return GetIndex(Wall::Front, x-1);
        // ...back wall
        if(y==LY+1) return GetIndex(Wall::Back, x-1);
        // ...domain
        return GetDomainIndex(x-1, y-1);
      };

      // define the neighbours, accounting for the periodic boundaries on a bigger
      // grid of size (LX+2)*(LY+2), and convert to correct indices.
      for(unsigned k=0; k<TotalSize; ++k)
      {
        const long int x = k/(LY+2);
        const long int y = k%(LY+2);

        for(unsigned v=0; v<9; ++v)
          neighbours_ptr->at(conv_index(x, y))[v] = conv_index(modu(x + xdir(v), LX+2),
                                                modu(y + ydir(v), LY+2));
      }
    }
    //Custom boundary condotions have more corners but are otherwise like the parallel grid.
    else if(Type==GridType::Custom1)
    {
      //determine auxilary parameter r
      r=LY-o2;
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }
      //Then adjust the pointers to the auxilary nodes.
      //Also adjust the auxilary nodes. Here only one direction is well-defined,
      //all others only point towards the node itself so that iterating over the
      //directions cannot bring you anywhere but to the one direction.
      //Also the "real" corner is adjusted such that there are no "double relations".
      //Additionally the "0-direction" of the "real" corner points to the fake corner
      //and vice versa.
      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //neighbouring field that points towards fake corner
      //unsigned k1=GetDomainIndex(LX-d-1, o-1);
      //fake corner that is connected only to this field
      //unsigned k2=GetIndex(ExtraNode_enum::ExtraNode, 0);
      //for the "real" corner this direction then is redirected towards itself
      //unsigned k3=GetDomainIndex(LX-d-1, o);
      //neighbours_ptr->at(k1)[3] = k2;
      //neighbours_ptr->at(k2)[4] = k1;
      //neighbours_ptr->at(k3)[4] = k3;
      //neighbours_ptr->at(k2)[0] = k3;
      //neighbours_ptr->at(k3)[0] = k2;
      init_extra_corner(LX-d-1, o  , 0, Corner::LeftFront);


      //k1=GetDomainIndex(d, o-1);
      //k2=GetIndex(ExtraNode_enum::ExtraNode, 1);
      //k3=GetDomainIndex(d, o);
      //neighbours_ptr->at(k1)[3] = k2;
      //neighbours_ptr->at(k2)[4] = k1;
      //neighbours_ptr->at(k3)[4] = k3;
      //neighbours_ptr->at(k2)[0] = k3;
      //neighbours_ptr->at(k3)[0] = k2;
      init_extra_corner(d, o  , 1, Corner::RightFront);


      //k1=GetDomainIndex(LX-d-1, o2);
      //k2=GetIndex(ExtraNode_enum::ExtraNode, 2);
      //k3=GetDomainIndex(LX-d-1, o2-1);
      //neighbours_ptr->at(k1)[4] = k2;
      //neighbours_ptr->at(k2)[3] = k1;
      //neighbours_ptr->at(k3)[3] = k3;
      //neighbours_ptr->at(k2)[0] = k3;
      //neighbours_ptr->at(k3)[0] = k2;
      init_extra_corner(LX-d-1, o2-1  , 2, Corner::LeftBack);


      //k1=GetDomainIndex(d, o2);
      //k2=GetIndex(ExtraNode_enum::ExtraNode, 3);
      //k3=GetDomainIndex(d, o2-1);
      //neighbours_ptr->at(k1)[4] = k2;
      //neighbours_ptr->at(k2)[3] = k1;
      //neighbours_ptr->at(k3)[3] = k3;
      //neighbours_ptr->at(k2)[0] = k3;
      //neighbours_ptr->at(k3)[0] = k2;
      init_extra_corner(d, o2-1  , 3, Corner::RightBack);

    }
    //Another custom configuration containing 2 concave corners.
    else if(Type==GridType::Custom2)
    {
      //determine auxilary parameter r
      r=LY-1-l;
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }

      //unsigned k1=GetDomainIndex(LX-d-1, l+1);
      //unsigned k2=GetIndex(ExtraNode_enum::ExtraNode, 0);
      //unsigned k3=GetDomainIndex(LX-d-1, l);
      //neighbours_ptr->at(k1)[4] = k2;
      //neighbours_ptr->at(k2)[3] = k1;
      //neighbours_ptr->at(k3)[3] = k3;
      //neighbours_ptr->at(k2)[0] = k3;
      //neighbours_ptr->at(k3)[0] = k2;

      //k1=GetDomainIndex(d, l+1);
      //k2=GetIndex(ExtraNode_enum::ExtraNode, 1);
      //k3=GetDomainIndex(d, l);
      //neighbours_ptr->at(k1)[4] = k2;
      //neighbours_ptr->at(k2)[3] = k1;
      //neighbours_ptr->at(k3)[3] = k3;
      //neighbours_ptr->at(k2)[0] = k3;
      //neighbours_ptr->at(k3)[0] = k2;

      init_extra_corner(LX-d-1, l , 0, Corner::LeftBack);
      init_extra_corner(d, l , 1, Corner::RightBack);


    }
    //Another custom configuration containing 2 concave corners, but with smooth corners.
    else if(Type==GridType::Custom3)
    {
      //determine auxilary parameter r
      r=LY-o2;
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }

      //Lower left corner
      init_extra_corner(d-5, o  , 0, Corner::RightFront);
      init_extra_corner(d-4, o+1, 1, Corner::RightFront);
      init_extra_corner(d-2, o+2, 2, Corner::RightFront);
      init_extra_corner(d-1, o+4, 3, Corner::RightFront);
      init_extra_corner(d  , o+5, 4, Corner::RightFront);

      //Lower right corner
      init_extra_corner(LX-d+4, o  , 5, Corner::LeftFront);
      init_extra_corner(LX-d+3, o+1, 6, Corner::LeftFront);
      init_extra_corner(LX-d+1, o+2, 7, Corner::LeftFront);
      init_extra_corner(LX-d  , o+4, 8, Corner::LeftFront);
      init_extra_corner(LX-d-1, o+5, 9, Corner::LeftFront);

      //Upper left corner
      init_extra_corner(d-5, o2-1, 10, Corner::RightBack);
      init_extra_corner(d-4, o2-2, 11, Corner::RightBack);
      init_extra_corner(d-2, o2-3, 12, Corner::RightBack);
      init_extra_corner(d-1, o2-5, 13, Corner::RightBack);
      init_extra_corner(d  , o2-6, 14, Corner::RightBack);

      //Upper right corner
      init_extra_corner(LX-d+4, o2-1, 15, Corner::LeftBack);
      init_extra_corner(LX-d+3, o2-2, 16, Corner::LeftBack);
      init_extra_corner(LX-d+1, o2-3, 17, Corner::LeftBack);
      init_extra_corner(LX-d  , o2-5, 18, Corner::LeftBack);
      init_extra_corner(LX-d-1, o2-6, 19, Corner::LeftBack);
    }

    //Another custom configuration containing smooth corners.
    else if(Type==GridType::Custom4)
    {
      //determine auxilary parameter r
      r=LY-o2;
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d-12, o+0, 0, Corner::RightFront);
      init_extra_corner(d-9, o+1, 1, Corner::RightFront);
      init_extra_corner(d-7, o+2, 2, Corner::RightFront);
      init_extra_corner(d-6, o+3, 3, Corner::RightFront);
      init_extra_corner(d-5, o+4, 4, Corner::RightFront);
      init_extra_corner(d-4, o+5, 5, Corner::RightFront);
      init_extra_corner(d-3, o+6, 6, Corner::RightFront);
      init_extra_corner(d-2, o+7, 7, Corner::RightFront);
      init_extra_corner(d-1, o+9, 8, Corner::RightFront);
      init_extra_corner(d+0, o+12, 9, Corner::RightFront);
      init_extra_corner(LX-d-1+0, o+12, 10, Corner::LeftFront);
      init_extra_corner(LX-d-1+1, o+9, 11, Corner::LeftFront);
      init_extra_corner(LX-d-1+2, o+7, 12, Corner::LeftFront);
      init_extra_corner(LX-d-1+3, o+6, 13, Corner::LeftFront);
      init_extra_corner(LX-d-1+4, o+5, 14, Corner::LeftFront);
      init_extra_corner(LX-d-1+5, o+4, 15, Corner::LeftFront);
      init_extra_corner(LX-d-1+6, o+3, 16, Corner::LeftFront);
      init_extra_corner(LX-d-1+7, o+2, 17, Corner::LeftFront);
      init_extra_corner(LX-d-1+9, o+1, 18, Corner::LeftFront);
      init_extra_corner(LX-d-1+12, o+0, 19, Corner::LeftFront);
      init_extra_corner(d+0, o2-1-12, 20, Corner::RightBack);
      init_extra_corner(d-1, o2-1-9, 21, Corner::RightBack);
      init_extra_corner(d-2, o2-1-7, 22, Corner::RightBack);
      init_extra_corner(d-3, o2-1-6, 23, Corner::RightBack);
      init_extra_corner(d-4, o2-1-5, 24, Corner::RightBack);
      init_extra_corner(d-5, o2-1-4, 25, Corner::RightBack);
      init_extra_corner(d-6, o2-1-3, 26, Corner::RightBack);
      init_extra_corner(d-7, o2-1-2, 27, Corner::RightBack);
      init_extra_corner(d-9, o2-1-1, 28, Corner::RightBack);
      init_extra_corner(d-12, o2-1+0, 29, Corner::RightBack);
      init_extra_corner(LX-d-1+12, o2-1+0, 30, Corner::LeftBack);
      init_extra_corner(LX-d-1+9, o2-1-1, 31, Corner::LeftBack);
      init_extra_corner(LX-d-1+7, o2-1-2, 32, Corner::LeftBack);
      init_extra_corner(LX-d-1+6, o2-1-3, 33, Corner::LeftBack);
      init_extra_corner(LX-d-1+5, o2-1-4, 34, Corner::LeftBack);
      init_extra_corner(LX-d-1+4, o2-1-5, 35, Corner::LeftBack);
      init_extra_corner(LX-d-1+3, o2-1-6, 36, Corner::LeftBack);
      init_extra_corner(LX-d-1+2, o2-1-7, 37, Corner::LeftBack);
      init_extra_corner(LX-d-1+1, o2-1-9, 38, Corner::LeftBack);
      init_extra_corner(LX-d-1+0, o2-1-12, 39, Corner::LeftBack);
    }
    //Another custom configuration containing 2 concave corners.
    else if(Type==GridType::Chimney)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(LX-d-1, o, 0, Corner::LeftFront);
      init_extra_corner(d, o, 1, Corner::RightFront);
    }
    //reservoir with chimney, radius corners=4
    else if(Type==GridType::Chimney4)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d-3, o+0, 0, Corner::RightFront);
      init_extra_corner(d-2, o+1, 1, Corner::RightFront);
      init_extra_corner(d-1, o+2, 2, Corner::RightFront);
      init_extra_corner(d+0, o+3, 3, Corner::RightFront);
      init_extra_corner(LX-d-1+0, o+3, 4, Corner::LeftFront);
      init_extra_corner(LX-d-1+1, o+2, 5, Corner::LeftFront);
      init_extra_corner(LX-d-1+2, o+1, 6, Corner::LeftFront);
      init_extra_corner(LX-d-1+3, o+0, 7, Corner::LeftFront);
    }
    //reservoir with chimney, radius corners=7
    else if(Type==GridType::Chimney7)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d-5, o+0, 0, Corner::RightFront);
      init_extra_corner(d-3, o+1, 1, Corner::RightFront);
      init_extra_corner(d-2, o+2, 2, Corner::RightFront);
      init_extra_corner(d-1, o+3, 3, Corner::RightFront);
      init_extra_corner(d+0, o+5, 4, Corner::RightFront);
      init_extra_corner(LX-d-1+0, o+5, 5, Corner::LeftFront);
      init_extra_corner(LX-d-1+1, o+3, 6, Corner::LeftFront);
      init_extra_corner(LX-d-1+2, o+2, 7, Corner::LeftFront);
      init_extra_corner(LX-d-1+3, o+1, 8, Corner::LeftFront);
      init_extra_corner(LX-d-1+5, o+0, 9, Corner::LeftFront);
    }
    //reservoir with chimney, radius corners=10
    else if(Type==GridType::Chimney10)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d-7, o+0, 0, Corner::RightFront);
      init_extra_corner(d-5, o+1, 1, Corner::RightFront);
      init_extra_corner(d-4, o+2, 2, Corner::RightFront);
      init_extra_corner(d-3, o+3, 3, Corner::RightFront);
      init_extra_corner(d-2, o+4, 4, Corner::RightFront);
      init_extra_corner(d-1, o+5, 5, Corner::RightFront);
      init_extra_corner(d+0, o+7, 6, Corner::RightFront);
      init_extra_corner(LX-d-1+0, o+7, 7, Corner::LeftFront);
      init_extra_corner(LX-d-1+1, o+5, 8, Corner::LeftFront);
      init_extra_corner(LX-d-1+2, o+4, 9, Corner::LeftFront);
      init_extra_corner(LX-d-1+3, o+3, 10, Corner::LeftFront);
      init_extra_corner(LX-d-1+4, o+2, 11, Corner::LeftFront);
      init_extra_corner(LX-d-1+5, o+1, 12, Corner::LeftFront);
      init_extra_corner(LX-d-1+7, o+0, 13, Corner::LeftFront);
    }
    //reservoir with chimney, radius corners=13
    else if(Type==GridType::Chimney13)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d-10, o+0, 0, Corner::RightFront);
      init_extra_corner(d-7, o+1, 1, Corner::RightFront);
      init_extra_corner(d-6, o+2, 2, Corner::RightFront);
      init_extra_corner(d-5, o+3, 3, Corner::RightFront);
      init_extra_corner(d-4, o+4, 4, Corner::RightFront);
      init_extra_corner(d-3, o+5, 5, Corner::RightFront);
      init_extra_corner(d-2, o+6, 6, Corner::RightFront);
      init_extra_corner(d-1, o+7, 7, Corner::RightFront);
      init_extra_corner(d+0, o+10, 8, Corner::RightFront);
      init_extra_corner(LX-d-1+0, o+10, 9, Corner::LeftFront);
      init_extra_corner(LX-d-1+1, o+7, 10, Corner::LeftFront);
      init_extra_corner(LX-d-1+2, o+6, 11, Corner::LeftFront);
      init_extra_corner(LX-d-1+3, o+5, 12, Corner::LeftFront);
      init_extra_corner(LX-d-1+4, o+4, 13, Corner::LeftFront);
      init_extra_corner(LX-d-1+5, o+3, 14, Corner::LeftFront);
      init_extra_corner(LX-d-1+6, o+2, 15, Corner::LeftFront);
      init_extra_corner(LX-d-1+7, o+1, 16, Corner::LeftFront);
      init_extra_corner(LX-d-1+10, o+0, 17, Corner::LeftFront);
    }
    //reservoir with chimney, radius corners=16
    else if(Type==GridType::Chimney16)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d-13, o+0, 0, Corner::RightFront);
      init_extra_corner(d-10, o+1, 1, Corner::RightFront);
      init_extra_corner(d-8, o+2, 2, Corner::RightFront);
      init_extra_corner(d-7, o+3, 3, Corner::RightFront);
      init_extra_corner(d-5, o+4, 4, Corner::RightFront);
      init_extra_corner(d-4, o+5, 5, Corner::RightFront);
      init_extra_corner(d-3, o+7, 6, Corner::RightFront);
      init_extra_corner(d-2, o+8, 7, Corner::RightFront);
      init_extra_corner(d-1, o+10, 8, Corner::RightFront);
      init_extra_corner(d+0, o+13, 9, Corner::RightFront);
      init_extra_corner(LX-d-1+0, o+13, 10, Corner::LeftFront);
      init_extra_corner(LX-d-1+1, o+10, 11, Corner::LeftFront);
      init_extra_corner(LX-d-1+2, o+8, 12, Corner::LeftFront);
      init_extra_corner(LX-d-1+3, o+7, 13, Corner::LeftFront);
      init_extra_corner(LX-d-1+4, o+5, 14, Corner::LeftFront);
      init_extra_corner(LX-d-1+5, o+4, 15, Corner::LeftFront);
      init_extra_corner(LX-d-1+7, o+3, 16, Corner::LeftFront);
      init_extra_corner(LX-d-1+8, o+2, 17, Corner::LeftFront);
      init_extra_corner(LX-d-1+10, o+1, 18, Corner::LeftFront);
      init_extra_corner(LX-d-1+13, o+0, 19, Corner::LeftFront);
    }
    //reservoir with chimney exit
    else if(Type==GridType::ChimneyExit)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(LX-d-1, o+1, 0, Corner::LeftBack);
      init_extra_corner(d, o+1, 1, Corner::RightBack);
    }


    //reservoir with chimney exit, radius corners=4
    else if(Type==GridType::ChimneyExit4)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d+0, o+1-3, 0, Corner::RightBack);
      init_extra_corner(d-1, o+1-2, 1, Corner::RightBack);
      init_extra_corner(d-2, o+1-1, 2, Corner::RightBack);
      init_extra_corner(d-3, o+1+0, 3, Corner::RightBack);
      init_extra_corner(LX-d-1+3, o+1+0, 4, Corner::LeftBack);
      init_extra_corner(LX-d-1+2, o+1-1, 5, Corner::LeftBack);
      init_extra_corner(LX-d-1+1, o+1-2, 6, Corner::LeftBack);
      init_extra_corner(LX-d-1+0, o+1-3, 7, Corner::LeftBack);
    }
    //reservoir with chimney exit, radius corners=7
    else if(Type==GridType::ChimneyExit7)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d+0, o+1-5, 0, Corner::RightBack);
      init_extra_corner(d-1, o+1-3, 1, Corner::RightBack);
      init_extra_corner(d-2, o+1-2, 2, Corner::RightBack);
      init_extra_corner(d-3, o+1-1, 3, Corner::RightBack);
      init_extra_corner(d-5, o+1+0, 4, Corner::RightBack);
      init_extra_corner(LX-d-1+5, o+1+0, 5, Corner::LeftBack);
      init_extra_corner(LX-d-1+3, o+1-1, 6, Corner::LeftBack);
      init_extra_corner(LX-d-1+2, o+1-2, 7, Corner::LeftBack);
      init_extra_corner(LX-d-1+1, o+1-3, 8, Corner::LeftBack);
      init_extra_corner(LX-d-1+0, o+1-5, 9, Corner::LeftBack);
    }
    //reservoir with chimney exit, radius corners=10
    else if(Type==GridType::ChimneyExit10)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d+0, o+1-7, 0, Corner::RightBack);
      init_extra_corner(d-1, o+1-5, 1, Corner::RightBack);
      init_extra_corner(d-2, o+1-4, 2, Corner::RightBack);
      init_extra_corner(d-3, o+1-3, 3, Corner::RightBack);
      init_extra_corner(d-4, o+1-2, 4, Corner::RightBack);
      init_extra_corner(d-5, o+1-1, 5, Corner::RightBack);
      init_extra_corner(d-7, o+1+0, 6, Corner::RightBack);
      init_extra_corner(LX-d-1+7, o+1+0, 7, Corner::LeftBack);
      init_extra_corner(LX-d-1+5, o+1-1, 8, Corner::LeftBack);
      init_extra_corner(LX-d-1+4, o+1-2, 9, Corner::LeftBack);
      init_extra_corner(LX-d-1+3, o+1-3, 10, Corner::LeftBack);
      init_extra_corner(LX-d-1+2, o+1-4, 11, Corner::LeftBack);
      init_extra_corner(LX-d-1+1, o+1-5, 12, Corner::LeftBack);
      init_extra_corner(LX-d-1+0, o+1-7, 13, Corner::LeftBack);
    }
    //reservoir with chimney exit, radius corners=13
    else if(Type==GridType::ChimneyExit13)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d+0, o+1-10, 0, Corner::RightBack);
      init_extra_corner(d-1, o+1-7, 1, Corner::RightBack);
      init_extra_corner(d-2, o+1-6, 2, Corner::RightBack);
      init_extra_corner(d-3, o+1-5, 3, Corner::RightBack);
      init_extra_corner(d-4, o+1-4, 4, Corner::RightBack);
      init_extra_corner(d-5, o+1-3, 5, Corner::RightBack);
      init_extra_corner(d-6, o+1-2, 6, Corner::RightBack);
      init_extra_corner(d-7, o+1-1, 7, Corner::RightBack);
      init_extra_corner(d-10, o+1+0, 8, Corner::RightBack);
      init_extra_corner(LX-d-1+10, o+1+0, 9, Corner::LeftBack);
      init_extra_corner(LX-d-1+7, o+1-1, 10, Corner::LeftBack);
      init_extra_corner(LX-d-1+6, o+1-2, 11, Corner::LeftBack);
      init_extra_corner(LX-d-1+5, o+1-3, 12, Corner::LeftBack);
      init_extra_corner(LX-d-1+4, o+1-4, 13, Corner::LeftBack);
      init_extra_corner(LX-d-1+3, o+1-5, 14, Corner::LeftBack);
      init_extra_corner(LX-d-1+2, o+1-6, 15, Corner::LeftBack);
      init_extra_corner(LX-d-1+1, o+1-7, 16, Corner::LeftBack);
      init_extra_corner(LX-d-1+0, o+1-10, 17, Corner::LeftBack);
    }
    //reservoir with chimney exit, radius corners=16
    else if(Type==GridType::ChimneyExit16)
    {
      // define the neighbours, accounting for the periodic boundaries
      for(unsigned k=0; k<DomainSize; ++k)
      {
        const long int x = GetXPosition(k);
        const long int y = GetYPosition(k);
        for(size_t v=0; v<9; ++v)
          neighbours_ptr->at(k)[v] = GetDomainIndex(modu(x + xdir(v), LX),
                                                    modu(y + ydir(v), LY));
      }

      for (unsigned i = 0; i < GetSize(BoundaryLayer_enum::BoundaryLayer); ++i)
      {
        unsigned k=GetIndex(ExtraNode_enum::ExtraNode, i);
        for(size_t v=0; v<9; ++v)
        {
          neighbours_ptr->at(k)[v] = k;
        }
      }
      //automatically generated code
      init_extra_corner(d+0, o+1-13, 0, Corner::RightBack);
      init_extra_corner(d-1, o+1-10, 1, Corner::RightBack);
      init_extra_corner(d-2, o+1-8, 2, Corner::RightBack);
      init_extra_corner(d-3, o+1-7, 3, Corner::RightBack);
      init_extra_corner(d-4, o+1-5, 4, Corner::RightBack);
      init_extra_corner(d-5, o+1-4, 5, Corner::RightBack);
      init_extra_corner(d-7, o+1-3, 6, Corner::RightBack);
      init_extra_corner(d-8, o+1-2, 7, Corner::RightBack);
      init_extra_corner(d-10, o+1-1, 8, Corner::RightBack);
      init_extra_corner(d-13, o+1+0, 9, Corner::RightBack);
      init_extra_corner(LX-d-1+13, o+1+0, 10, Corner::LeftBack);
      init_extra_corner(LX-d-1+10, o+1-1, 11, Corner::LeftBack);
      init_extra_corner(LX-d-1+8, o+1-2, 12, Corner::LeftBack);
      init_extra_corner(LX-d-1+7, o+1-3, 13, Corner::LeftBack);
      init_extra_corner(LX-d-1+5, o+1-4, 14, Corner::LeftBack);
      init_extra_corner(LX-d-1+4, o+1-5, 15, Corner::LeftBack);
      init_extra_corner(LX-d-1+3, o+1-7, 16, Corner::LeftBack);
      init_extra_corner(LX-d-1+2, o+1-8, 17, Corner::LeftBack);
      init_extra_corner(LX-d-1+1, o+1-10, 18, Corner::LeftBack);
      init_extra_corner(LX-d-1+0, o+1-13, 19, Corner::LeftBack);
    }
  }
  // if the list already exists then simply set the ref to the correct value
  else neighbours_ptr = neighbours_pre[std::make_tuple(LX, LY, Type)].lock();
}

Grid::~Grid()
{
  // release ptr
  neighbours_ptr.reset();
  // cleanup memory
  const auto i = neighbours_pre.find(make_tuple(LX, LY, Type));
  if(i->second.expired()) neighbours_pre.erase(i);
}
