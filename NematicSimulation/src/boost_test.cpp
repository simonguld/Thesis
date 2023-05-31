
#include <iostream>
#include <iterator>
#include <algorithm>
#include <ostream>

#include "boost/lambda/lambda.hpp"


int main()
{
    std::cout << "Hej";

    #if 0
    {
        using namespace boost::lambda;
        typedef std::istream_iterator<int> in;

        std::for_each(
            in(std::cin), in(), std::cout << (_1 * 3) << " " );
    }
    #endif
    return 0;
}
