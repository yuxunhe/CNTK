#ifndef CORE_GRAPH_UTILS_H
#define CORE_GRAPH_UTILS_H

#include <xstring>

#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)
#include "../protobuf/Type.pb.h"
#pragma warning(pop)

namespace CommonIR
{
    namespace Utils
    {
        class OpUtils
        {
        public:

            static std::string ToString(const TypeProto& p_type);
        };

    }
}

#endif // ! COMMONIR_UTILS_H
