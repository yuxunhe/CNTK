#ifndef CORE_GRAPH_UTILS_H
#define CORE_GRAPH_UTILS_H

#include <xstring>

#include "core/protobuf/Type.pb.h"

namespace LotusIR
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

#endif // ! CORE_GRAPH_UTILS_H
