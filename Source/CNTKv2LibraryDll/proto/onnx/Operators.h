//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

#include <set>

namespace LotusIR
{
    class Graph;
}

namespace CNTK
{
namespace ONNX
{

    class Operators
    {
    public:
        static inline bool IsSupportedCNTKOP(const std::wstring& opName)
        {
            return _cntkToONNXOpName.find(opName) != _cntkToONNXOpName.end();
        }

        static inline const std::unordered_multimap<std::wstring, std::string>& CntkToONNXLookup()
        {
            return _cntkToONNXOpName;
        }

    private:
        static std::unordered_multimap<std::wstring, std::string> _cntkToONNXOpName;
    };

}
}