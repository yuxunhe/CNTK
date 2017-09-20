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

    struct AttributesMapping
    {
        std::unordered_map<std::wstring, std::string> map;
    };

    class Operators
    {
    public:
        static inline bool IsSupportedCNTKOP(const std::wstring& opName)
        {
            return _cntkToONNXOpName.find(opName) != _cntkToONNXOpName.end();
        }

        static inline const std::unordered_multimap<std::wstring, AttributesMapping>& CntkToONNXLookup()
        {
            return _cntkToONNXOpName;
        }

        static inline bool IsValidInputs(const std::wstring& opName, size_t index)
        {
            assert(_cntkBlockOPInvalidIndices.find(opName) != _cntkBlockOPInvalidIndices.end());

            auto invalidIndices = _cntkBlockOPInvalidIndices[opName];
            return invalidIndices.find(index) == invalidIndices.end();
        }

    private:
        static std::unordered_multimap<std::wstring, AttributesMapping> _cntkToONNXOpName;
        static std::unordered_map<std::wstring, std::set<size_t>> _cntkBlockOPInvalidIndices;
    };

}
}