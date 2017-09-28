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

        static inline bool HasInputIndexMap(const std::wstring& opName)
        {
            return _cntkToONNXInputIndices.find(opName) != _cntkToONNXInputIndices.end();
        }

        static inline const std::vector<int>& ToONNXInputIndexMap(const std::wstring& opName)
        {
            assert(_cntkToONNXInputIndices.find(opName) != _cntkToONNXInputIndices.end());
            return _cntkToONNXInputIndices[opName];
        }

    private:
        static std::unordered_multimap<std::wstring, AttributesMapping> _cntkToONNXOpName;
        static std::unordered_map<std::wstring, std::set<size_t>> _cntkBlockOPInvalidIndices;
        static std::unordered_map<std::wstring, std::vector<int>> _cntkToONNXInputIndices;
    };

}
}