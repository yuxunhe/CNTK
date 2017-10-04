//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace LotusIR
{
    class Graph;
}

namespace CNTK
{
    class CNTKToONNX
    {
    public:
        static std::unique_ptr<::LotusIR::Graph> CreateGraph(const FunctionPtr& src);
    };
}