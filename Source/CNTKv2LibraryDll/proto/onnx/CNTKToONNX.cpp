//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKToONNX.h"
#include "./core/ONNXGraph.h"

using namespace CNTK;

std::unique_ptr<CommonIR::Graph> CNTKToONNX::CreateGraph(const FunctionPtr& src)
{
    std::unique_ptr<CommonIR::Graph> graph(new CommonIR::Graph("CNTKGraph", 1));
    src;

    return graph;
}