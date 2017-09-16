//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNXToCNTK.h"
#include "./core/graph.h"
#include "Utils.h"

namespace CNTK
{

FunctionPtr ONNXToCNTK::CreateGraph(const std::unique_ptr<LotusIR::Graph>& src)
{
    src;
    return nullptr;
}

}