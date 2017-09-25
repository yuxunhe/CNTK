//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNX.h"
#include "CNTKToONNX.h"
#include "./core/graph.h"
#include "Utils.h"

namespace CNTK
{

void ONNX::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    std::unique_ptr<LotusIR::Graph> graph = CNTKToONNX::CreateGraph(src);
    LotusIR::Graph::Save(graph->ToGraphProto(), ToString(filepath));
}

FunctionPtr ONNX::Load(const std::wstring& filepath)
{
    filepath;
    return nullptr;
}

}