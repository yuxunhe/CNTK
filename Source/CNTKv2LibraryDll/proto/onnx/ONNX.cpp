//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNX.h"
#include "CNTKToONNX.h"
#include "./core/ONNXGraph.h"
#include "Utils.h"

namespace CNTK
{

void ONNX::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    std::unique_ptr<CommonIR::Graph> graph = CNTKToONNX::CreateGraph(src);
    filepath;
}

}