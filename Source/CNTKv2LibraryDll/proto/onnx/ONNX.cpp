//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNX.h"
#include "CNTKToONNX.h"
#include "./core/graph.h"
#include "Utils.h"

#include <iostream>

#include "ONNXToCNTK.h"

namespace CNTK
{
    static void PrintGraph(FunctionPtr function, int spaces, bool useName = false)
    {
        if (function->Inputs().size() == 0)
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
            return;
        }

        for(auto input: function->Inputs())
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
                "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString()) << std::endl;
        }

        for(auto input: function->Inputs())
        {
            if (input.Owner() != NULL)
            {
                FunctionPtr f = input.Owner();
                PrintGraph(f, spaces + 4, useName);
            }
        }
    }

void ONNX::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    PrintGraph(src, 0, true);
    PrintGraph(src, 0, false);
    std::unique_ptr<LotusIR::Graph> graph = CNTKToONNX::CreateGraph(src);

    LotusIR::Graph::Save(graph->ToGraphProto(), filepath);
}

FunctionPtr ONNX::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice)
{
    LotusIR::GraphProto grapu;
    bool loadStatus = LotusIR::Graph::Load(filepath, &grapu);
    if (!loadStatus)
    {
        return nullptr;
    }

    std::unique_ptr<LotusIR::Graph> graph(new LotusIR::Graph(grapu));
    graph->Resolve();

    FunctionPtr cntkFunction = ONNXToCNTK::CreateGraph(graph, computeDevice);
    return cntkFunction;
}

}