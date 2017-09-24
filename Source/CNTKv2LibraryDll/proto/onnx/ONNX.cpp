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
    static void PrintGraph(FunctionPtr function, int spaces)
    {
        if (function->Inputs().size() == 0)
        {
            cout << string(spaces, '.') + "(" + ToString(function->Name()) + ")" + ToString(function->AsString()) << std::endl;
            return;
        }

        for(auto input: function->Inputs())
        {
            cout << string(spaces, '.') + "(" + ToString(function->Name()) + ")" + "->" +
                "(" + ToString(input.Name()) + ")" + ToString(input.AsString()) << std::endl;
        }

        for(auto input: function->Inputs())
        {
            if (input.Owner() != NULL)
            {
                FunctionPtr f = input.Owner();
                PrintGraph(f, spaces + 4);
            }
        }
    }

void ONNX::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    PrintGraph(src, 0);
    std::unique_ptr<LotusIR::Graph> graph = CNTKToONNX::CreateGraph(src);

    // Liqun: experiment Create CNTK function from ONNX graph
    graph->Resolve();
    FunctionPtr cntkFunction = ONNXToCNTK::CreateGraph(graph);

    PrintGraph(cntkFunction, 0);

    filepath;
}

FunctionPtr ONNX::Load(const std::wstring& filepath)
{
    filepath;
    return nullptr;
}

}