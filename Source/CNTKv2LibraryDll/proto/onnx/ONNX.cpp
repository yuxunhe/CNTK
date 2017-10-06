//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNX.h"
#include "CNTKToONNX.h"
#include "proto/onnx/core/model.h"
#include "proto/onnx/core/graph.h"
#include "Utils.h"

#include <iostream>

#include "ONNXToCNTK.h"

using namespace CNTK;

namespace CNTK
{
    static void PrintGraph(FunctionPtr function, int spaces, bool useName = false)
    {
        if (function->Inputs().size() == 0)
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
            return;
        }

        for (auto input : function->Inputs())
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
                "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString()) << std::endl;
        }

        for (auto input : function->Inputs())
        {
            if (input.Owner() != NULL)
            {
                FunctionPtr f = input.Owner();
                PrintGraph(f, spaces + 4, useName);
            }
        }
    }
}

void ONNX::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    auto model = CNTKToONNX::CreateModel(src);
#ifdef _WIN32
    LotusIR::Model::Save(*model, filepath);
#else
    LotusIR::Model::Save(*model, ToString(filepath));
#endif
}

FunctionPtr ONNX::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice)
{
    LotusIR::ModelProto modelProto;

#ifdef _WIN32
    bool loadStatus = LotusIR::Model::Load(filepath, &modelProto);
#else
    bool loadStatus = LotusIR::Model::Load(ToString(filepath), &modelProto);
#endif
    if (!loadStatus)
        LogicError("Failed to load the model.");

    LotusIR::Model model(modelProto);
    auto status = model.MainGraph()->Resolve();
    if (!status.Ok())
        LogicError("%s", status.ErrorMsg().c_str());

    FunctionPtr cntkFunction = ONNXToCNTK::CreateGraph(model.MainGraph(), computeDevice);    
    PrintGraph(cntkFunction->RootFunction(), 0, false);
    return cntkFunction;
}
