//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace ONNXIR
{
    class Graph;
}

namespace CNTK
{
    class ONNXToCNTK
    {
    public:
        static FunctionPtr CreateGraph(ONNXIR::Graph* src, const DeviceDescriptor& computeDevice);
    };
}