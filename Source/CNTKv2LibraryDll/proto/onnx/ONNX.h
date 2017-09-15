//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"


namespace CNTK
{
    class ONNX
    {
    public:
        static void Save(const FunctionPtr& src, const std::wstring& filepath);
    };
}