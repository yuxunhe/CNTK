//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Operators.h"
#include "./core/graph.h"
#include "Utils.h"

namespace CNTK
{
namespace ONNX
{

    std::unordered_multimap<std::wstring, std::string> Operators::_cntkToONNXOpName = {
        // From nn
        { L"Pooling", "AveragePool" },
        { L"Pooling",  "MaxPool"},
        { L"Convolution", "Conv" }, 
        { L"ConvolutionTranspose", "ConvTranspose" },
        { L"Pooling", "GlobalAveragePool" },
        { L"Pooling", "GlobalMaxPool" },
        { L"BatchNormalization", "BatchNormalization" },
        { L"Dropout", "Dropout" },
        // { L"", "Flatten" },

        // From Generator
        { L"RandomUniform", "RandomUniform" },
        { L"RandomNormal", "RandomNormal" },
        { L"RandomUniformLike", "RandomUniformLike" },
        { L"RandomNormalLike", "RandomNormalLike" },

        // From Math 
        { L"Plus", "Add" },
        { L"Minus", "Sub" },
        { L"ElementTimes", "Mul" },
        { L"ElementDivide", "Div" },
        { L"Negate", "Neg" },
        { L"Abs", "Abs" },
        { L"Reciprocal", "Reciprocal" },
        { L"Floor", "Floor" },
        { L"Ceil", "Ceil" },
        { L"Sqrt", "Sqrt" },
        { L"ReLU", "Relu" },
        { L"LeakyReLU", "LeakyRelu" },
        { L"SELU", "Selu" },
        { L"ELU", "Elu" },
        { L"Exp", "Exp" }, 
        { L"Log", "Log" },
        { L"Tanh", "Tanh" },
        { L"", "Pow" },
        { L"Times", "Dot" },
        { L"PReLU", "PRelu" },
        { L"Sigmoid", "Sigmoid" },
        { L"", "Max" },
        { L"", "Min" },
        { L"", "Sum" },
        { L"Softmax", "Softmax" },

        // From reduction
        { L"ReduceMax", "ReduceMax" },
        { L"ReduceMin", "ReduceMin" },
        { L"ReduceSum", "ReduceSum" },
        { L"ReduceMean", "ReduceMean" },
        { L"ReduceProd", "ReduceProd" },
        { L"ReduceLogSum", "ReduceLogSumExp" },
        { L"Argmax", "ArgMax" },
        { L"Argmin", "ArgMin" },

        // From tensor
        // { L"", "Cast" },
        { L"Reshape", "Reshape" },
        { L"Splice", "Concat" },
        // { L"", "Split" },
        { L"Slice", "Slice" },
        { L"Transpose", "Transpose" },
        { L"GatherOp", "Gather" },
    };
}
}