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
    //
    // Support ONNX OPs from https://github.com/onnx/onnx/tree/master/onnx/defs
    //
    // The format of the below structure is simply a key which is CNTK OpName and a corresponding
    // lookup table, the  corrsponding lookup table map the OpName and all its attributes from CNTK
    // to ONNX.
    //
    // Eventually, it would be good to change CNTK OpName to match ONNX in order to avoid the need 
    // of the below table.
    //
    std::unordered_multimap<std::wstring, AttributesMapping> Operators::_cntkToONNXOpName = {
        // From nn
        { L"Pooling", { {
            { L"Pooling", "AveragePool" },
        } } },
        { L"Pooling",  { {
            { L"Pooling",  "MaxPool" },
        } } },
        { L"Convolution", { {
            { L"Convolution", "Conv" },
        } } },
        { L"ConvolutionTranspose", { {
            { L"ConvolutionTranspose", "ConvTranspose" },
        } } },
        { L"GlobalMaxPooling", { {
            { L"GlobalMaxPooling", "GlobalAveragePool" },
        } } },
        { L"GlobalAveragePooling", { {
            { L"GlobalAveragePooling", "GlobalMaxPool" },
        } } },
        { L"BatchNormalization", { {
            { L"BatchNormalization", "BatchNormalization" },
        } } },
        { L"Dropout", { {
            { L"Dropout", "Dropout" },
        } } },
        // { L"", "Flatten" },

        // From Generator
        { L"RandomUniform", { {
            { L"RandomUniform", "RandomUniform" },
        } } },
        { L"RandomNormal", { {
            { L"RandomNormal", "RandomNormal" },
        } } },
        { L"RandomUniformLike", { {
            { L"RandomUniformLike", "RandomUniformLike" },
        } } },
        { L"RandomNormalLike", { {
            { L"RandomNormalLike", "RandomNormalLike" },
        } } },

        // From Math 
        { L"Plus", { {
            { L"Plus", "Add" },
        } } },
        { L"Minus", { {
            { L"Minus", "Sub" },
        } } },
        { L"ElementTimes", { {
            { L"ElementTimes", "Mul" },
        } } },
        { L"ElementDivide", { {
            { L"ElementDivide", "Div" },
        } } },
        { L"Negate", { {
            { L"Negate", "Neg" },
        } } },
        { L"Abs", { {
            { L"Abs", "Abs" },
        } } },
        { L"Reciprocal", { {
            { L"Reciprocal", "Reciprocal" },
        } } },
        { L"Floor", { {
            { L"Floor", "Floor" },
        } } },
        { L"Ceil", { {
            { L"Ceil", "Ceil" },
        } } },
        { L"Sqrt", { {
            { L"Sqrt", "Sqrt" },
        } } },
        { L"ReLU", { {
            { L"ReLU", "Relu" },
        } } },
        { L"LeakyReLU", { {
            { L"LeakyReLU", "LeakyRelu" },
        } } },
        { L"SELU", { {
            { L"SELU", "Selu" },
        } } },
        { L"ELU", { {
            { L"ELU", "Elu" },
        } } },
        { L"Exp", { {
            { L"Exp", "Exp" },
        } } },
        { L"Log", { {
            { L"Log", "Log" },
        } } },
        { L"Tanh", { {
            { L"Tanh", "Tanh" },
        } } },
        { L"Pow", { {
            { L"Pow", "Pow" },
        } } },
        { L"Times", { {
            { L"Times", "Dot" },
        } } },
        { L"PReLU", { {
            { L"PReLU", "PRelu" },
        } } },
        { L"Sigmoid", { {
            { L"Sigmoid", "Sigmoid" },
        } } },
        { L"ElementMax", { {
            { L"ElementMax", "Max" },
        } } },
        { L"ElementMax", { {
            { L"ElementMax", "Min" },
        } } },
        // { L"", "Sum" },
        { L"Softmax", { {
            { L"Softmax", "Softmax" },
        } } },

        // From reduction
        { L"ReduceMax", { {
            { L"ReduceMax", "ReduceMax" },
        } } },
        { L"ReduceMin", { {
            { L"ReduceMin", "ReduceMin" },
        } } },
        { L"ReduceSum", { {
            { L"ReduceSum", "ReduceSum" },
        } } },
        { L"ReduceMean", { {
            { L"ReduceMean", "ReduceMean" },
        } } },
        { L"ReduceProd", { {
            { L"ReduceProd", "ReduceProd" },
        } } },
        { L"ReduceLogSum", { {
            { L"ReduceLogSum", "ReduceLogSumExp" },
        } } },
        { L"Argmax", { {
            { L"Argmax", "ArgMax" },
        } } },
        { L"Argmin", { {
            { L"Argmin", "ArgMin" },
        } } },

        // From tensor
        // { L"", "Cast" },
        { L"Reshape", { {
            { L"Reshape", "Reshape" },
        } } },
        { L"Splice", { {
            { L"Splice", "Concat" },
        } } },
        // { L"", "Split" },
        { L"Slice", { {
            { L"Slice", "Slice" },
        } } },
        { L"Transpose", { {
            { L"Transpose", "Transpose" },
        } } },
        { L"GatherOp", { {
            { L"GatherOp", "Gather" },
        } } },
    };

    std::unordered_map<std::wstring, std::set<size_t>> Operators::_cntkBlockOPInvalidIndices = {
        { L"LeakyReLU", {0, 1} },
        { L"SELU", {1} },
        { L"PReLU", {1} },
        { L"ElementMax", {} },
        { L"ElementMax", {} },
    };
}
}