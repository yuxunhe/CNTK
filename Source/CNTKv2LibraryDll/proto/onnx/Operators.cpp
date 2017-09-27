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
            { L"poolingWindowShape", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
        } } },
        { L"Pooling",  { {
            { L"Pooling",  "MaxPool" },
            { L"poolingWindowShape", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
        } } },
        { L"Convolution", { {
            { L"Convolution", "Conv" },
            // { L"", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
            { L"dilation", "dilations" },
            // { L"", "group" },
        } } },
        { L"ConvolutionTranspose", { {
            { L"ConvolutionTranspose", "ConvTranspose" },
            // { L"", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
            { L"dilation", "dilations" },
        } } },
        { L"GlobalMaxPooling", { {
            { L"GlobalMaxPooling", "GlobalAveragePool" },
        } } },
        { L"GlobalAveragePooling", { {
            { L"GlobalAveragePooling", "GlobalMaxPool" },
        } } },
        { L"BatchNormalization", { {
            { L"BatchNormalization", "BatchNormalization" },
            { L"spatial", "spatial" },
            // { L"", "is_test" },
            { L"epsilon", "epsilon" },
            // { L"", "momentum" },
        } } },
        { L"Dropout", { {
            { L"Dropout", "Dropout" },
            { L"dropoutRate", "ratio" },
            // { L"", "is_test" },
        } } },
        { L"Flatten",{ {
            { L"Flatten", "Flatten" },
        } } },

        // From Generator
        { L"UniformRandom", { {
            { L"UniformRandom", "RandomUniform" },
            // { L"", "low" },
            // { L"", "high" },
            { L"rngSeed", "seed" },
            { L"newShape", "shape" },
        } } },
        { L"NormalRandom", { {
            { L"NormalRandom", "RandomNormal" },
            // { L"", "mean" },
            // { L"", "scale" },
            { L"rngSeed", "seed" },
            { L"newShape", "shape" },
        } } },
        { L"UniformRandomLike", { {
            { L"UniformRandomLike", "RandomUniformLike" },
            // { L"", "low" },
            // { L"", "high" },
            { L"rngSeed", "seed" },
        } } },
        { L"NormalRandomLike", { {
            { L"NormalRandomLike", "RandomNormalLike" },
            // { L"", "mean" },
            // { L"", "scale" },
            { L"rngSeed", "seed" },
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
            { L"alpha", "alpha" },
        } } },
        { L"SELU", { {
            { L"SELU", "Selu" },
            { L"alpha", "alpha" },
            { L"gamma", "gamma" },
        } } },
        { L"ELU", { {
            { L"ELU", "Elu" },
            // { L"", "alpha" },
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
            // { L"", "exponent" },
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
            { L"axis", "axis" },
        } } },

        // From reduction
        { L"ReduceMax", { {
            { L"ReduceMax", "ReduceMax" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceMin", { {
            { L"ReduceMin", "ReduceMin" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceSum", { {
            { L"ReduceSum", "ReduceSum" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceMean", { {
            { L"ReduceMean", "ReduceMean" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceProd", { {
            { L"ReduceProd", "ReduceProd" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceLogSum", { {
            { L"ReduceLogSum", "ReduceLogSumExp" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"Argmax", { {
            { L"Argmax", "ArgMax" },
            { L"axis", "axes" },
            // { L"", "keepdims" },
        } } },
        { L"Argmin", { {
            { L"Argmin", "ArgMin" },
            { L"axis", "axes" },
            // { L"", "keepdims" },
        } } },

        // From tensor
        // { L"", "Cast" },
        { L"Reshape", { {
            { L"Reshape", "Reshape" },
            { L"newShape", "shape" },
        } } },
        { L"Splice", { {
            { L"Splice", "Concat" },
            { L"axis", "axis" },
        } } },
        // { L"", "Split" },
        { L"Slice", { {
            { L"Slice", "Slice" },
            { L"beginIndexVec", "starts" },
            { L"endIndexVec", "ends" },
        } } },
        { L"Transpose", { {
            { L"Transpose", "Transpose" },
            { L"axisVec", "perm" },
        } } },
        { L"GatherOp", { {
            { L"GatherOp", "Gather" },
        } } },
        // { L"", "Squeeze" },
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