//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

#include <iostream>
#include <cstdio>
#include <locale>
#include <codecvt>

using namespace CNTK;
using namespace std::placeholders;

void TrainCifarResnet();
void TrainLSTMSequenceClassifier();
void MNISTClassifierTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifier();
void TestFrameMode();
void TestDistributedCheckpointing();

// #include "../proto/onnx/onnx.h";
// #include "../proto/onnx/CNTKToONNX.h"
// #include "../proto/onnx/ONNXToCNTK.h"
// #include "../proto/onnx/core/graph.h"

std::string ToString(const std::wstring& wstring)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.to_bytes(wstring);
}

static void PrintGraph(const FunctionPtr& function, int spaces, bool useName = false)
{
    std::vector<Variable> inputs = function->Inputs();
    size_t count = inputs.size();
    if (count == 0)
    {
        std::cout << std::string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
        return;
    }

    for (Variable input : inputs)
    {
        std::cout << std::string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
            "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString()) << std::endl;
    }

    for (Variable input : inputs)
    {
        if (input.Owner() != NULL)
        {
            FunctionPtr f = input.Owner();
            PrintGraph(f, spaces + 4);
        }
    }
}

void RunLotus(DeviceDescriptor device)
{
    {
        // vgg model load failed
        //const std::wstring vgg16 = L"E:/LiqunWA/CNTK/ONNX/vgg16/vgg16/graph.pb";
        //FunctionPtr cntkModel = Function::Load(vgg16, device, ModelFormat::ONNX);
    }

    // CNTK function saved as ONNX. 
    // Loaded but graph is not the same as in memory graph. See ONNX.Save commented code.
    // FunctionPtr cntkModel = Function::Load(savedONNXModelFile, device, ModelFormat::ONNX);

    // following are experiment code
    {
        const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolutionONNX.model";
        const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolution.model";
        const std::wstring cntkModelFile2 = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolution.model";

        //const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTMLP.model";
        //const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTMLPONNX.model";

        //const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/LogisticRegression.model";
        //const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/LogisticRegressionONNX.model";

        FunctionPtr cntkModel = Function::Load(cntkModelFile, device, ModelFormat::CNTKv2);

        cntkModel->Save(savedONNXModelFile, ModelFormat::ONNX);
        // TODO PrintGraph does not work for some reason
        // PrintGraph(cntkModel->RootFunction(), 0);

        // failed here because graph save/load operation is not idempotent
        FunctionPtr cntkModelFromONNX = Function::Load(savedONNXModelFile, device, ModelFormat::ONNX);
        // PrintGraph(cntkModelFromONNX, 0);
        cntkModelFromONNX->Save(cntkModelFile2, ModelFormat::CNTKv2);
    }
}

int main(int argc, char *argv[])
{
#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

    RunLotus(DeviceDescriptor::CPUDevice());

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

#ifndef CPUONLY
    fprintf(stderr, "Run tests using GPU build.\n");
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    if (argc > 2)
    {
        if (argc == 3 && !std::string(argv[1]).compare("Distribution")) {
            {
                auto communicator = MPICommunicator();
                std::string logFilename = argv[2] + std::to_string(communicator->CurrentWorker().m_globalRank);
                auto result = freopen(logFilename.c_str(), "w", stdout);
                if (result == nullptr)
                {
                    fprintf(stderr, "Could not redirect stdout.\n");
                    return -1;
                }
            }

            TestFrameMode();

            TestDistributedCheckpointing();

            std::string testsPassedMsg = "\nCNTKv2Library-Distribution tests: Passed\n";

            printf("%s", testsPassedMsg.c_str());

            fflush(stdout);
            DistributedCommunicator::Finalize();
            fclose(stdout);
            return 0;
        }
        else
        {
            fprintf(stderr, "Wrong number of arguments.\n");
            return -1;
        }
    }

    std::string testName(argv[1]);

    if (!testName.compare("CifarResNet"))
    {
        if (ShouldRunOnGpu())
        {
            fprintf(stderr, "Run test on a GPU device.\n");
            TrainCifarResnet();
        }
        
        if (ShouldRunOnCpu())
        {
            fprintf(stderr, "Cannot run TrainCifarResnet test on a CPU device.\n");
        }
    }
    else if (!testName.compare("LSTMSequenceClassifier"))
    {
        TrainLSTMSequenceClassifier();
    }
    else if (!testName.compare("MNISTClassifier"))
    {
        MNISTClassifierTests();
    }
    else if (!testName.compare("SequenceToSequence"))
    {
        TrainSequenceToSequenceTranslator();
    }
    else if (!testName.compare("TruncatedLSTMAcousticModel"))
    {
        TrainTruncatedLSTMAcousticModelClassifier();
    }
    else
    {
        fprintf(stderr, "End to end test not found.\n");
        return -1;
    }

    std::string testsPassedMsg = "\nCNTKv2Library-" + testName + " tests: Passed\n";

    fprintf(stderr, "%s", testsPassedMsg.c_str());
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif
}
