# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk as C

def test_load_save_constant(tmpdir):
    # import pdb;pdb.set_trace()
    c = C.constant(value=[1,3])
    root_node = c * 5

    result = root_node.eval()
    expected = [[[[5,15]]]]
    assert np.allclose(result, expected)

    filename = os.path.join(str(tmpdir), R'c_plus_c.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    loaded_result = loaded_node.eval()
    assert np.allclose(loaded_result, expected)

def test_dense_layer(tmpdir):
    img_shape = (1, 5, 5)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = C.layers.Dense(5, activation=C.softmax)(x)
    
    filename = os.path.join(str(tmpdir), R'dense_layer.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    x_ = loaded_node.arguments[0];
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

def test_conv_model(tmpdir):
    def create_model(input):
        with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
            model = C.layers.Sequential([
                C.layers.For(range(3), lambda i: [
                    C.layers.Convolution((5,5), [32,32,64][i], pad=True),
                    C.layers.MaxPooling((3,3), strides=(2,2))
                    ]),
                C.layers.Dense(64),
                C.layers.Dense(10, activation=None)
            ])

        return model(input)

    img_shape = (3, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'conv_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    x_ = loaded_node.arguments[0];
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

def test_vgg9_model(tmpdir):
    def create_model(input):
        with C.layers.default_options(activation=C.relu, init=C.glorot_uniform()):
            model = C.layers.Sequential([
                C.layers.For(range(3), lambda i: [
                    C.layers.Convolution((3,3), [64,96,128][i], pad=True),
                    C.layers.Convolution((3,3), [64,96,128][i], pad=True),
                    C.layers.MaxPooling((3,3), strides=(2,2))
                ]),
                C.layers.For(range(2), lambda : [
                    C.layers.Dense(1024)
                ]),
                C.layers.Dense(10, activation=None)
            ])
        
        return model(input)

    img_shape = (3, 32, 32)
    img = np.asarray(np.random.uniform(-1, 1, img_shape), dtype=np.float32)

    x = C.input_variable(img.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'vgg9_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    x_ = loaded_node.arguments[0];
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

def test_conv3d_model(tmpdir):
    def create_model(input):
        with C.default_options (activation=C.relu):
            model = C.layers.Sequential([
                    C.layers.Convolution3D((3,3,3), 64, pad=True),
                    C.layers.MaxPooling((1,2,2), (1,2,2)),
                    C.layers.For(range(3), lambda i: [
                        C.layers.Convolution3D((3,3,3), [96, 128, 128][i], pad=True),
                        C.layers.Convolution3D((3,3,3), [96, 128, 128][i], pad=True),
                        C.layers.MaxPooling((2,2,2), (2,2,2))
                    ]),
                    C.layers.For(range(2), lambda : [
                        C.layers.Dense(1024), 
                        # C.layers.Dropout(0.5)
                    ]),
                C.layers.Dense(100, activation=None)
            ])

        return model(input)

    video_shape = (3, 20, 32, 32)
    video = np.asarray(np.random.uniform(-1, 1, video_shape), dtype=np.float32)

    x = C.input_variable(video.shape)
    root_node = create_model(x)

    filename = os.path.join(str(tmpdir), R'conv3d_model.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    x_ = loaded_node.arguments[0];

    assert np.allclose(loaded_node.eval({x_:video}), root_node.eval({x:video}))