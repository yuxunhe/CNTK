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
    img = np.reshape(np.arange(float(np.prod(img_shape)), dtype=np.float32), img_shape)

    x = C.input_variable(img.shape)
    root_node = C.layers.Dense(5, activation=C.softmax)(x)
    
    filename = os.path.join(str(tmpdir), R'dense_layer.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    x_ = loaded_node.arguments[0];
    assert np.allclose(loaded_node.eval({x_:img}), root_node.eval({x:img}))

def test_conv_layer(tmpdir):
    image_width = 28
    image_height = 28
    input_dim_model = (1, image_width, image_height)
    input_dim = image_width * image_height
    num_output_classes = 10

    features = C.input_variable(input_dim_model)

    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        convModel = features
        convModel = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=8,
                                   strides=(2,2),
                                   pad=True, name='first_conv')(convModel)
        convModel = C.layers.Convolution2D(filter_shape=(5,5),
                                   num_filters=16,
                                   strides=(2,2),
                                   pad=True, name='second_conv')(convModel)
        convModel = C.layers.Dense(num_output_classes, activation=None, name='classify')(convModel)

    img_data = np.random.rand(image_width * image_height).reshape(1, 1, image_width, image_height).astype('float32')

    filename = os.path.join(str(tmpdir), R'conv_layer.onnx')

    convModel.save(filename, format=C.ModelFormat.ONNX)
    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)

    assert np.allclose(loaded_node.eval(img_data), convModel.eval(img_data))
