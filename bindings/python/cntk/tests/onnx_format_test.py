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
    assert np.allclose(loaded_result.eval({x:img}), root_node.eval({x:img}))
