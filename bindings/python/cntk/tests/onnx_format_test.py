# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import cntk as C


def test_load_save_constant(tmpdir):
    c = C.constant(value=[1,3])
    root_node = c * 5

    result = root_node.eval()
    expected = [[[[5,15]]]]
    assert np.allclose(result, expected)

    filename = str(tmpdir / 'c_plus_c.onnx')
    root_node.save(filename, format=C.ModelFormat.ONNX)

    loaded_node = C.Function.load(filename, format=C.ModelFormat.ONNX)
    loaded_result = loaded_node.eval()
    assert np.allclose(loaded_result, expected)
