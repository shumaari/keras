# from file: keras\src\ops\numpy_test.py
# !pytest -c /content/keras/pytest.ini keras/src/ops/numpy_test.py::NumpyTwoInputOpsCorrectnessTest::test_digitize keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_bfloat16 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_bool keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_float16 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_float32 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_float64 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int16 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int32 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int64 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int8 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_none keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_uint16 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_uint32 keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_uint8
# /content/keras/keras/src/ops/numpy_test.py
# ============== 14 failed, 1623 passed, 4189 skipped, 56 warnings in 6
# keras/src/ops/numpy_test.py::NumpyTwoInputOpsCorrectnessTest::test_digitize
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_bfloat16 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_bool 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_float16 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_float32 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_float64 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int16 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int32 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int64 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_int8 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_none 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_uint16 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_uint32 
# keras/src/ops/numpy_test.py::NumpyDtypeTest::test_digitize_uint8

# these two lines were removed from exclude tests file
# so worry about these two tests only
# NumpyTwoInputOpsCorrectnessTest::test_digitize
# NumpyDtypeTest::test_digitize

import contextlib
import functools
import itertools
import math
import warnings

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import is_int_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import numpy as knp
from keras.src.testing.test_utils import named_product



class NumpyTwoInputOpsStaticShapeTest(testing.TestCase):
    def test_digitize(self):
        x = KerasTensor((2, 3))
        bins = KerasTensor((3,))
        self.assertEqual(knp.digitize(x, bins).shape, (2, 3))
        self.assertTrue(knp.digitize(x, bins).dtype == "int32")

        with self.assertRaises(ValueError):
            x = KerasTensor((2, 3))
            bins = KerasTensor((2, 3, 4))
            knp.digitize(x, bins)


class NumpyTwoInputOpsCorrectnessTest(testing.TestCase):
    def test_digitize(self):
        x = np.array([0.0, 1.0, 3.0, 1.6])
        bins = np.array([0.0, 3.0, 4.5, 7.0])
        self.assertAllClose(knp.digitize(x, bins), np.digitize(x, bins))
        self.assertAllClose(knp.Digitize()(x, bins), np.digitize(x, bins))
        self.assertTrue(
            standardize_dtype(knp.digitize(x, bins).dtype) == "int32"
        )
        self.assertTrue(
            standardize_dtype(knp.Digitize()(x, bins).dtype) == "int32"
        )

        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        self.assertAllClose(knp.digitize(x, bins), np.digitize(x, bins))
        self.assertAllClose(knp.Digitize()(x, bins), np.digitize(x, bins))
        self.assertTrue(
            standardize_dtype(knp.digitize(x, bins).dtype) == "int32"
        )
        self.assertTrue(
            standardize_dtype(knp.Digitize()(x, bins).dtype) == "int32"
        )

        x = np.array([1, 4, 10, 15])
        bins = np.array([4, 10, 14, 15])
        self.assertAllClose(knp.digitize(x, bins), np.digitize(x, bins))
        self.assertAllClose(knp.Digitize()(x, bins), np.digitize(x, bins))
        self.assertTrue(
            standardize_dtype(knp.digitize(x, bins).dtype) == "int32"
        )
        self.assertTrue(
            standardize_dtype(knp.Digitize()(x, bins).dtype) == "int32"
        )

#sparse tensors and testing??

class NumpyDtypeTest(testing.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    # TODO: Using uint64 will lead to weak type promotion (`float`),
    # resulting in different behavior between JAX and Keras. Currently, we
    # are skipping the test for uint64
    ALL_DTYPES = [
        x
        for x in dtypes.ALLOWED_DTYPES
        if x not in ["string", "uint64", "complex64", "complex128"]
    ] + [None]
    INT_DTYPES = [x for x in dtypes.INT_TYPES if x != "uint64"]
    FLOAT_DTYPES = dtypes.FLOAT_TYPES

    if backend.backend() == "torch":
        # TODO: torch doesn't support uint16, uint32 and uint64
        ALL_DTYPES = [
            x for x in ALL_DTYPES if x not in ["uint16", "uint32", "uint64"]
        ]
        INT_DTYPES = [
            x for x in INT_DTYPES if x not in ["uint16", "uint32", "uint64"]
        ]
    # Remove float8 dtypes for the following tests
    ALL_DTYPES = [x for x in ALL_DTYPES if x not in dtypes.FLOAT8_TYPES]

    def setUp(self):
        from jax.experimental import enable_x64

        self.jax_enable_x64 = enable_x64()
        self.jax_enable_x64.__enter__()
        return super().setUp()

    def tearDown(self):
        self.jax_enable_x64.__exit__(None, None, None)
        return super().tearDown()


    @parameterized.named_parameters(named_product(dtype=ALL_DTYPES))
    def test_digitize(self, dtype):
        import jax.numpy as jnp

        x = knp.ones((1,), dtype=dtype)
        bins = knp.ones((1,), dtype=dtype)
        x_jax = jnp.ones((1,), dtype=dtype)
        x_bins = jnp.ones((1,), dtype=dtype)
        expected_dtype = standardize_dtype(jnp.digitize(x_jax, x_bins).dtype)

        self.assertEqual(
            standardize_dtype(knp.digitize(x, bins).dtype), expected_dtype
        )
        self.assertEqual(
            standardize_dtype(knp.Digitize().symbolic_call(x, bins).dtype),
            expected_dtype,
        )
