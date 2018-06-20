import numpy as np
import numpy.testing as npt

import modules.images as im


def test_coordinate_conversion():

    x_res, y_res = 640, 480

    x_to_z = 1.11146664619446
    y_to_z = 0.833599984645844

    point_real = np.array([10, 5, 3])

    point_proj = im.real_to_proj(point_real, x_res, y_res, x_to_z, y_to_z)

    point_real_new = im.proj_to_real(point_proj, x_res, y_res, x_to_z, y_to_z)

    npt.assert_allclose(point_real, point_real_new)

    for _ in range(10):
        x_res, y_res = np.random.randint(1, 1000), np.random.randint(1, 1000)
        x_to_z, y_to_z = np.random.rand(), np.random.rand()

        point_real = np.random.randint(-100, 100, size=(3,))

        point_proj = im.real_to_proj(point_real, x_res, y_res, x_to_z, y_to_z)

        point_real_new = im.proj_to_real(point_proj, x_res, y_res,
                                         x_to_z, y_to_z)

        npt.assert_allclose(point_real, point_real_new)
