from __future__ import division, absolute_import

import numpy as np

from unittest import main, TestCase
from plane import point_to_plane_distance, point_to_segment_distance

class TestPlane(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_point_to_plane_distance(self):
        t_abcd = np.array([2, -2, 5, 8])
        t_point = np.array([4, -4, 3])

        obs = point_to_plane_distance(t_abcd, t_point)
        np.testing.assert_almost_equal(obs, 6.78902858)

        t_point = np.array([0, 0, 3])
        obs = point_to_plane_distance(t_abcd, t_point)
        np.testing.assert_almost_equal(obs, 4.00378608)

        # distance from point (2, 8, 5) to plane x-2y-2z=1
        obs = point_to_plane_distance([1, -2, -2, -1], [2, 8, 5])
        np.testing.assert_almost_equal(obs, 8.3333333333333339)

    def test_point_to_segment_distance(self):
        xyz = np.array([[54, -59, -66],
                        [41, 41, 94],
                        [62, 71, 49],
                        [77, -5, -54],
                        [-34, 37, 60],
                        [-66, 31, 20],
                        [-64, 11, 22],
                        [10, -52, -34],
                        [-93, -86, -20],
                        [99, -40, 95]])
        abcd = np.array([0.05675653, 0.62083863, -1., 19.27817089])

        point = np.array([99, -40, 95])
        obs = point_to_segment_distance(abcd, point, xyz)
        np.testing.assert_almost_equal(obs, 80.65654409263848)

        # this point lays inside the ranges of the plane, so it should give
        # the same result as with the point_to_plane_distance calculation
        point = np.zeros(3)
        obs = point_to_segment_distance(abcd, point, xyz)
        np.testing.assert_almost_equal(obs, 16.35940725554848)

        # this point lays inside the ranges of the plane, so it should give
        # the same result as with the point_to_plane_distance calculation
        point = np.ones(3) * -1
        obs = point_to_segment_distance(abcd, point, xyz)
        np.testing.assert_almost_equal(obs, 16.63299919102415)

if __name__ == '__main__':
    main()

