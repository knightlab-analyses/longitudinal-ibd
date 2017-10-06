from __future__ import division

__credits__ = "Robert Kern"
__url__ = "https://groups.google.com/forum/#!topic/comp.lang.python/0JiqYeo0qu4"

import numpy as np
import pandas as pd

from emperor.parse import parse_coords
from scipy.spatial.distance import euclidean

def point_to_plane_distance(abcd, point):
    """
    Calculates the euclidean distance from a point to a plane

    Parameters
    ----------
    abcd : array-like
        The four coefficients of an equation that defines a
        plane of the form a*x + b*y + c*z + d = 0
    point : array-like
        The values for x, y and z for the point that you want
        to calculate the distance to.

    Returns
    -------
    float
        Distance from the point to the plane as defined in the
        References listed below.

    References
    ----------
    .. [1] Math Insight, Distance from point to plane,
           http://mathinsight.org/distance_point_plane
    .. [2] Ballantine, J. P., Essentials of Engineering Mathematics
           Prentice Hall, 1938.
    """
    abc = abcd[:3]
    d = abcd[3]

    dist = np.abs(np.dot(abc, point)+d)
    return dist/np.linalg.norm(abc)


def compute_coefficients(xyz):
    """

    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    A = np.column_stack([x, y, np.ones_like(x)])
    abd, residuals, rank, s = np.linalg.lstsq(A, z)

    # add the coefficient of Z to
    return np.insert(abd, 2, -1)


def point_to_segment_distance(abcd, point, xyz):
    """Compute the distance from a point to a segment of a plane

    Parameters
    ----------
    point : array-like
        A point in space
    xyz : array-like
        Points that were used to define the plane
    abcd : array-like
        Coefficients of the plane in the form a*x + b*y + c*z + d = 0, where
        the equivalent array looks like [a, b, c, d]

    Returns
    -------
    float
        Distance from the point to the segment of the plane that spans
        throughout the points in `xyz`.

    Notes
    -----
    This function doesn't handle the N-dimensional problem and is specific to 3
    dimensions, but it should be straight-forward to extend into an
    N-dimensional solution.

    References
    ----------
    .. [1] http://stackoverflow.com/a/16459129
    """
    def plane(_abcd, xy):
        _a, _b, _c, _d = _abcd
        x, y = xy
        return (_a*x + _b*y + _d)/(-1*_c)

    a, b, c, d = abcd
    p, q, r = point
    l = ((d*-1.0) - p*a - b*q -c*r) / (a**2 +b**2 +c**2)
    extreme = np.array([p + l*a, q + l*b, r + l*c])

    for i in range(xyz.shape[-1]):
        vector = xyz[:, i]
        ranges = (vector.min(), vector.max())

        if extreme[i] < ranges[0]:
            extreme[i] = ranges[0]
        elif extreme[i] > ranges[1]:
            extreme[i] = ranges[1]

    extreme[-1] = plane(abcd, extreme[:-1])

    return euclidean(point, extreme)


def main():
    coords_ids, coords, _, _ = parse_coords('beta/pcoa_unweighted_unifrac_otu'
            '_table_even_14553.filtered.txt')
    mf = pd.read_csv('IBD_mapping_may_13_current.txt', sep='\t',
            index_col='#SampleID')
    coords = pd.DataFrame(coords, coords_ids)


if __name__ == '__main__':
    main()
