import math
from collections import namedtuple
import numpy as np

from visible_segments import VisibleSegments
from math_utils import dist


Intersection = namedtuple('Intersection', 'theta, segment, intersection, distance')


def find_intersections(p, segments, step=1):
    """
    Finds the closest intersection between rays starting at p and the line segments.

    Args:
        p (np.array): rays' starting point (x, y)
        segments (list[Segment]): list of line segments (Segment)
        step (float): step between rays (degrees). Default: 1

    Return:
       list[Intersection], list[Segment]: intersections and visible segments
    """

    visible = VisibleSegments(p)
    visible.add_segments(segments)
    it = iter(visible.segments)
    segment = next(it, None)

    intersections = []
    for theta in range(0, 360, step):
        trad = math.radians(theta)

        while segment and segment.theta2 < trad:
            segment = next(it, None)
        if segment is None:
            break

        if segment.theta1 <= trad:
            exists, intersection = segment.intersect(p, np.array([math.cos(trad), math.sin(trad)]))
            if exists:
                intersections.append(Intersection(theta, segment, intersection, dist(p, intersection)))
    return intersections, visible.segments
