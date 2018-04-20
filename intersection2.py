import math
from collections import namedtuple
import numpy as np

from visible_segments import create_segments, intersect_segments
from math_utils import dist


Intersection = namedtuple('Intersection', 'theta, segment, intersection, distance')


def segment_limits(segment):
    min_i = math.ceil(math.degrees(segment.theta1))
    max_i = math.floor(math.degrees(segment.theta2))
    return min_i, max_i


class IntersectionFinder:
    def __init__(self, ref=None):
        if ref is None:
            ref = np.array([0, 0])
        self.ref = ref
        self._angles = [None for _ in range(0, 361, 1)]
        self._lims = []

    def add_segments(self, segments):
        for segment in segments:
            self.add_segment(segment)

    def add_segment(self, segment):
        segments = create_segments(segment, self.ref)
        for seg in segments:
            self._add_segment(seg)

    def _add_segment(self, segment):
        min_i, max_i = segment_limits(segment)

        i = min_i
        while i <= max_i:
            seg0 = self._angles[i]
            if seg0 is None:
                self._angles[i] = segment
            elif seg0 != segment:
                last_i = math.floor(math.degrees(seg0.theta2))
                new_segments = intersect_segments(seg0, segment)
                for seg1 in new_segments[:-1]:
                    min_i1, max_i1 = segment_limits(seg1)
                    l = max_i1+1-min_i1
                    self._angles[min_i1:max_i1+1] = [seg1 for _ in range(l)]
                segment = new_segments[-1]
                min_i1, max_i1 = segment_limits(segment)
                for j in range(min_i1, max_i1+1):
                    if self._angles[j] is None or j <= last_i:
                        self._angles[j] = segment
                    else:
                        break
                i = last_i
            i += 1

    def compute_intersections(self):
        intersections = []
        for i, seg in enumerate(self._angles):
            if seg is not None:
                trad = math.radians(i)
                direction = np.array([math.cos(trad), math.sin(trad)])
                exists, int_pt = seg.intersect(self.ref, direction)
                if exists:
                    new_int = Intersection(trad, seg, int_pt, dist(self.ref, int_pt))
                    intersections.append(new_int)

        return intersections

    @property
    def segments(self):
        return sorted(list(set([seg for seg in self._angles if seg is not None])), key=lambda s: s.theta1)


def find_intersections(p, segments):
    """
    Finds the closest intersection between rays starting at p and the line segments.

    Args:
        p (np.array): rays' starting point (x, y)
        segments (list[Segment]): list of line segments (Segment)

    Return:
       list[Intersection], list[Segment]: intersections and visible segments
    """

    if not segments:
        return [], []
    finder = IntersectionFinder(segments[0].ref)
    for seg in segments:
        finder.add_segment(seg)
    return finder.compute_intersections(), finder.segments
