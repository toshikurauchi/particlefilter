import math
import bisect
import attr
import numpy as np


EPS = 1e-8


def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


@attr.s(cmp=False)
class Segment:
    p1 = attr.ib()
    p2 = attr.ib()
    _theta1 = attr.ib(default=None)
    _theta2 = attr.ib(default=None)
    ref = attr.ib(default=attr.Factory(lambda: np.array([0, 0])))

    @property
    def theta1(self):
        if self._theta1 is None:
            self._theta1 = my_atan2(self.p1[1] - self.ref[1],
                                    self.p1[0] - self.ref[0])
        return self._theta1

    @theta1.setter
    def theta1(self, t1):
        self._theta1 = t1

    @property
    def theta2(self):
        if self._theta2 is None:
            self._theta2 = my_atan2(self.p2[1] - self.ref[1],
                                    self.p2[0] - self.ref[0])
            if self._theta2  < EPS and self._theta2 < self._theta1:
                self._theta2 = 2 * math.pi
        return self._theta2

    @theta2.setter
    def theta2(self, t2):
        self._theta2 = t2

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return np.allclose(self.p1, other.p1) and \
               np.allclose(self.p2, other.p2) and \
               abs(self.theta1 - other.theta1) < EPS and \
               abs(self.theta2 - other.theta2) < EPS and \
               np.allclose(self.ref, other.ref)

    def __lt__(self, other):
        return self.theta2 < other.theta1

    def __le__(self, other):
        return self.theta2 <= other.theta1

    def __gt__(self, other):
        return self.theta1 > other.theta2

    def __ge__(self, other):
        return self.theta1 >= other.theta2

    def merge(self, other):
        if other is None:
            return
        if self.theta1 > other.theta1:
            self.theta1 = other.theta1
            self.p1 = other.p1
        if self.theta2 < other.theta2:
            self.theta2 = other.theta2
            self.p2 = other.p2

    def intersect(self, orig, direct):
        """
        Finds intersection between ray and self.

        Args:
            orig (np.array): ray's starting point
            direct (np.array): ray's direction

        Return:
            exists, intersection: exists is a boolean indicating if the
            intersection point exists.
        """
        # Init vars
        direct = direct / np.linalg.norm(direct)
        ctheta, stheta = direct
        px, py = orig
        x1, y1 = self.p1
        x2, y2 = self.p2

        # Compute s and r
        denom = ((x2 - x1) * stheta + (y1 - y2) * ctheta)
        if abs(denom) < EPS:
            return False, None
        s = ((px - x1) * stheta + (y1 - py) * ctheta) / denom
        if abs(ctheta) > abs(stheta):
            r = (x1 + s * (x2 - x1) - px) / ctheta
        else:
            r = (y1 + s * (y2 - y1) - py) / stheta

        if r < -EPS or s < -EPS or s > 1+EPS:
            return False, None
        return True, np.array([px + r * ctheta, py + r * stheta])


def my_atan2(y, x):
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2 * math.pi
    return theta


def create_segments(segment, ref=None):
    if ref is None:
        ref = np.array([0, 0])

    p1 = segment.p1
    p2 = segment.p2
    x1 = p1[0] - ref[0]
    y1 = p1[1] - ref[1]
    x2 = p2[0] - ref[0]
    y2 = p2[1] - ref[1]

    if x1 * y2 - x2 * y1 < 0:
        p1, p2 = p2, p1
        x1, y1, x2, y2 = x2, y2, x1, y1

    theta1 = my_atan2(y1, x1)
    theta2 = my_atan2(y2, x2)

    hor = np.array([1, 0])
    intersects, intersection = segment.intersect(ref, hor)
    if intersects:
        if abs(theta2) < EPS and abs(theta1) > EPS:
            return [Segment(p1, p2, theta1, 2 * math.pi, ref)]
        elif abs(theta1) < EPS and abs(theta2) > EPS:
            return [Segment(p1, p2, 0, theta2, ref)]
        return [
            Segment(intersection, p2, 0, theta2, ref),
            Segment(p1, intersection, theta1, 2 * math.pi, ref),
        ]
    return [Segment(p1, p2, theta1, theta2, ref)]


def make_segment(p1, p2, ref=None):
    if p1 is not None and p2 is not None and not np.allclose(p1, p2):
        return Segment(p1, p2, ref=ref)
    return None


def intersect_segments(n1, n2):
    if n1.theta2 <= n2.theta1:
        return [n1, n2]
    elif n2.theta2 <= n1.theta1:
        return [n2, n1]
    # Just to make sure things are ok
    np.testing.assert_allclose(n1.ref, n2.ref)
    ref = n1.ref
    angles = sorted([n1.theta1, n1.theta2, n2.theta1, n2.theta2])
    directs = [np.array([math.cos(t), math.sin(t)]) for t in angles]
    pts1 = [n1.intersect(ref, d)[1] for d in directs]
    pts2 = [n2.intersect(ref, d)[1] for d in directs]
    segs1 = [make_segment(p1, p2, ref) for p1, p2 in zip(pts1[:-1], pts1[1:])]
    segs2 = [make_segment(p1, p2, ref) for p1, p2 in zip(pts2[:-1], pts2[1:])]
    # The intersection (segsX[1]) always exists
    # But we don't know about the others (segsX[0] and segsX[2] may be None)
    d11 = dist_sq(segs1[1].p1, ref)
    d12 = dist_sq(segs1[1].p2, ref)
    d21 = dist_sq(segs2[1].p1, ref)
    d22 = dist_sq(segs2[1].p2, ref)
    segments = []
    if d11 <= d21 and d12 <= d22:
        segment = Segment(pts1[1], pts1[2], angles[1], angles[2], ref)
        if pts1[0] is not None:
            segment.merge(Segment(pts1[0], pts1[1], angles[0], angles[1], ref))
        if pts1[3] is not None:
            segment.merge(Segment(pts1[2], pts1[3], angles[2], angles[3], ref))
        segments.append(segment)
        if segs2[0] is not None:
            segments = [Segment(pts2[0], pts2[1], angles[0], angles[1], ref)] + segments
        if segs2[2] is not None:
            segments.append(Segment(pts2[2], pts2[3], angles[2], angles[3], ref))
    elif d21 <= d11 and d22 <= d12:
        pts1, pts2 = pts2, pts1
        segs1, segs2 = segs2, segs1
        segment = Segment(pts1[1], pts1[2], angles[1], angles[2], ref)
        if pts1[0] is not None:
            segment.merge(Segment(pts1[0], pts1[1], angles[0], angles[1], ref))
        if pts1[3] is not None:
            segment.merge(Segment(pts1[2], pts1[3], angles[2], angles[3], ref))
        segments.append(segment)
        if segs2[0] is not None:
            segments = [Segment(pts2[0], pts2[1], angles[0], angles[1], ref)] + segments
        if segs2[2] is not None:
            segments.append(Segment(pts2[2], pts2[3], angles[2], angles[3], ref))
    elif d11 <= d21 and d22 <= d12:
        # Crossed
        inters = segs1[1].intersect(segs2[1].p1, segs2[1].p2-segs2[1].p1)[1]
        inters_centered = inters - ref
        inters_ang = my_atan2(inters_centered[1], inters_centered[0])
        segments.append(Segment(pts1[1], inters, angles[1], inters_ang, ref))
        segments.append(Segment(inters, pts2[2], inters_ang, angles[2], ref))
        if pts1[0] is not None:
            segments[0].merge(Segment(pts1[0], pts1[1], angles[0], angles[1], ref))
        if pts2[3] is not None:
            segments[1].merge(Segment(pts2[2], pts2[3], angles[2], angles[3], ref))
        if segs2[0] is not None:
            segments = [Segment(pts2[0], pts2[1], angles[0], angles[1], ref)] + segments
        if segs1[2] is not None:
            segments.append(Segment(pts1[2], pts1[3], angles[2], angles[3], ref))
    else:
        # Crossed
        inters = segs1[1].intersect(segs2[1].p1, segs2[1].p2-segs2[1].p1)[1]
        inters_centered = inters - ref
        inters_ang = my_atan2(inters_centered[1], inters_centered[0])
        segments.append(Segment(pts2[1], inters, angles[1], inters_ang, ref))
        segments.append(Segment(inters, pts1[2], inters_ang, angles[2], ref))
        if pts2[0] is not None:
            segments[0].merge(Segment(pts2[0], pts2[1], angles[0], angles[1], ref))
        if pts1[3] is not None:
            segments[1].merge(Segment(pts1[2], pts1[3], angles[2], angles[3], ref))
        if segs1[0] is not None:
            segments = [Segment(pts1[0], pts1[1], angles[0], angles[1], ref)] + segments
        if segs2[2] is not None:
            segments.append(Segment(pts2[2], pts2[3], angles[2], angles[3], ref))
    return segments


@attr.s
class VisibleSegments:
    _ref = attr.ib(default=attr.Factory(lambda: np.array([0, 0])))
    segments = attr.ib(default=attr.Factory(list))

    def add_segments(self, segments):
        for segment in segments:
            self.add_segment(segment)

    def add_segment(self, segment):
        new_segments = create_segments(segment, self._ref)
        for seg in new_segments:
            self._add_segment_r(seg)

    def _readd_intersecting(self, idx, segment):
        previous = self.segments[idx]
        del self.segments[idx]
        new_segs = intersect_segments(segment, previous)
        for seg in new_segs:
            self._add_segment_r(seg)

    def _add_segment_r(self, segment):
        i = bisect.bisect(self.segments, segment)
        if i > 0 and self.segments[i-1].theta2 > segment.theta1:
            self._readd_intersecting(i-1, segment)
        elif i < len(self.segments) - 1 and \
            self.segments[i+1].theta1 < segment.theta2:
            self._readd_intersecting(i+1, segment)
        else:
            self.segments.insert(i, segment)

