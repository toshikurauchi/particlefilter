import random
import time
import math
import numpy as np

from segment import Segment, random_segment
from intersection import IntersectionFinder, find_intersections


def test_add_segments():
    visible = IntersectionFinder()
    segment = Segment(np.array([1, 0]), np.array([1, 1]))
    visible.add_segment(segment)
    segments = visible.segments
    assert len(segments) == 1
    assert segments[0] == segment


def test_add_two_segments():
    visible = IntersectionFinder()
    segment1 = Segment(np.array([1, 0]), np.array([1, 1]))
    segment2 = Segment(np.array([-1, 0]), np.array([-1, -1]))
    visible.add_segment(segment2)
    visible.add_segment(segment1)
    segments = visible.segments
    assert len(segments) == 2
    assert segments[0] == segment1
    assert segments[1] == segment2

def test_add_split():
    visible = IntersectionFinder()
    p1 = np.array([1, -1])
    p2 = np.array([1, 0])
    p3 = np.array([1, 1])
    segment = Segment(p3, p1)
    visible.add_segment(segment)
    segments = visible.segments
    assert len(segments) == 2
    assert segments[0] == Segment(p2, p3)
    assert segments[1] == Segment(p1, p2)

def test_add_square():
    visible = IntersectionFinder()
    pts = [
        np.array([1, 0]),
        np.array([1, 1]),
        np.array([-1, 1]),
        np.array([-1, -1]),
        np.array([1, -1]),
    ]
    segs = [Segment(p1, p2) for p1, p2 in zip(pts, pts[1:] + pts[0:1])]
    visible.add_segments(segs)
    segments = visible.segments
    assert len(segments) == 5
    for s1, s2 in zip(segments, segs):
        assert s1 == s2

def test_add_intersecting_segments():
    visible = IntersectionFinder()
    p1 = np.array([1, 1])
    p2 = np.array([-2, 1])
    p3 = np.array([-1, 2])
    p4 = np.array([-1, -1])
    p5 = np.array([-1, 1])
    visible.add_segment(Segment(p1, p2))
    segments = visible.segments
    assert len(segments) == 1
    assert segments[0] == Segment(p1, p2)
    visible.add_segment(Segment(p3, p4))
    segments = visible.segments
    assert len(segments) == 2
    assert segments[0] == Segment(p1, p5)
    assert segments[1] == Segment(p5, p4)


def test_add_intersecting_segments_inverse():
    visible = IntersectionFinder()
    p1 = np.array([1, 1])
    p2 = np.array([-2, 1])
    p3 = np.array([-1, 2])
    p4 = np.array([-1, -1])
    p5 = np.array([-1, 1])
    visible.add_segment(Segment(p3, p4))
    segments = visible.segments
    assert len(segments) == 1
    assert segments[0] == Segment(p3, p4)
    visible.add_segment(Segment(p1, p2))
    segments = visible.segments
    assert len(segments) == 2
    assert segments[0] == Segment(p1, p5)
    assert segments[1] == Segment(p5, p4)


def test_add_intersecting_segments_first_farther():
    ref = np.array([10, -100])
    visible = IntersectionFinder(ref)
    segs = [
        Segment(np.array([-3, 3]) + ref, np.array([1, 0]) + ref, ref=ref),
        Segment(np.array([3, 3]) + ref, np.array([-1, 0]) + ref, ref=ref),
    ]
    exp = [
        Segment(np.array([1, 0]) + ref, np.array([0, 0.75]) + ref, ref=ref),
        Segment(np.array([0, 0.75]) + ref, np.array([-1, 0]) + ref, ref=ref),
    ]
    visible.add_segments(segs)
    for segment, expected in zip(visible.segments, exp):
        assert segment == expected

    # Inverse
    visible = IntersectionFinder(ref)
    visible.add_segments(segs[::-1])
    for segment, expected in zip(visible.segments, exp):
        assert segment == expected


def test_intersecting_square():
    visible = IntersectionFinder()
    visible.add_segments([
        Segment(np.array([-2, 1]), np.array([2, 1])),
        Segment(np.array([-2, -1]), np.array([2, -1])),
        Segment(np.array([1, 2]), np.array([1, -2])),
        Segment(np.array([-1, 2]), np.array([-1, -2])),
    ])
    segs = [
        Segment(np.array([1, 0]), np.array([1, 1])),
        Segment(np.array([1, 1]), np.array([-1, 1])),
        Segment(np.array([-1, 1]), np.array([-1, -1])),
        Segment(np.array([-1, -1]), np.array([1, -1])),
        Segment(np.array([1, -1]), np.array([1, 0])),
    ]
    for segment, expected in zip(visible.segments, segs):
        assert segment == expected


def test_intersecting_limit():
    ref = np.array([10, -100])
    visible = IntersectionFinder(ref)
    segs = [
        Segment(np.array([1, 1]) + ref, np.array([-1, 1]) + ref, ref=ref),
        Segment(np.array([-1, 1]) + ref, np.array([-1, -1]) + ref, ref=ref),
    ]
    visible.add_segments(segs)
    for segment, expected in zip(visible.segments, segs):
        assert segment == expected

    # Inverse
    visible = IntersectionFinder(ref)
    visible.add_segments(segs[::-1])
    for segment, expected in zip(visible.segments, segs):
        assert segment == expected


def test_intersecting_first():
    ref = np.array([10, -100])
    visible = IntersectionFinder(ref)
    segs = [
        Segment(np.array([1, 1]) + ref, np.array([-2, 1]) + ref, ref=ref),
        Segment(np.array([-1, 1]) + ref, np.array([-1, -1]) + ref, ref=ref),
    ]
    exp = [
        Segment(np.array([1, 1]) + ref, np.array([-1, 1]) + ref, ref=ref),
        segs[1],
    ]
    visible.add_segments(segs)
    for segment, expected in zip(visible.segments, exp):
        assert segment == expected

    # Inverse
    visible = IntersectionFinder(ref)
    visible.add_segments(segs[::-1])
    for segment, expected in zip(visible.segments, exp):
        assert segment == expected


def test_intersecting_second():
    ref = np.array([10, -100])
    visible = IntersectionFinder(ref)
    segs = [
        Segment(np.array([1, 1]) + ref, np.array([-1, 1]) + ref, ref=ref),
        Segment(np.array([-1, 2]) + ref, np.array([-1, -1]) + ref, ref=ref),
    ]
    exp = [
        segs[0],
        Segment(np.array([-1, 1]) + ref, np.array([-1, -1]) + ref, ref=ref),
    ]
    visible.add_segments(segs)
    for segment, expected in zip(visible.segments, exp):
        assert segment == expected

    # Inverse
    visible = IntersectionFinder(ref)
    visible.add_segments(segs[::-1])
    for segment, expected in zip(visible.segments, exp):
        assert segment == expected


def main():
    # Times
    times = []
    # Number of random segments
    N = 1000
    TRIALS = 10

    for _ in range(TRIALS):
        origin = np.array([random.uniform(4, 6), random.uniform(4, 6)])
        segments = [random_segment(origin) for _ in range(N)]

        t0 = time.time()
        find_intersections(origin, segments)
        times.append(time.time() - t0)
    mt = np.mean(times)
    print(times)
    print('Mean: {:.4f}'.format(mt))
    print('{:.2f} fps'.format(1/mt))


if __name__ == '__main__':
    main()
