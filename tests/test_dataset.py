from video_classification.dataset import random_range


def test_random_range_short():
    start, stop, step = random_range(29, 50)
    assert (start, stop, step) == (0, 29, 1)


def test_random_range_step_range():
    start, stop, step = random_range(100, 50)
    assert step < 3
