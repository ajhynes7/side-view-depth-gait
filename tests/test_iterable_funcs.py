import modules.iterable_funcs as itf


def test_pairwise():

    x = [1, 2, 3, 4]

    assert [*itf.pairwise(x)] == [(1, 2), (2, 3), (3, 4)]


def test_iterable_to_dict():

    assert itf.iterable_to_dict([]) == {}

    x = [1, 2, 3, 10, 50]
    d = itf.iterable_to_dict(x)

    assert d == {0: 1, 1: 2, 2: 3, 3: 10, 4: 50}
