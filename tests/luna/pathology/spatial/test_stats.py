from luna.pathology.spatial.stats import Kfunction
import numpy as np

def test_ls():
    r = 50
    R = np.linspace(1,100,10)
    p1 = np.random.rand(41,2)
    p2 = np.random.rand(17,2)
    I = np.random.rand(17)

    F00 = Kfunction(p1,p2,r,ls=True)
    F01 = Kfunction(p1,p2,r,ls=False)
    F10 = Kfunction(p1,p2,R,ls=True)
    F11 = Kfunction(p1,p2,R,ls=False)

    print (F00['count'])

    assert(len(F00['count']) == len(p1))
    assert(type(F01['count']) == np.float64)

    assert(len(F10['count']) == len(R))
    assert(type(F10['count'][0]) == np.ndarray)

    assert(len(F11['count']) == len(R))
    assert(type(F11['count'][0]) == np.float64)

def test_count():
    p1 = np.array([
        [0, 0],
        [0, 30],
        [30, 0],
        [30, 30]
    ])

    p2 = np.array([
        [0, 39],
        [40, 40]
    ])

    r = 10

    F00 = Kfunction(p1,p2,r,ls=True, count=True)
    F01 = Kfunction(p1,p2,r,ls=False, count=True)

    assert np.array_equal(F00['count'], [0, 1, 0, 0])
    assert F01['count'] == 0.25

def test_intensity():
    p1 = np.array([
        [0, 0],
        [0, 30],
        [30, 0],
        [30, 30]
    ])

    p2 = np.array([
        [1,1],
        [0, 39],
        [40, 40]
    ])

    I = np.array([1,2,3])

    r = 10

    F00 = Kfunction(p1,p2,r,ls=True, count=False, intensity=I)
    F01 = Kfunction(p1,p2,r,ls=False, count=False, intensity=I)

    assert np.array_equal(F00['intensity'], [1, 2, 0, 0])
    assert F01['intensity'] == 0.75


def test_distance():
    p1 = np.array([
        [0, 0],
        [0, 30],
        [30, 0],
        [30, 30]
    ])

    p2 = np.array([
        [1,1],
        [0, 39],
        [40, 40]
    ])

    I = np.array([1,2,3])

    r = 10

    F00 = Kfunction(p1,p2,r,ls=True, count=False, intensity=I, distance=True)
    F01 = Kfunction(p1,p2,r,ls=False, count=False, intensity=I, distance=True)

    assert F01['distance'] == 0.07159559660409909
