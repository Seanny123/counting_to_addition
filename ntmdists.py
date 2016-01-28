from nengo.dists import *
from sobol_seq import i4_sobol_generate


class SphericalCoords(Distribution):

    def __init__(self, m):
        self.m = m

    def sample(self, num, d=None, rng=np.random):
        shape = self._sample_shape(num, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        from scipy.special import beta
        return np.pi * np.sin(np.pi * x) ** (self.m-1) / beta(self.m / 2.0, 0.5)

    def cdf(self, x):
        from scipy.special import betainc
        y = 0.5 * betainc(self.m / 2.0, 0.5, np.sin(np.pi * x) ** 2)
        return np.where(x < 0.5, y, 1 - y)

    def ppf(self, y):
        from scipy.special import betaincinv
        y_reflect = np.where(y < 0.5, y, 1 - y)
        z_sq = betaincinv(self.m / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)


class Sobol(Distribution):

    def sample(self, num, d=None, rng=np.random):
        if d is None or d < 1 or d > 40: # TODO: also check if integer
            raise ValueError("d (%d) must be integer in range [1, 40]" % d)
        num, d = self._sample_shape(num, d)
        #from scipy.stats import uniform
        #mdd = MultiDimDistribution([uniform for _ in range(d)])
        #return np.asarray(mdd.sobol(num))
        return i4_sobol_generate(d, num, skip=0)


def random_orthogonal(d, rng=np.random):
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    # formula for nearest orthogonal matrix (assumes m is linearly independent)
    u, s, v = np.linalg.svd(m)
    return np.dot(u, v)


class ScatteredHypersphere(UniformHypersphere):

    def sample(self, num, d=1.0, rng=np.random, ntm=Sobol()):
        if self.surface:
            cube = ntm.sample(num, d-1)
            radius = 1.0
        else:
            dcube = ntm.sample(num, d)
            cube, radius = dcube[:, :-1], dcube[:, -1:] ** (1.0 / d)

        # inverse transform method (section 1.5.2)
        for j in range(d-1):
            cube[:, j] = SphericalCoords(d-1-j).ppf(cube[:, j])

        # spherical coordinate transform
        i = np.ones(d-1)
        i[-1] = 2.0
        s = np.sin(i[None, :] * np.pi * cube)
        c = np.cos(i[None, :] * np.pi * cube)
        mapped = np.ones((num, d))
        mapped[:, 1:] = np.cumprod(s, axis=1)
        mapped[:, :-1] *= c
        assert np.allclose(npext.norm(mapped, axis=1, keepdims=True), 1)

        # radius adjustment for ball versus sphere, and rotate
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)
        
        
if __name__ == '__main__':

    import pylab
    pylab.figure(figsize=(6, 6))
    pylab.scatter(*ScatteredHypersphere(surface=False).sample(400, 2).T, lw=0)
    pylab.scatter(*ScatteredHypersphere(surface=True).sample(100, 2).T, c='r', lw=0)
    pylab.show()