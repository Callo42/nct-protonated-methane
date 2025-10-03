"""MoleNet: Equivariant Flow For Molecule"""

from typing import Optional
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk


class MoleNet(hk.Module):
    """MoleNet, modified from FermiNet."""

    def __init__(
        self,
        depth: int,
        h1_size: int,
        h2_size: int,
        partitions: list[int],
        init_stddev: float = 0.0001,
        remat: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.depth = depth
        self.partitions = partitions
        self.init_stddev = init_stddev
        self.remat = remat

        self.fc1 = [
            hk.Linear(h1_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev))
            for d in range(depth)
        ]
        self.fc2 = [
            hk.Linear(h2_size, w_init=hk.initializers.TruncatedNormal(self.init_stddev))
            for d in range(depth)
        ]

    # def _spstream0(self, x, *args):
    #     pass

    def _tpstream0(self, x):
        n = x.shape[0]
        rij = x[:, None, :] - x

        # Avoid computing the norm of zero, as is has undefined grad
        r = jnp.linalg.norm(rij + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n))

        f = [r[..., None]]
        f += [
            rij,
        ]
        return jnp.concatenate(f, axis=-1)

    def _combine(self, h1, h2, partitions):
        n = h2.shape[0]

        h2s = jnp.split(h2, partitions, axis=0)
        g2 = [jnp.mean(h, axis=0) for h in h2s if h.size > 0]

        if h1 is None:
            f = jnp.concatenate(g2, axis=1)
        else:
            h1s = jnp.split(h1, partitions, axis=0)
            g1 = [jnp.mean(h, axis=0, keepdims=True) for h in h1s if h.size > 0]
            g1 = [jnp.tile(g, [n, 1]) for g in g1]
            f = jnp.concatenate([h1] + g1 + g2, axis=1)
        return f

    def _h1_h2(self, h10, h20, partitions):
        h1, h2 = h10, h20

        def block(h1, h2, d):
            f = self._combine(h1, h2, partitions)
            h1_update = jax.nn.tanh(self.fc1[d](f))
            h2_update = jax.nn.tanh(self.fc2[d](h2))
            if d > 0:
                h1 = h1_update + h1
                h2 = h2_update + h2
            else:
                h1 = h1_update
                h2 = h2_update
            return h1, h2

        if self.remat:
            block = hk.remat(block, static_argnums=2)

        for d in range(self.depth):
            h1, h2 = block(h1, h2, d)

        return h1, h2


class MoleNetFlow(MoleNet):
    """Flow from MoleNet Building blocks"""

    def _spstream0(self, x: jax.Array) -> jax.Array:
        """Init single particle features.
        W.r.t center of mass (suppose at origin).

        Args:
            x: (num_of_atoms, dim) the cartesian coordinates.
        Returns:
            spstream:
        """
        cm = jnp.array(0.0, dtype=jnp.float64)
        delta_r = x - cm
        delta_r_norm = jnp.linalg.norm(delta_r, axis=1, keepdims=True)
        spstream = jnp.concatenate((delta_r_norm, delta_r), axis=-1)
        return spstream

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward function

        Args:
            x: (num_of_atoms, dim) the cartesian coordinates.
        """
        cartesian_coors = x

        n, dim = cartesian_coors.shape
        h10 = self._spstream0(cartesian_coors)
        h20 = self._tpstream0(cartesian_coors)
        h1, h2 = self._h1_h2(h10, h20, self.partitions)
        f = self._combine(h1, h2, self.partitions)

        final = hk.Linear(
            dim,
            w_init=hk.initializers.RandomNormal(self.init_stddev),
            # b_init=jnp.zeros,
        )
        cartesian_coors = final(f) + x
        # cartesian_coors = final(f)

        return cartesian_coors
