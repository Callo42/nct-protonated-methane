import jax
import jax.numpy as jnp
import haiku as hk


class RealNVP(hk.Module):
    """
    Real-valued non-volume preserving (real NVP) transform.
    The implementation follows the paper "arXiv:1605.08803."
    """

    def __init__(
        self,
        nvp_depth: int,
        mlp_width: int,
        mlp_depth: int,
        event_size: int,
        key: jax.Array,
    ) -> None:
        """Init Network
        Args:
            nvp_depth: the number of nvp layers, in each nvp layer,
                the chosen n_channels of features are updates
                while others remains unchanged.
            mlp_width: the width of each layer in MLP of each nvp layer.
            mlp_depth: the depth of each MLP of each nvp layer.
            event_size: the feature's size
            key: the jax.random.PRNGKey
        """
        super().__init__()
        self.nvp_depth = nvp_depth
        self.n_channels = event_size // 2
        self.key = key

        # algorithm: the number of units in each
        # hidden layer is decreased by 1
        # with each hidden layer

        # check if the last layer of mlp
        # has more than event_size unit
        if (mlp_width - mlp_depth) < event_size:
            raise ValueError(
                "In the MLP embedding algorithm,"
                "need (mlp_width-mlp_depth)>=event_size"
                " to ensure that the last hidden layer"
                " has more than event_size units.\n"
                f"Get mlp_width={mlp_width}, mlp_depth={mlp_depth}"
                f" and event_size={event_size}\n"
            )

        mlp_shape = []
        for i in range(mlp_depth):
            mlp_shape.append(mlp_width - i)
        mlp_shape.append(event_size - self.n_channels)

        # MLP (Multi-Layer Perceptron) layers for the real NVP.
        self.fc_mlp = [
            hk.nets.MLP(
                mlp_shape,
                activation=jax.nn.tanh,
                w_init=hk.initializers.TruncatedNormal(stddev=0.00001),
                b_init=hk.initializers.TruncatedNormal(stddev=0.00001),
                activate_final=True,
            )
            for _ in range(nvp_depth)
        ]
        self.fc_mlp_2 = [
            hk.nets.MLP(
                mlp_shape,
                activation=jax.nn.tanh,
                w_init=hk.initializers.TruncatedNormal(stddev=0.00001),
                b_init=hk.initializers.TruncatedNormal(stddev=0.00001),
            )
            for _ in range(nvp_depth)
        ]
        self.scale = hk.get_parameter(
            "scale",
            [
                event_size - self.n_channels,
            ],
            init=jnp.ones,
            dtype=jnp.float64,
        )

    def coupling_forward(
        self, x1: jax.Array, x2: jax.Array, layer: int
    ) -> tuple[jax.Array, jax.Array]:
        """The affine coupling layer
        Args:
            x1: (n_channels,) the features that stay unchanged.
            x2: (event_size - n_channels,) the features that get updated
                in single affine coupling layer.
            layer: the int indicating the layer number
                of nvp_depth.
        Returns:
            y2: (event_size - n_channels,) the updated features, by appling
                affine mapping to x2: scale and shift.
            sum_logscale: the sum of different log jacobian determinant
                within this affine coupling layer.
        """
        # get shift and log(scale) from x1
        logscale = self.fc_mlp[layer](x1) * self.scale
        shift = self.fc_mlp_2[layer](x1)

        # transform: y2 = x2 * scale + shift
        y2 = x2 * jnp.exp(logscale) + shift

        # calculate: logjacdet for each layer
        sum_logscale = jnp.sum(logscale)

        return y2, sum_logscale

    def __call__(self, x):
        # ========== Real NVP (forward) ==========
        n, dim = x.shape

        # initial x and logjacdet
        x_flatten = jnp.reshape(x, (n * dim,))
        logjacdet = 0

        for layer in range(self.nvp_depth):
            # split x into two parts: x1, x2
            x1 = x_flatten[: self.n_channels]
            x2 = x_flatten[self.n_channels :]

            # get y2 from fc(x1), and calculate logjacdet = sum_l log(scale_l)
            y2, sum_logscale = self.coupling_forward(x1, x2, layer)
            logjacdet += sum_logscale

            # exchange feature
            x_flatten = jnp.concatenate([y2, x1])

        x = jnp.reshape(x_flatten, (n, dim))
        return x, logjacdet


def make_flow(
    key: jax.Array, nvp_depth: int, mlp_width: int, mlp_depth: int, event_size: int
) -> hk.Transformed:
    """Make flow (wrapper)
    Args:
        key: the jax.random.PRNGKey
        nvp_depth: the number of nvp layers, in each nvp layer,
            the chosen n_channels of features are updates
            while others remains unchanged.
        mlp_width: the width of each layer in MLP of each nvp layer.
        mlp_depth: the depth of each MLP of each nvp layer.
        event_size: the feature's size
    Returns:
        flow: the haiku transformed RNVP flow.
    """
    if not nvp_depth % 2 == 0:
        raise ValueError(
            "Need nvp_depth to be even number"
            "(for exchanging x1,x2 to get correct final order),"
            f" get {nvp_depth}."
        )

    def forward_fn(x):
        model = RealNVP(nvp_depth, mlp_width, mlp_depth, event_size, key)
        return model(x)

    flow = hk.transform(forward_fn)
    return flow
