import jax
import jax.numpy as jnp
import haiku as hk

# Max magnitude for log-scale (after tanh)
_LOGSCALE_LIMIT = 5.0


def _random_binary_masks_init(shape, dtype):
    """Initializer for random binary masks with cyclic shifts across layers.

    Args:
        shape: (n_layers, event_size)
        dtype: typically jnp.float64.
    """
    if len(shape) != 2:
        raise ValueError(f"Mask shape must be (n_layers, event_size); got {shape}")
    n_layers, event_size = shape

    key = hk.next_rng_key()
    base = jax.random.bernoulli(key, p=0.5, shape=(event_size,))
    base = base.astype(dtype)

    # Build per-layer masks by cyclically shifting a base random pattern.
    masks = [jnp.roll(base, shift=i) for i in range(n_layers)]
    return jnp.stack(masks, axis=0)


class ResidualMLP(hk.Module):
    """Residual MLP mapping R^{event_size} -> R^{event_size}.

    - depth residual blocks: each block is x -> x + f(x)
    - final linear layer is zero-initialized to keep output near zero at init
    """

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        event_size: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.depth = depth
        self.event_size = event_size

        # Residual blocks: each block keeps dimension event_size
        self.fc1_layers = [
            hk.Linear(
                hidden_size,
                name=f"res_block_{i}_fc1",
            )
            for i in range(depth)
        ]
        self.fc2_layers = [
            hk.Linear(
                event_size,
                name=f"res_block_{i}_fc2",
            )
            for i in range(depth)
        ]

        # Final projection, zero-initialized => output is ~0 at init
        self.out = hk.Linear(
            event_size,
            w_init=lambda shape, dtype: jnp.zeros(shape, dtype),
            b_init=lambda shape, dtype: jnp.zeros(shape, dtype),
            name="out",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (event_size,)
        h = x
        for fc1, fc2 in zip(self.fc1_layers, self.fc2_layers):
            residual = h
            h = jax.nn.silu(fc1(h))
            h = fc2(h)
            h = h + residual  # residual connection

        return self.out(h)


class RealNVP(hk.Module):
    """Real-valued non-volume-preserving (RealNVP) transform with random masks.

    Forward-only: returns (y, log|det J|) for a single event.

    Expects x with shape (n, dim) such that n * dim == event_size.
    """

    def __init__(
        self,
        nvp_depth: int,
        mlp_width: int,
        mlp_depth: int,
        event_size: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if nvp_depth <= 0:
            raise ValueError(f"nvp_depth must be positive; got {nvp_depth}")
        if event_size <= 0:
            raise ValueError(f"event_size must be positive; got {event_size}")
        if mlp_depth < 0:
            raise ValueError(f"mlp_depth must be >= 0; got {mlp_depth}")

        self.nvp_depth = nvp_depth
        self.event_size = event_size

        # Residual MLPs for shift and log-scale in each coupling layer
        self.shift_nets = [
            ResidualMLP(
                hidden_size=mlp_width,
                depth=mlp_depth,
                event_size=event_size,
                name=f"shift_net_{i}",
            )
            for i in range(nvp_depth)
        ]
        self.logscale_nets = [
            ResidualMLP(
                hidden_size=mlp_width,
                depth=mlp_depth,
                event_size=event_size,
                name=f"logscale_net_{i}",
            )
            for i in range(nvp_depth)
        ]

        # Learnable per-dimension scaling of log-scale outputs (small init).
        self.logscale_scale = hk.get_parameter(
            "logscale_scale",
            shape=(event_size,),
            dtype=jnp.float64,
            init=lambda shape, dtype: jnp.ones(shape, dtype),
        )

        # Per-layer ActNorm parameters: log-scale and bias, identity at init.
        self.act_log_scale = hk.get_parameter(
            "act_log_scale",
            shape=(nvp_depth, event_size),
            dtype=jnp.float64,
            init=jnp.zeros,
        )
        self.act_bias = hk.get_parameter(
            "act_bias",
            shape=(nvp_depth, event_size),
            dtype=jnp.float64,
            init=jnp.zeros,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward RealNVP transform for a single event.

        Args:
            x: array with shape (n, dim), n * dim must equal event_size.

        Returns:
            y: transformed x with same shape as input (n, dim), dtype float64.
            logjacdet: scalar log|det J| (float64).
        """
        # Force 64-bit inputs (assuming jax_enable_x64=True globally)
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be rank-2 (n, dim); got shape {x.shape}")

        n, dim = x.shape
        if n * dim != self.event_size:
            raise ValueError(
                f"Mismatch: n*dim={n*dim} but event_size={self.event_size}. "
                "Ensure event_size matches n*dim."
            )

        # Flatten to a single event vector of length event_size.
        v = jnp.reshape(x, (self.event_size,))

        # Random binary masks per layer, created once at init.
        masks = hk.get_parameter(
            "masks",
            shape=(self.nvp_depth, self.event_size),
            dtype=jnp.float64,
            init=_random_binary_masks_init,
        )
        # Treat masks as constants (effectively non-trainable).
        masks = jax.lax.stop_gradient(masks)

        logjacdet = jnp.zeros((), dtype=jnp.float64)

        for layer in range(self.nvp_depth):
            # ---------- ActNorm (invertible per-layer affine) ----------
            log_s = self.act_log_scale[layer]  # (event_size,)
            b = self.act_bias[layer]           # (event_size,)
            s = jnp.exp(log_s)

            v = v * s + b
            logjacdet = logjacdet + jnp.sum(log_s)

            # ---------- Affine coupling with random mask ----------
            mask = masks[layer]              # (event_size,)
            inv_mask = 1.0 - mask

            # Condition on masked-in part
            x1 = v * mask

            # Compute log-scale and shift from x1
            raw_logscale = self.logscale_nets[layer](x1)
            raw_logscale = raw_logscale * self.logscale_scale

            # Stabilize log-scale: tanh + global limit
            logscale = jnp.tanh(raw_logscale) * _LOGSCALE_LIMIT

            shift = self.shift_nets[layer](x1)

            # Affine transform only on masked-out dimensions
            exp_logscale = jnp.exp(logscale)
            v = v * mask + inv_mask * (v * exp_logscale + shift)

            # Log-det from coupling: only transformed dims contribute
            logjacdet = logjacdet + jnp.sum(inv_mask * logscale)

        y = jnp.reshape(v, (n, dim))
        return y, logjacdet


def make_flow(
    key,
    nvp_depth: int,
    mlp_width: int,
    mlp_depth: int,
    event_size: int,
) -> hk.Transformed:
    """Create a Haiku-transformed RealNVP forward function.

    Usage:
        flow = make_flow(nvp_depth, mlp_width, mlp_depth, event_size)
        params = flow.init(rng_key, x0)  # x0: shape (n, dim), n*dim == event_size
        y, logdet = flow.apply(params, rng_key, x)
    """

    def forward_fn(x):
        model = RealNVP(nvp_depth, mlp_width, mlp_depth, event_size)
        return model(x)

    return hk.transform(forward_fn)
