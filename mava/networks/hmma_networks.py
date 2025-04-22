from typing import Callable, Tuple, Union, Dict, Sequence
from typing_extensions import NamedTuple
import chex
import distrax
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal
from mava.networks.mamba_bidirectional_block import BiMamba
from mava.networks.mamba_crossattention_block import CrossMamba
from mava.networks.mamba_selfattention_block import Mamba
from mava.utils.mam_utils import FIFOCircularBuffer
from mava.types import Buffer, HiddenState, Action, Done, State, Value, ValueNormParams
from jumanji.types import TimeStep
from optax._src.base import OptState
from flax.core.frozen_dict import FrozenDict

############# Mamba Self Attention ####################
class ModelArgs:
    """Mamba model arguments."""

    def __init__(
        self,
        num_agents: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        delta_rank: int,
        expand: int = 2,
        delta_min: float = 0.001,
        delta_max: float = 0.1,
        delta_init: str = "random",
        delta_scale: float = 1.0,
        delta_init_floor: float = 1e-4,
    ):
        self.num_agents = num_agents
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.delta_rank = delta_rank
        self.expand = expand
        self.d_inner = self.d_model * self.expand
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_init = delta_init
        self.delta_scale = delta_scale
        self.delta_init_floor = delta_init_floor

class Mamba(nn.Module):
    """Full Mamba model."""

    num_agents: int
    d_model: int
    d_state: int
    d_conv: int
    delta_rank: int

    def setup(self) -> None:
        self.args = ModelArgs(
            self.num_agents,
            self.d_model,
            self.d_state,
            self.d_conv,
            self.delta_rank,
        )
        self.outer_norm = nn.LayerNorm()
        self.block = ResidualBlock(self.args)

    def __call__(self, x: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Applies the module, passing in a single sequence."""
        x = self.block(x)  # We use only a single Mamba block and instead chain MAM blocks.
        x = self.outer_norm(x)
        return x

    def recurrent(
        self,
        x: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> Tuple[chex.Array, HiddenState, Buffer]:
        """Applies the module, passing in a single sequence."""
        x, hidden_state, buffer = self.block.recurrent(x, hidden_state, buffer)
        x = self.outer_norm(x)
        return x, hidden_state, buffer

class ResidualBlock(nn.Module):
    """Simple block wrapping Mamba block with normalisation and residual connection."""

    args: ModelArgs

    def setup(self) -> None:
        self.block = MambaBlock(self.args)
        self.norm = nn.RMSNorm()

    def __call__(
        self, x: Union[chex.Array, Tuple[chex.Array, chex.Array]]
    ) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Forward pass."""
        residual = x
        x = self.block(self.norm(x))
        return x + residual

    def recurrent(
        self,
        x: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> Union[
        Tuple[chex.Array, HiddenState, Buffer],
        Tuple[Tuple[chex.Array, chex.Array], HiddenState, Buffer],
    ]:
        """Recurrent forward pass."""
        residual = x
        x, hidden_state, buffer = self.block.recurrent(self.norm(x), hidden_state, buffer)
        return x + residual, hidden_state, buffer

class MambaBlock(nn.Module):
    """
    A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper.
    """

    args: ModelArgs

    def setup(self) -> None:
        # Set up projections for input entering and exiting the block.
        self.setup_input_and_output_projs()

        # Set up the 1D causal convolutional layer.
        self.setup_conv_layer()

        # Set up the SSM parameters.
        self.setup_A()
        self.setup_D()
        self.setup_delta()
        # x_proj takes in `x` and outputs the input-specific Δ, B, C.
        self.x_proj = nn.Dense(self.args.delta_rank + self.args.d_state * 2, use_bias=False)

    def setup_input_and_output_projs(self) -> None:
        """Set up the projections for input entering and exiting the Mamba block."""
        # in_proj takes in `x` and outputs a transformed `x` and its residual.
        self.in_proj = nn.Dense(self.args.d_inner * 2, use_bias=False)
        # out_proj projects back to output dimension.
        self.out_proj = nn.Dense(self.args.d_model, use_bias=False)

    def setup_conv_layer(self) -> None:
        """
        Set up Mamba's 1D causal convolutional layer.
        Note that this convolution uses valid padding,
        so we must implement causal padding manually.
        """
        self.conv1d = nn.Conv(
            features=self.args.d_inner,
            kernel_size=(self.args.d_conv,),
            feature_group_count=self.args.d_inner,
            padding="VALID",
        )

    def setup_delta(self) -> None:
        """
        Set up and initialise the input-dependent SSM parameter Δ's projection
        from dimension delta_rank to d_inner.
        """

        # Initialise Δ projection weights to preserve variance at initialisation.
        def init_delta_weights(
            key: jax.random.PRNGKey, shape: Sequence[int], args: ModelArgs
        ) -> chex.Array:
            delta_init_std = args.delta_rank**-0.5 * args.delta_scale
            if args.delta_init == "random":
                return jax.random.uniform(key, shape, minval=-delta_init_std, maxval=delta_init_std)
            elif args.delta_init == "constant":
                return jax.numpy.full(shape, delta_init_std)

        # Initialise Δ projection bias so softplus(delta_bias) is between delta_min and delta_max.
        def init_delta_bias(
            key: jax.random.PRNGKey, shape: Sequence[int], args: ModelArgs
        ) -> chex.Array:
            dt = jnp.exp(
                jax.random.uniform(key, shape) * (jnp.log(args.delta_max) - jnp.log(args.delta_min))
                + jnp.log(args.delta_min)
            )
            dt = jnp.clip(dt, a_min=args.delta_init_floor)
            # Inverse of softplus for numerical stability.
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt

        init_delta_kernel_fn = lambda key, shape, dtype: init_delta_weights(key, shape, self.args)
        init_delta_bias_fn = lambda key, shape, dtype: init_delta_bias(key, shape, self.args)

        # delta_proj projects Δ from delta_rank to d_inner (parameter implicit in bias).
        self.delta_proj = nn.Dense(
            self.args.d_inner,
            use_bias=True,
            kernel_init=init_delta_kernel_fn,
            bias_init=init_delta_bias_fn,
        )

    def setup_A(self) -> None:
        """Set up and initialise the learnable SSM parameter A, using S4D-Real initialisation."""

        # S4D-Real initialisation
        def init_A_log(key: jax.random.PRNGKey, shape: Sequence[int]) -> chex.Array:
            return jnp.log(jnp.tile(jnp.arange(1, shape[1] + 1, dtype=jnp.float32), (shape[0], 1)))

        # Initialise learnable, time-invariant A parameter in log space.
        self.A_log = self.param("A_log", init_A_log, (self.args.d_inner, self.args.d_state))

    def setup_D(self) -> None:
        """Set up and initialise the learnable SSM parameter D."""
        # Initialise learnable, time-invariant D 'skip' parameter.
        self.D = self.param("D", lambda key: jnp.ones(self.args.d_inner))

    def __call__(self, x: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Forward pass."""
        x_and_res = self.in_proj(x)
        x, residual = jnp.split(x_and_res, indices_or_sections=2, axis=-1)

        # Manually causal-pad inputs for convolutional layer.
        padded_x = jnp.concatenate(
            (jnp.zeros((x.shape[0], self.args.d_conv - 1, self.args.d_inner)), x), axis=1
        )
        x = self.conv1d(padded_x)

        x = nn.activation.silu(x)
        x = self.ssm(x)
        x = x * nn.activation.silu(residual)
        x = self.out_proj(x)

        return x

    def recurrent(
        self,
        x: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> Tuple[chex.Array, HiddenState, Buffer]:
        """Recurrent forward pass."""
        x_and_res = self.in_proj(x)
        x, residual = jnp.split(x_and_res, indices_or_sections=2, axis=-1)

        # Retrieve the current and preceding agents from buffer,
        # including causal padding when necessary.
        buffer = FIFOCircularBuffer.add(buffer, x)

        # Pass the correctly padded input through the conv layer.
        x = self.conv1d(buffer)

        x = nn.activation.silu(x)
        x, hidden_state = self.ssm_recurrent(x, hidden_state)
        x = x * nn.activation.silu(residual)
        x = self.out_proj(x)

        return x, hidden_state, buffer

    def ssm(self, x: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Runs the SSM."""
        # Compute ∆ A B C D, the state space parameters.
        #   - A, D are input independent
        #   - ∆, B, C are input-dependent
        A_matrix = -jnp.exp(self.A_log)
        delta_B_C = self.x_proj(x)
        (delta, B, C) = jnp.split(
            delta_B_C, [self.args.delta_rank, self.args.d_state + self.args.delta_rank], axis=-1
        )
        delta = nn.activation.softplus(self.delta_proj(delta))
        x = self.selective_scan(x, delta, A_matrix, B, C)
        return x

    def ssm_recurrent(
        self, x: Union[chex.Array, Tuple[chex.Array, chex.Array]], hidden_state: HiddenState
    ) -> Tuple[chex.Array, HiddenState]:
        """Runs the SSM."""
        # Compute ∆ A B C D, the state space parameters.
        #   - A, D are input independent
        #   - ∆, B, C are input-dependent
        A_matrix = -jnp.exp(self.A_log)
        delta_B_C = self.x_proj(x)
        (delta, B, C) = jnp.split(
            delta_B_C, [self.args.delta_rank, self.args.d_state + self.args.delta_rank], axis=-1
        )
        delta = nn.activation.softplus(self.delta_proj(delta))
        x, hidden_state = self.recurrent_selective_scan(x, delta, A_matrix, B, C, hidden_state)
        return x, hidden_state

    def selective_scan(
        self,
        x: chex.Array,
        delta: chex.Array,
        A: chex.Array,
        B: chex.Array,
        C: chex.Array,
    ) -> chex.Array:
        """
        Does the selective scan algorithm.

        This a (discretised) version of the classic discrete state space formula:
            h(t + 1) = Ah(t) + Bx(t)
            y(t)     = Ch(t) + Dx(t)
        except B and C (and Δ) are input-dependent.
        """

        @jax.vmap
        def associative_operation(
            C_prev: Tuple[chex.Array, chex.Array], C_now: Tuple[chex.Array, chex.Array]
        ) -> Tuple[chex.Array, chex.Array]:
            """
            A single iteration of Mamba's sequential SSM operation
            written as an associative operation, using the operator ∙.
            C_i = [a_i, b_i]
            C_i ∙ C_j = [a_j * a_i, a_j * b_i + b_j]
            """
            (a_prev, b_prev) = C_prev
            (a_now, b_now) = C_now
            a_next = a_now * a_prev
            b_next = a_now * b_prev + b_now
            C_next = (a_next, b_next)
            return C_next

        # Discretise continuous parameters (A, B)
        # - A is discretised using zero-order hold (ZOH) discretisation
        #   (see Section 2 Equation 4 in the Mamba paper [1]).
        # - B is discretised using a simplified Euler discretisation instead of ZOH.
        #   From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with
        #   the simplification on B"
        A_bar = jnp.exp(jnp.einsum("b l d, d n -> b l d n", delta, A))
        B_bar_x = jnp.einsum("b l d, b l n, b l d -> b l d n", delta, B, x)

        # Perform the selective scan.
        input_tuples = (A_bar, B_bar_x)
        (_, hidden_state) = jax.lax.associative_scan(associative_operation, input_tuples, axis=1)

        y = jnp.einsum("b l d n, b l n -> b l d", hidden_state, C)
        y = y + x * self.D

        return y

    def recurrent_selective_scan(
        self,
        x: chex.Array,
        delta: chex.Array,
        A: chex.Array,
        B: chex.Array,
        C: chex.Array,
        hidden_state: HiddenState,
    ) -> Tuple[chex.Array, HiddenState]:
        A_bar = jnp.exp(jnp.einsum("b l d, d n -> b l d n", delta, A))
        B_bar_x = jnp.einsum("b l d, b l n, b l d -> b l d n", delta, B, x)

        hidden_state = A_bar * hidden_state + B_bar_x

        y = jnp.einsum("b l d n, b l n -> b l d", hidden_state, C)
        y = y + x * self.D

        return y, hidden_state
    
########### Mamba module Forward and Reverse in Encoder ##############

class BiMamba(Mamba):
    """Full Mamba model with bidirectionality."""

    def setup(self) -> None:
        self.args = ModelArgs(
            self.num_agents,
            self.d_model,
            self.d_state,
            self.d_conv,
            self.delta_rank,
        )
        self.outer_norm = nn.LayerNorm()
        self.block = BiResidualBlock(self.args)

class BiResidualBlock(ResidualBlock):
    """
    Simple block wrapping a bidirectional Mamba block with normalisation and residual connection.
    """

    def setup(self) -> None:
        self.block = BiMambaBlock(self.args)
        self.norm = nn.RMSNorm()

class BiMambaBlock(MambaBlock):
    """A bidirectional Mamba block."""

    def __call__(self, x: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Forward pass of a bidirectional Mamba block."""
        # Process in the forward direction.
        x = self.one_directional_pass(x)
        # Process in the reverse direction.
        x_reverse = self.one_directional_pass(jnp.flip(x, axis=1))

        # Combine the two processed outputs.
        x_reverse = jnp.flip(x_reverse, axis=1)
        x = x * x_reverse

        return x

    def one_directional_pass(self, x: chex.Array) -> chex.Array:
        """Run an input sequence through a Mamba block in one direction."""
        x_and_res = self.in_proj(x)
        x, residual = jnp.split(x_and_res, indices_or_sections=2, axis=-1)

        # Manually causal-pad inputs for convolutional layer.
        padded_x = jnp.concatenate(
            (jnp.zeros((x.shape[0], self.args.d_conv - 1, self.args.d_inner)), x), axis=1
        )
        x = self.conv1d(padded_x)

        x = nn.activation.silu(x)
        x = self.ssm(x)
        x = x * nn.activation.silu(residual)
        x = self.out_proj(x)

        return x

########### Mamba module cross attention in Decoder ##############

class CrossMamba(Mamba):
    """Full Mamba model with 'cross-attention'."""

    def setup(self) -> None:
        self.args = ModelArgs(
            self.num_agents,
            self.d_model,
            self.d_state,
            self.d_conv,
            self.delta_rank,
        )
        self.outer_norm = nn.LayerNorm()
        self.block = CrossResidualBlock(self.args)

    def __call__(self, x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Applies the module, passing in both input sequences."""
        x1_x2 = self.block(x1_x2)
        # Unpack both sequences.
        (x1, x2) = x1_x2
        x1 = self.outer_norm(x1)

        return x1

    def recurrent(
        self,
        x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> Tuple[chex.Array, HiddenState, Buffer]:
        """Applies the module, passing in both input sequences."""
        x1_x2, hidden_state, buffer = self.block.recurrent(x1_x2, hidden_state, buffer)
        # # Unpack both sequences.
        (x1, x2) = x1_x2
        x1 = self.outer_norm(x1)

        return x1, hidden_state, buffer

class CrossResidualBlock(ResidualBlock):
    """Simple block wrapping a CrossMamba block with normalisation and residual connection."""

    def setup(self) -> None:
        self.block = CrossMambaBlock(self.args)
        self.norm1 = nn.RMSNorm()
        self.norm2 = nn.RMSNorm()

    def __call__(
        self, x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]]
    ) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Forward pass."""
        # Unpack both input sequences.
        (x1, x2) = x1_x2

        residual = x1
        x1 = self.block((self.norm1(x1), self.norm2(x2)))

        return (x1 + residual, x2)

    def recurrent(
        self,
        x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> Union[
        Tuple[chex.Array, HiddenState, Buffer],
        Tuple[Tuple[chex.Array, chex.Array], HiddenState, Buffer],
    ]:
        """Forward pass."""
        # Unpack both input sequences.
        (x1, x2) = x1_x2

        residual = x1
        x1, hidden_state, buffer = self.block.recurrent(
            (self.norm1(x1), self.norm2(x2)), hidden_state, buffer
        )

        return (x1 + residual, x2), hidden_state, buffer

class CrossMambaBlock(MambaBlock):
    """
    A single Mamba block, modified to allow selectivity from both input sequences:
        - B and Δ depend on the the source sequence (x1).
        - C depends on the target sequence (x2).
    """

    def setup(self) -> None:
        # Set up projections for input entering and exiting the block.
        self.setup_input_and_output_projs()

        # Set up the 1D convolutional layer.
        self.setup_conv_layer()

        # Set up the SSM parameters.
        self.setup_A()
        self.setup_D()
        self.setup_delta()
        # x_proj takes in `x` and outputs the input-specific Δ and B.
        self.x_proj = nn.Dense(self.args.delta_rank + self.args.d_state, use_bias=False)
        # C_proj takes in the target sequence and outputs the input-specific C.
        self.C_proj = nn.Dense(self.args.d_state, use_bias=False)

    def __call__(self, x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Forward pass."""
        # Unpack both input sequences.
        (x1, x2) = x1_x2

        x_and_res = self.in_proj(x1)
        x1, residual = jnp.split(x_and_res, indices_or_sections=2, axis=-1)

        # Manually causal-pad inputs for convolutional layer.
        padded_x1 = jnp.concatenate(
            (jnp.zeros((x1.shape[0], self.args.d_conv - 1, self.args.d_inner)), x1), axis=1
        )
        x1 = self.conv1d(padded_x1)

        x1 = nn.activation.silu(x1)
        x1 = self.ssm((x1, x2))
        x1 = x1 * nn.activation.silu(residual)
        x1 = self.out_proj(x1)

        return x1

    def recurrent(
        self,
        x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
        buffer: Buffer,
    ) -> Tuple[chex.Array, HiddenState, Buffer]:
        """Forward pass."""
        # Unpack both input sequences.
        (x1, x2) = x1_x2

        x_and_res = self.in_proj(x1)
        x1, residual = jnp.split(x_and_res, indices_or_sections=2, axis=-1)

        # Retrieve the current and preceding agents from buffer,
        # including causal padding when necessary.
        buffer = FIFOCircularBuffer.add(buffer, x1)

        # Pass the correctly padded input through the conv layer.
        x1 = self.conv1d(buffer)

        x1 = nn.activation.silu(x1)
        x1, hidden_state = self.ssm_recurrent((x1, x2), hidden_state)
        x1 = x1 * nn.activation.silu(residual)
        x1 = self.out_proj(x1)

        return x1, hidden_state, buffer

    def ssm(self, x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]]) -> chex.Array:
        """Runs the SSM."""
        # Unpack both input sequences.
        (x1, x2) = x1_x2

        # Compute ∆ A B C D, the state space parameters.
        #   - A, D are input independent
        #   - ∆, B, C are input-dependent
        A_matrix = -jnp.exp(self.A_log)
        delta_B = self.x_proj(x1)
        (delta, B) = jnp.split(delta_B, [self.args.delta_rank], axis=-1)
        C = self.C_proj(x2)
        delta = nn.activation.softplus(self.delta_proj(delta))
        x1 = self.selective_scan(x1, delta, A_matrix, B, C)
        return x1

    def ssm_recurrent(
        self,
        x1_x2: Union[chex.Array, Tuple[chex.Array, chex.Array]],
        hidden_state: HiddenState,
    ) -> Tuple[chex.Array, HiddenState]:
        """Runs the SSM."""
        # Unpack both input sequences.
        (x1, x2) = x1_x2

        # Compute ∆ A B C D, the state space parameters.
        #   - A, D are input independent
        #   - ∆, B, C are input-dependent
        A_matrix = -jnp.exp(self.A_log)
        delta_B = self.x_proj(x1)
        (delta, B) = jnp.split(delta_B, [self.args.delta_rank], axis=-1)
        C = self.C_proj(x2)
        delta = nn.activation.softplus(self.delta_proj(delta))
        x1, hidden_state = self.recurrent_selective_scan(x1, delta, A_matrix, B, C, hidden_state)
        return x1, hidden_state

########### Mamba Basic Types #################
class MambaHiddenStates(NamedTuple):
    """
    Hidden states for a Mamba decoder's recurrent
    self-attention and cross-attention blocks.
    """

    self_hidden_state: HiddenState
    cross_hidden_state: HiddenState

class MambaBuffers(NamedTuple):
    """
    Circular FIFO buffers for a Mamba decoder's
    recurrent self-attention and cross-attention blocks.
    This is needed to correctly pad single-agent entries.
    """

    self_buffer: Buffer
    cross_buffer: Buffer

class Params(NamedTuple):
    """Parameters of an actor critic network."""

    # 定义一个名为Params的类，继承自NamedTuple，用于存储Actor-Critic网络的参数
    actor_params: FrozenDict
    # actor_params是一个FrozenDict类型的属性，用于存储Actor网络的参数
    # FrozenDict是一种不可变的字典类型，确保参数在创建后不会被修改
    critic_params: FrozenDict

class OptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState

class LearnerState(NamedTuple):
    """State of the learner."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    value_norm_params: ValueNormParams

class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict

class FIFOCircularBuffer:
    @staticmethod
    def init(buffer_shape: Tuple) -> Buffer:
        """Initialise the buffer with zeros."""
        return jnp.zeros(buffer_shape)

    @staticmethod
    def add(buffer: Buffer, value: chex.Array) -> Buffer:
        """
        Add a new value to the buffer (value shape: (batch_size, 1, feature_size))
        and return the updated buffer.
        """
        buffer = jnp.roll(buffer, shift=-1, axis=1)  # Shift left
        buffer = buffer.at[:, -1, :].set(value.squeeze(1))  # Set the new value
        return buffer

    @staticmethod
    def reset(buffer: Buffer) -> Buffer:
        """Reset the buffer with zeros."""
        return jnp.zeros_like(buffer)

class LinearAttention(nn.Module):
    n_embd: int
    num_heads: int = 8

    def setup(self):
        self.head_dim = self.n_embd // self.num_heads
        assert self.head_dim * self.num_heads == self.n_embd, "dim must be divisible by num_heads"
        
        self.q_linear = nn.Dense(self.n_embd)
        self.k_linear = nn.Dense(self.n_embd)
        self.v_linear = nn.Dense(self.n_embd)
        self.out_proj = nn.Dense(self.n_embd)

    def __call__(self, x: chex.Array) -> chex.Array:
        batch_size, seq_len = x.shape[:2]
        
        # 1. 计算QKV
        q = self.q_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))  # [B, H, L, D]
        k = self.k_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))  # [B, H, L, D]
        v = self.v_linear(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))  # [B, H, L, D]

        # 2. 线性注意力计算
        q = nn.relu(q)
        k = nn.relu(k)
        
        # 修正点：保持序列维度
        context = jnp.einsum("bhld,bhkd->bhlk", q, k) / (seq_len * self.head_dim)  # [B, H, L, L]
        
        # 3. 加权求和
        attn_output = jnp.einsum("bhlk,bhkd->bhld", context, v)  # [B, H, L, D]
        attn_output = attn_output.transpose((0, 2, 1, 3))  # [B, L, H, D]
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_embd)  # [B, L, D]
        
        # 4. 输出投影
        return self.out_proj(attn_output)
        
class EncodeBlock(nn.Module):
    n_embd: int
    n_agent: int
    latent_state_dim: int
    conv_dim: int
    delta_rank: int
    masked: bool = False

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.mamba_self_attn = BiMamba(
            self.n_agent,
            self.n_embd,
            self.latent_state_dim,
            self.conv_dim,
            self.delta_rank,
        )
        # 新增线性注意力机制
        self.linear_attn = LinearAttention(self.n_embd, num_heads=8)
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        # Mamba自注意力
        mamba_output = self.mamba_self_attn(x)
        x = self.ln1(x + mamba_output)

        # 线性注意力
        attn_output = self.linear_attn(x)
        x = self.ln2(x + attn_output)

        # MLP层
        x = x + self.mlp(x)
        return x

class Encoder(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    latent_state_dim: int
    conv_dim: int
    delta_rank: int

    def setup(self) -> None:
        self.obs_encoder = nn.Sequential(
            [nn.LayerNorm(), nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
        )
        self.ln = nn.LayerNorm()
        self.blocks = nn.Sequential(
            [
                EncodeBlock(
                    self.n_embd,
                    self.n_agent,
                    self.latent_state_dim,
                    self.conv_dim,
                    self.delta_rank,
                )
                for _ in range(self.n_block)
            ]
        )
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.LayerNorm(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )
        self.act_head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.LayerNorm(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep

class LinearCrossAttention(nn.Module):
    n_embd: int
    num_heads: int
    n_agent: int  # 新增智能体数量参数
    masked: bool = True

    def setup(self):
        self.head_dim = self.n_embd // self.num_heads
        assert self.head_dim * self.num_heads == self.n_embd, "dim must be divisible by num_heads"
        
        # 与SelfAttention一致的初始化方式
        self.q_proj = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))
        self.k_proj = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))
        self.v_proj = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))
        self.out_proj = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))

        # 交叉注意力掩码定义（解码器智能体不能看到编码器未来信息）
        self.mask = jnp.tril(
            jnp.ones((self.n_agent + 1, self.n_agent + 1)),  # 保持与SelfAttention相同的掩码形状
            k=0
        ).reshape(1, 1, self.n_agent + 1, self.n_agent + 1)

    def __call__(self, query: chex.Array, key: chex.Array, value: chex.Array) -> chex.Array:
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key.shape

        # 投影并重塑形状（对齐SelfAttention）
        q = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))  # [B, H, Q_L, D]
        k = self.k_proj(key).reshape(batch_size, kv_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))    # [B, H, KV_L, D]
        v = self.v_proj(value).reshape(batch_size, kv_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))  # [B, H, KV_L, D]

        # 线性注意力特征映射（保持ReLU激活）
        q = nn.relu(q)
        k = nn.relu(k)

        # 计算注意力分数（含缩放因子）
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # 应用交叉智能体掩码
        if self.masked:
            # 动态调整掩码尺寸以适应实际序列长度
            effective_mask = self.mask[:, :, :q_len, :kv_len]
            attn_scores = jnp.where(
                effective_mask == 0, 
                jnp.finfo(jnp.float32).min, 
                attn_scores
            )

        # 注意力权重计算
        attn_weights = nn.softmax(attn_scores, axis=-1)

        # 聚合值向量
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.transpose((0, 2, 1, 3))  # [B, Q_L, H, D]
        attn_output = attn_output.reshape(batch_size, q_len, self.n_embd)

        return self.out_proj(attn_output)

class MaskedLinearDecodeBlock(nn.Module):
    n_embd: int
    n_head: int
    n_agent: int  # 新增参数
    masked: bool = True

    def setup(self) -> None:
        # 保持与SelfAttention解码器相同的LayerNorm结构
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()
        
        # 使用修正后的线性注意力模块
        self.self_attn = LinearCrossAttention(self.n_embd, self.n_head, self.n_agent, self.masked)
        self.cross_attn = LinearCrossAttention(self.n_embd, self.n_head, self.n_agent, self.masked)  # 交叉注意力通常无需掩码
        
        # MLP扩展比例与标准Transformer对齐
        self.mlp = nn.Sequential([
            nn.Dense(4 * self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
            nn.gelu,
            nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
        ])

    def __call__(self, x: chex.Array, enc_out: chex.Array) -> chex.Array:
        # 自注意力（带掩码）
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))
        
        # 交叉注意力（无掩码）
        x = x + self.cross_attn(self.ln2(x), self.ln2(enc_out), self.ln2(enc_out))
        
        # MLP（标准前馈结构）
        x = x + self.mlp(self.ln3(x))
        return x

class DecodeBlock(nn.Module):
    n_embd: int
    n_agent: int
    latent_state_dim: int
    conv_dim: int
    delta_rank: int
    masked: bool = True

    def setup(self) -> None:
        self.mamba_self_attn = Mamba(
            self.n_agent,
            self.n_embd,
            self.latent_state_dim,
            self.conv_dim,
            self.delta_rank,
        )
        
        self.mamba_cross_attn = CrossMamba(
            self.n_agent,
            self.n_embd,
            self.latent_state_dim,
            self.conv_dim,
            self.delta_rank,
        )
        self.ln = nn.LayerNorm()
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
            ],
        )

        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()

    def __call__(self, x: chex.Array, rep_enc: chex.Array) -> chex.Array:
        # x is the action encodings, rep_enc is the obs encodings
        return self.parallel(x, rep_enc)

    def parallel(self, x: chex.Array, rep_enc: chex.Array) -> chex.Array:
        x_new = self.mamba_self_attn(x)
        x = self.ln1(x + x_new)
        x_new = self.mamba_cross_attn((x, rep_enc))
        x = self.ln2(rep_enc + x_new)
        x = self.ln3(x + self.mlp(x))
        return x

    def recurrent(
        self,
        x: chex.Array,  # single shifted action
        rep_enc: chex.Array,  # single obs encoding
        hidden_states: MambaHiddenStates,
        buffers: MambaBuffers,
    ) -> Tuple[chex.Array, MambaHiddenStates, MambaBuffers]:
        (self_hs, cross_hs) = hidden_states
        (self_buffer, cross_buffer) = buffers

        x_new, self_hs, self_buffer = self.mamba_self_attn.recurrent(x, self_hs, self_buffer)
        x = self.ln1(x + x_new)
        x_new, cross_hs, cross_buffer = self.mamba_cross_attn.recurrent(
            (x, rep_enc), cross_hs, cross_buffer
        )
        x = self.ln2(rep_enc + x_new)
        x = self.ln3(x + self.mlp(x))

        hidden_states = MambaHiddenStates(self_hs, cross_hs)
        buffers = MambaBuffers(self_buffer, cross_buffer)

        return x, hidden_states, buffers

class Decoder(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    latent_state_dim: int
    conv_dim: int
    delta_rank: int
    action_space_type: str = "discrete"
    n_head = 8

    def setup(self) -> None:
        if self.action_space_type == "discrete":
            self.action_encoder = nn.Sequential(
                [
                    nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                ],
            )
        else:
            self.action_encoder = nn.Sequential(
                [nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
            )
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Always initialize log_std but set to None for discrete action spaces
        # This ensures the attribute exists but signals it should not be used.
        if self.action_space_type == "discrete":
            self.log_std = None

        self.obs_encoder = nn.Sequential(
            [nn.LayerNorm(), nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
        )
        self.ln = nn.LayerNorm()

        self.blocks = [
            DecodeBlock(
                self.n_embd,
                self.n_agent,
                self.latent_state_dim,
                self.conv_dim,
                self.delta_rank,
                name=f"cross_attention_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.LayerNorm(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )
        # n_head = 1 固定写入

    def __call__(self, action: chex.Array, obs_rep: chex.Array, obs: chex.Array) -> chex.Array:
        return self.parallel(action, obs_rep, obs)

    def parallel(self, action: chex.Array, obs_rep: chex.Array, obs: chex.Array) -> chex.Array:
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Need to loop here because the input and output of the blocks are different.
        # Blocks take an action embedding and observation encoding as input but only give the cross
        # attention output as output.
        
        for block in self.blocks:
            x = block(x, obs_rep)
        
        logit = self.head(x)
        return logit

    def recurrent(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        obs: chex.Array,
        hidden_states: MambaHiddenStates,
        buffers: MambaBuffers,
    ) -> Tuple[chex.Array, MambaHiddenStates, MambaBuffers]:
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        for i, block in enumerate(self.blocks):
            # Index out current block's hidden states and buffers.
            block_hidden_states = jax.tree_util.tree_map(
                lambda x, i=i: x[:, i, :, :, :], hidden_states
            )  # (batch, 1, d_inner, d_latent)
            block_buffers = jax.tree_util.tree_map(
                lambda x, i=i: x[:, i, :, :], buffers
            )  # (batch, d_conv, d_inner)

            x, block_hidden_states_new, block_buffers_new = block.recurrent(
                x,
                obs_rep,
                block_hidden_states,
                block_buffers,
            )

            # Update the current block's hidden states and buffers.
            self_hs = hidden_states[0].at[:, i, :, :, :].set(block_hidden_states_new[0])
            cross_hs = hidden_states[1].at[:, i, :, :, :].set(block_hidden_states_new[1])
            hidden_states = MambaHiddenStates(self_hs, cross_hs)
            self_buffer = buffers[0].at[:, i, :, :].set(block_buffers_new[0])
            cross_buffer = buffers[1].at[:, i, :, :].set(block_buffers_new[1])
            buffers = MambaBuffers(self_buffer, cross_buffer)

        logit = self.head(x)

        return logit, hidden_states, buffers

class HybridMambaNetwork(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_agent: int
    latent_state_dim: int
    conv_dim: int
    delta_rank: int
    action_space_type: str = "discrete"

    def setup(self) -> None:

        if self.action_space_type not in ["discrete", "continuous"]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        self.encoder = Encoder(
            self.obs_dim,
            self.action_dim,
            self.n_block,
            self.n_embd,
            self.n_agent,
            self.latent_state_dim,
            self.conv_dim,
            self.delta_rank,
        )
        self.decoder = Decoder(
            self.obs_dim,
            self.action_dim,
            self.n_block,
            self.n_embd,
            self.n_agent,
            self.latent_state_dim,
            self.conv_dim,
            self.delta_rank,
            self.action_space_type,
        )
        if self.action_space_type == "discrete":
            self.autoregressive_act = discrete_autoregressive_act
            # self.autoregressive_act = setup_discrete_autoregressive_act_scan(
            #     self.action_dim, self.n_agent
            # )
            self.parallel_act = discrete_parallel_act

        else:
            self.autoregressive_act = continuous_autoregressive_act
            # self.autoregressive_act = setup_continuous_autoregressive_scan(
            #     self.action_dim, self.n_agent
            # )
            self.parallel_act = continuous_parallel_act

    def __call__(
        self,
        obs: chex.Array,
        action: chex.Array,
        legal_actions: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        v_loc, obs_rep = self.encoder(obs)

        action_log, entropy = self.parallel_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            obs=obs,
            batch_size=obs.shape[0],
            action=action,
            n_agent=self.n_agent,
            action_dim=self.action_dim,
            legal_actions=legal_actions,
        )

        return action_log, v_loc, entropy

    def get_actions(
        self, obs: chex.Array, legal_actions: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        # obs: (batch, n_agent, obs_dim)
        # obs_rep: (batch, n_agent, n_embd)
        # v_loc: (batch, n_agent, 1)

        v_loc, obs_rep = self.encoder(obs)
        output_action, output_action_log, raw_action = self.autoregressive_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            obs=obs,
            batch_size=obs.shape[0],
            n_agent=self.n_agent,
            action_dim=self.action_dim,
            conv_dim=self.conv_dim,
            num_features=self.n_embd,
            d_latent_state=self.latent_state_dim,
            n_block=self.n_block,
            legal_actions=legal_actions,
            key=key,
        )
        return output_action, output_action_log, v_loc, raw_action

    def get_values(self, obs: chex.Array) -> chex.Array:
        v_loc, _ = self.encoder(obs)
        return v_loc

def discrete_parallel_act(
    decoder: Decoder,
    obs_rep: chex.Array,  # (batch, n_agent, n_embd)
    obs: chex.Array,  # (batch, n_agent, obs_dim)
    action: chex.Array,  # (batch, n_agent, 1)
    batch_size: int,  # (, )
    n_agent: int,  # (, )
    action_dim: int,  # (, )
    legal_actions: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    one_hot_action = jax.nn.one_hot(action, action_dim)  # (batch, n_agent, action_dim)
    shifted_action = jnp.zeros(
        (batch_size, n_agent, action_dim + 1)
    )  # (batch, n_agent, action_dim + 1)
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    # This should look like this for all batches:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0]]
    shifted_action = shifted_action.at[:, 1:, 1:].set(one_hot_action[:, :-1, :])
    # If the action is:
    # [[2],
    #  [1],
    #  [0]]

    # The one hot action is:
    # [[0, 0, 1, 0, 0],
    #  [0, 1, 0, 0, 0],
    #  [1, 0, 0, 0, 0]]

    # The shifted action will be:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0, 0]]

    logit = decoder(shifted_action, obs_rep, obs)  # (batch, n_agent, action_dim)

    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    distribution = distrax.Categorical(logits=masked_logits)
    action_log_prob = distribution.log_prob(action)
    action_log_prob = jnp.expand_dims(action_log_prob, axis=-1)  # (batch, n_agent, 1)
    entropy = jnp.expand_dims(distribution.entropy(), axis=-1)  # (batch, n_agent, 1)

    return action_log_prob, entropy

def continuous_parallel_act(
    decoder: Decoder,
    obs_rep: chex.Array,  # (batch, n_agent, n_embd)
    obs: chex.Array,  # (batch, n_agent, obs_dim)
    action: chex.Array,  # (batch, n_agent, 1 <- should prob be action_dim)
    batch_size: int,  # (, )
    n_agent: int,  # (, )
    action_dim: int,  # (, )
    legal_actions: chex.Array,
) -> Tuple[chex.Array, chex.Array]:

    shifted_action = jnp.zeros((batch_size, n_agent, action_dim))  # (batch, n_agent, action_dim)

    shifted_action = shifted_action.at[:, 1:, :].set(action[:, :-1, :])

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = jax.nn.softplus(decoder.log_std)

    distribution = tfd.MultivariateNormalDiag(loc=act_mean, scale_diag=action_std)
    action_log_prob = distribution.log_prob(action)
    action_log_prob -= jnp.sum(
        2.0 * (jnp.log(2.0) - action - jax.nn.softplus(-2.0 * action)), axis=-1
    )  # (batch, n_agent, 1)
    entropy = distribution.entropy()

    return action_log_prob, entropy

def discrete_autoregressive_act(
    decoder: Decoder,
    obs_rep: chex.Array,
    obs: chex.Array,
    batch_size: int,
    n_agent: int,
    action_dim: int,
    conv_dim: int,
    num_features: int,
    d_latent_state: int,
    n_block: int,
    legal_actions: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, None]:
    shifted_action = jnp.zeros(
        (batch_size, n_agent, action_dim + 1)
    )  # (batch, n_agent, action_dim + 1)
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    # This should look like:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0]]
    output_action = jnp.zeros((batch_size, n_agent, 1))
    output_action_log = jnp.zeros_like(output_action)
    # both have shape (batch, n_agent, 1)

    # Initialise convolutional buffers to zeroes for first agent,
    # accounting for Mamba's expansion factor.
    buffer_shape = (batch_size, n_block, conv_dim, num_features * 2)
    buffer_self = FIFOCircularBuffer.init(buffer_shape)
    buffer_cross = FIFOCircularBuffer.init(buffer_shape)
    init_buffers = (buffer_self, buffer_cross)
    buffers = jax.tree_util.tree_map(jnp.copy, init_buffers)
    buffers = MambaBuffers(*buffers)

    # Initialise Mamba decoder hidden states
    hidden_state_shape = (batch_size, n_block, 1, num_features * 2, d_latent_state)
    _hs = jnp.zeros(hidden_state_shape)
    init_hs = (_hs[:], _hs[:])
    hidden_states = jax.tree_util.tree_map(jnp.copy, init_hs)
    hidden_states = MambaHiddenStates(*hidden_states)

    for i in range(n_agent):
        logit, hidden_states, buffers = decoder.recurrent(
            shifted_action[:, i : i + 1, :],
            obs_rep[:, i : i + 1, :],
            obs,
            hidden_states,
            buffers,
        )
        logit = logit.squeeze(1)  # (batch, action_dim)
        masked_logits = jnp.where(
            legal_actions[:, i, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        distribution = distrax.Categorical(logits=masked_logits)
        key, sample_key = jax.random.split(key)
        action, action_log = distribution.sample_and_log_prob(seed=sample_key)  # both just integers

        output_action = output_action.at[:, i, :].set(
            jnp.expand_dims(action, axis=-1)
        )  # (batch, n_agent, 1)
        output_action_log = output_action_log.at[:, i, :].set(
            jnp.expand_dims(action_log, axis=-1)
        )  # (batch, n_agent, 1)

        update_shifted_action = i + 1 < n_agent
        shifted_action = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_action=shifted_action: shifted_action.at[
                :, i + 1, 1:
            ].set(jax.nn.one_hot(action, action_dim)),
            lambda shifted_action=shifted_action: shifted_action,
        )

        # An example of a shifted action:
        # [[1, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 1],
        #  [0, 0, 0, 1, 0, 0]]

        # Assuming the actions where [4, 2, 4]
        # An important note, the shifted actions are not really relevant,
        # they are just used to act autoregressively.

    return output_action.astype(jnp.int32), output_action_log, None

def continuous_autoregressive_act(
    decoder: Decoder,
    obs_rep: chex.Array,  # (batch, n_agent, n_embd)
    obs: chex.Array,  # (batch, n_agent, obs_dim)
    batch_size: int,  # (, )
    n_agent: int,  # (, )
    action_dim: int,  # (, )
    conv_dim: int,
    num_features: int,
    d_latent_state: int,
    n_block: int,
    legal_actions: Union[chex.Array, None],
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:

    shifted_action = jnp.zeros((batch_size, n_agent, action_dim))  # (batch, n_agent, action_dim)
    output_action = jnp.zeros((batch_size, n_agent, action_dim))
    raw_output_action = jnp.zeros((batch_size, n_agent, action_dim))
    output_action_log = jnp.zeros((batch_size, n_agent))

    # Initialise convolutional buffers to zeroes for first agent,
    # accounting for Mamba's expansion factor.
    buffer_shape = (batch_size, n_block, conv_dim, num_features * 2)
    buffer_self = FIFOCircularBuffer.init(buffer_shape)
    buffer_cross = FIFOCircularBuffer.init(buffer_shape)
    init_buffers = (buffer_self, buffer_cross)
    buffers = jax.tree_util.tree_map(jnp.copy, init_buffers)
    buffers = MambaBuffers(*buffers)

    # Initialise Mamba decoder hidden states
    hidden_state_shape = (batch_size, n_block, 1, num_features * 2, d_latent_state)
    _hs = jnp.zeros(hidden_state_shape)
    init_hs = (_hs[:], _hs[:])
    hidden_states = jax.tree_util.tree_map(jnp.copy, init_hs)
    hidden_states = MambaHiddenStates(*hidden_states)

    for i in range(n_agent):
        act_mean, hidden_states, buffers = decoder.recurrent(
            shifted_action[:, i : i + 1, :],
            obs_rep[:, i : i + 1, :],
            obs,
            hidden_states,
            buffers,
        )
        act_mean = act_mean.squeeze(1)  # (batch, action_dim)
        action_std = jax.nn.softplus(decoder.log_std)

        key, sample_key = jax.random.split(key)

        distribution = tfd.MultivariateNormalDiag(loc=act_mean, scale_diag=action_std)
        raw_action = distribution.sample(seed=sample_key)
        action_log = distribution.log_prob(raw_action)
        action_log -= jnp.sum(
            2.0 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)), axis=-1
        )  # (batch, 1)
        action = jnp.tanh(raw_action)

        raw_output_action = raw_output_action.at[:, i, :].set(raw_action)
        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        update_shifted_action = i + 1 < n_agent
        shifted_action = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_action=shifted_action: shifted_action.at[
                :, i + 1, :
            ].set(action),
            lambda shifted_action=shifted_action: shifted_action,
        )

    return output_action, output_action_log, raw_output_action

def setup_discrete_autoregressive_act_scan(action_dim: int, n_agent: int) -> Callable:
    def scan_step(decoder: Decoder, carry: Tuple, x: chex.Numeric) -> Tuple[Tuple, None]:
        (
            shifted_action,
            output_action,
            output_action_log,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            key,
        ) = carry

        i = x  # x is the agent index
        indexed_action = shifted_action[:, i, jnp.newaxis, ...]
        indexed_obs_rep = obs_rep[:, i, jnp.newaxis, ...]

        logit, hidden_states, buffers = decoder.recurrent(
            indexed_action, indexed_obs_rep, obs, hidden_states, buffers
        )
        logit = logit.squeeze(1)
        masked_logits = jnp.where(
            legal_actions[:, i, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        distribution = distrax.Categorical(logits=masked_logits)
        key, sample_key = jax.random.split(key)
        action, action_log = distribution.sample_and_log_prob(seed=sample_key)

        output_action = output_action.at[:, i, :].set(jnp.expand_dims(action, axis=-1))
        output_action_log = output_action_log.at[:, i, :].set(jnp.expand_dims(action_log, axis=-1))

        update_shifted_action = i + 1 < n_agent
        shifted_action = jax.lax.cond(
            update_shifted_action,
            lambda: shifted_action.at[:, i + 1, 1:].set(jax.nn.one_hot(action, action_dim)),
            lambda: shifted_action,
        )

        return (
            shifted_action,
            output_action,
            output_action_log,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            key,
        ), None

    scan = nn.scan(scan_step, variable_broadcast="params", split_rngs={"params": False}, unroll=16)

    def discrete_autoregressive_act_scan(
        decoder: Decoder,
        obs_rep: chex.Array,
        obs: chex.Array,
        batch_size: int,
        n_agent: int,
        action_dim: int,
        conv_dim: int,
        num_features: int,
        d_latent_state: int,
        n_block: int,
        legal_actions: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, None]:
        shifted_action = jnp.zeros((batch_size, n_agent, action_dim + 1))
        shifted_action = shifted_action.at[:, 0, 0].set(1)
        output_action = jnp.zeros((batch_size, n_agent, 1))
        output_action_log = jnp.zeros_like(output_action)

        # Initialise convolutional buffers to zeroes for first agent,
        # accounting for Mamba's expansion factor.
        buffer_shape = (batch_size, n_block, conv_dim, num_features * 2)
        buffer_self = FIFOCircularBuffer.init(buffer_shape)
        buffer_cross = FIFOCircularBuffer.init(buffer_shape)
        init_buffers = (buffer_self, buffer_cross)
        buffers = jax.tree_util.tree_map(jnp.copy, init_buffers)
        buffers = MambaBuffers(*buffers)

        # Initialise Mamba decoder hidden states
        hidden_state_shape = (batch_size, n_block, 1, num_features * 2, d_latent_state)
        _hs = jnp.zeros(hidden_state_shape)
        init_hs = (_hs[:], _hs[:])
        hidden_states = jax.tree_util.tree_map(jnp.copy, init_hs)
        hidden_states = MambaHiddenStates(*hidden_states)

        initial_carry = (
            shifted_action,
            output_action,
            output_action_log,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            key,
        )
        agents = jnp.arange(n_agent)
        (
            shifted_action,
            output_action,
            output_action_log,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            _,
        ), _ = scan(decoder, initial_carry, agents)

        return output_action.astype(jnp.int32), output_action_log, None

    return discrete_autoregressive_act_scan

def setup_continuous_autoregressive_scan(action_dim: int, n_agent: int) -> Callable:
    def scan_step(decoder: Decoder, carry: Tuple, x: chex.Numeric) -> Tuple[Tuple, None]:
        (
            shifted_action,
            output_action,
            output_action_log,
            raw_output_action,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            key,
        ) = carry

        i = x  # x is the agent index
        indexed_action = shifted_action[:, i, jnp.newaxis, ...]
        indexed_obs_rep = obs_rep[:, i, jnp.newaxis, ...]

        act_mean, hidden_states, buffers = decoder.recurrent(
            indexed_action,
            indexed_obs_rep,
            obs,
            hidden_states,
            buffers,
        )
        act_mean = act_mean.squeeze(1)  # (batch, action_dim)
        action_std = jax.nn.softplus(decoder.log_std)

        key, sample_key = jax.random.split(key)

        distribution = tfd.MultivariateNormalDiag(loc=act_mean, scale_diag=action_std)
        raw_action = distribution.sample(seed=sample_key)
        action_log = distribution.log_prob(raw_action)
        action_log -= jnp.sum(
            2.0 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)), axis=-1
        )  # (batch, 1)
        action = jnp.tanh(raw_action)

        output_action = output_action.at[:, i, :].set(action)
        raw_output_action = raw_output_action.at[:, i, :].set(raw_action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        update_shifted_action = i + 1 < n_agent
        shifted_action = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_action=shifted_action: shifted_action.at[
                :, i + 1, :
            ].set(action),
            lambda shifted_action=shifted_action: shifted_action,
        )

        return (
            shifted_action,
            output_action,
            output_action_log,
            raw_output_action,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            key,
        ), None

    scan = nn.scan(scan_step, variable_broadcast="params", split_rngs={"params": False}, unroll=16)

    def continuous_autoregressive_act_scan(
        decoder: Decoder,
        obs_rep: chex.Array,
        obs: chex.Array,
        batch_size: int,
        n_agent: int,
        action_dim: int,
        conv_dim: int,
        num_features: int,
        d_latent_state: int,
        n_block: int,
        legal_actions: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        shifted_action = jnp.zeros(
            (batch_size, n_agent, action_dim)
        )  # (batch, n_agent, action_dim)
        output_action = jnp.zeros((batch_size, n_agent, action_dim))
        raw_output_action = jnp.zeros((batch_size, n_agent, action_dim))
        output_action_log = jnp.zeros((batch_size, n_agent))

        # Initialise convolutional buffers to zeroes for first agent,
        # accounting for Mamba's expansion factor.
        buffer_shape = (batch_size, n_block, conv_dim, num_features * 2)
        buffer_self = FIFOCircularBuffer.init(buffer_shape)
        buffer_cross = FIFOCircularBuffer.init(buffer_shape)
        init_buffers = (buffer_self, buffer_cross)
        buffers = jax.tree_util.tree_map(jnp.copy, init_buffers)
        buffers = MambaBuffers(*buffers)

        # Initialise Mamba decoder hidden states
        hidden_state_shape = (batch_size, n_block, 1, num_features * 2, d_latent_state)
        _hs = jnp.zeros(hidden_state_shape)
        init_hs = (_hs[:], _hs[:])
        hidden_states = jax.tree_util.tree_map(jnp.copy, init_hs)
        hidden_states = MambaHiddenStates(*hidden_states)

        initial_carry = (
            shifted_action,
            output_action,
            output_action_log,
            raw_output_action,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            key,
        )

        agents = jnp.arange(n_agent)
        (
            shifted_action,
            output_action,
            output_action_log,
            raw_output_action,
            obs_rep,
            obs,
            legal_actions,
            hidden_states,
            buffers,
            _,
        ), _ = scan(decoder, initial_carry, agents)

        return output_action, output_action_log, raw_output_action

    return continuous_autoregressive_act_scan
