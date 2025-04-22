# eval_demo.py

import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf
from mava.utils import make_env as environments
from mava.evaluator import make_eval_fns
from mava.networks.hmma_network import HybridMambaNetwork

SEED = 42

TASK_LIST = [
    '2s3z',
    '3s5z',
    '5m_vs_6m',
    '10m_vs_11m',
    '27m_vs_30m',
    '3s5z_vs_3s6z',
    '3s_vs_5z',
    '6h_vs_8z',
    'smacv2_5_units',
    'smacv2_10_units',
    'smacv2_20_units'
]


@hydra.main(config_path="/root/mam-code/mava/configs", config_name="default_mtm.yaml", version_base="1.2")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    for task_name in TASK_LIST:
        print(f"\n[•] Running evaluation for task: {task_name} | seed: {SEED}")

        cfg.system.seed = SEED
        cfg.env.name = "smax"
        cfg.env.scenario.task_name = task_name

        env, eval_env = environments.make(cfg)

        obs_spec = env.observation_spec().generate_value()
        dummy_obs = obs_spec.agents_view[None]
        dummy_action = jnp.zeros((1, env.num_agents), dtype=jnp.int32)
        dummy_mask = obs_spec.action_mask[None]

        actor_network = HybridMambaNetwork(
            obs_dim=obs_spec.agents_view.shape[-1],
            action_dim=env.action_dim,
            n_block=cfg.network.actor_network.n_block,
            n_embd=cfg.network.actor_network.n_embd,
            n_agent=env.num_agents,
            latent_state_dim=cfg.network.actor_network.latent_state_dim,
            conv_dim=cfg.network.actor_network.conv_dim,
            delta_rank=cfg.network.actor_network.delta_rank,
        )

        actor_params = actor_network.init(
            jax.random.PRNGKey(cfg.system.seed),
            dummy_obs,
            dummy_action,
            dummy_mask,
        )

        from flax import jax_utils
        actor_params = jax_utils.replicate(actor_params)

        _, evaluator = make_eval_fns(
            eval_env=eval_env,
            network_apply_fn=actor_network.apply,
            config=cfg,
            use_mat=True,
            use_recurrent_net=False,
            mat=True,
        )

        key = jax.random.PRNGKey(cfg.system.seed)
        key, *eval_keys = jax.random.split(key, len(jax.devices()) + 1)
        eval_keys = jnp.stack(eval_keys).reshape(len(jax.devices()), -1)

        evaluator_output = evaluator(actor_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        print(f"[✓] Evaluation complete for task: {task_name}")
        for k, v in evaluator_output.episode_metrics.items():
            print(f"{k}: {float(jnp.mean(v)):.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
