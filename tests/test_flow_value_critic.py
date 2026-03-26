import os
import sys
import unittest

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ppo.algorithms.ppo.algorithm.PPO import FlowValueCritic


class FlowValueCriticTests(unittest.TestCase):
    def test_forward_flow_shapes(self):
        critic = FlowValueCritic(obs_dim=4, n_embd=16, num_quants=8, solver_steps=4)
        obs = torch.randn(3, 4)

        quantiles, log_jacobian, path_length, _, _ = critic.forward_flow(obs)

        self.assertEqual(quantiles.shape, (3, 8))
        self.assertEqual(log_jacobian.shape, (3, 8))
        self.assertEqual(path_length.shape, (3, 8))

    def test_log_jacobian_matches_affine_map(self):
        critic = FlowValueCritic(obs_dim=4, n_embd=16, num_quants=8, solver_steps=2)
        base = critic.base_grid.unsqueeze(0).expand(2, -1)
        transported = 2.5 * base + 1.25

        log_jacobian = critic._estimate_log_jacobian(transported, base)
        expected = torch.full_like(log_jacobian, torch.log(torch.tensor(2.5)))

        self.assertTrue(torch.allclose(log_jacobian, expected, atol=1e-4, rtol=1e-4))

    def test_monotonicity_penalty_is_non_negative(self):
        critic = FlowValueCritic(obs_dim=4, n_embd=16, num_quants=8, solver_steps=4)
        obs = torch.randn(2, 4)
        target = torch.sort(torch.randn(2, 8), dim=-1).values

        losses = critic.flow_matching_loss(obs, target)

        self.assertGreaterEqual(losses["flow_monotonicity_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
