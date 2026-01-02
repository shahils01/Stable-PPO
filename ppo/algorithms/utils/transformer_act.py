import torch
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent
from torch.nn import functional as F


def discrete_decentralized_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    logit = decoder(None, None, obs)

    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
    action_log = distri.log_prob(action)

    output_action = action.unsqueeze(-1)
    output_action_log = action_log.unsqueeze(-1)
    
    return output_action, output_action_log


def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]

        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs)
        
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs, batch_size, action_dim, tpdv,
                                  deterministic=False, obs_image=None):
    act_mean = decoder(obs, obs_image)
    action_std = torch.sigmoid(decoder.log_std) * 0.5

    distri = Normal(act_mean, action_std)
    action = act_mean if deterministic else distri.sample()
    action_log = distri.log_prob(action)

    return action, action_log


def continuous_parallel_act(decoder, obs, action, batch_size, action_dim, tpdv, obs_image=None):
    act_mean = decoder(obs, obs_image)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    
    distri = Normal(act_mean, action_std)

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy


def gmm_entropy_mc_per_dim(distri, mus, sigmas, weights, n=1024, eps=1e-8):
    """
    Returns per-dimension entropy estimate: [B, A]
    """
    # Sample from the true joint GMM
    x = distri.sample((n,))          # [n, B, A]
    x = x.detach()                   # no grad through sampling

    # action_std = torch.sigmoid(sigmas) * 0.5

    # Component log probs per dim
    # Normal broadcast: [n, B, K, A]
    logp_x_given_k = Normal(
        mus.unsqueeze(0), 
        sigmas.unsqueeze(0)
    ).log_prob(x.unsqueeze(2))

    # Log weights: [1, B, K, 1]
    log_w = weights.clamp_min(eps).log().unsqueeze(0).unsqueeze(-1)

    # Per-dim mixture log prob: log sum_k w_k p_k(x_d)
    logp_per_dim = torch.logsumexp(log_w + logp_x_given_k, dim=2)  # [n, B, A]

    # Entropy estimate
    return -logp_per_dim.mean(dim=0)  # [B, A]



def gmm_log_prob_per_dim(action, mus, sigmas, weights, eps=1e-8):
    """
    Returns per-action-dimension log prob under a diagonal-cov GMM:
      action:  [B, A]
      mus:     [B, K, A]
      sigmas:  [B, K, A]
      weights: [B, K]
    Output:
      logp:    [B, A]
    """

    # action_std = torch.sigmoid(sigmas) * 0.5

    # Component log probs per dim: [B, K, A]
    comp = Normal(mus, sigmas)
    logp_x_given_k = comp.log_prob(action.unsqueeze(1))  # [B, K, A]

    # Mixture weights in log-space: [B, K, 1] for broadcasting over A
    log_w = (weights.clamp_min(eps)).log().unsqueeze(-1)  # [B, K, 1]

    # Mixture log prob per dim: log sum_k w_k * p_k(x_d)
    # This is a "per-dimension mixture"; note discussion below.
    logp_per_dim = torch.logsumexp(log_w + logp_x_given_k, dim=1)  # [B, A]
    return logp_per_dim


def continuous_moe_act(mus, sigmas, weights):
    # action_std = torch.sigmoid(sigmas) * 0.5

    comp = Independent(Normal(mus, sigmas), reinterpreted_batch_ndims=1)
    mix = Categorical(probs=weights)
    distri = MixtureSameFamily(mix, comp)

    action = distri.sample()
    action_log = gmm_log_prob_per_dim(action, mus, sigmas, weights)
    
    return action, action_log.unsqueeze(-1)


def continuous_moe_eval(mus, sigmas, weights, actions):
    # action_std = torch.sigmoid(sigmas) * 0.5

    comp = Independent(Normal(mus, sigmas), reinterpreted_batch_ndims=1)
    mix = Categorical(probs=weights)
    distri = MixtureSameFamily(mix, comp)

    action_log = gmm_log_prob_per_dim(actions, mus, sigmas, weights)
    entropy = gmm_entropy_mc_per_dim(distri, mus, sigmas, weights, 100)

    # gate_entropy = mix.entropy()

    # Load Balancing Loss (The "Aux Loss")
    # Encourages using ALL experts across the batch
    # Calculate average weight assigned to each expert across the batch
    mean_weights = weights.mean(dim=0) # Shape: [num_experts]
    # Calculate entropy of this average distribution
    marginal_entropy = -torch.sum(mean_weights * torch.log(mean_weights + 1e-8))
    
    return action_log.unsqueeze(-1), entropy, marginal_entropy
