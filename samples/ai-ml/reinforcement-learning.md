# Reinforcement Learning

## What Is Reinforcement Learning?

Reinforcement learning (RL) is a branch of machine learning where an agent learns to make decisions by interacting with an environment. The agent takes actions, receives feedback in the form of rewards or penalties, and updates its behaviour to maximise the cumulative reward over time. Unlike supervised learning, RL does not require labelled input–output pairs; the learning signal comes entirely from the environment's reward signal.

Reinforcement learning is the primary technique behind breakthrough systems such as AlphaGo, OpenAI Five (which beat professional Dota 2 players), and ChatGPT's alignment via Reinforcement Learning from Human Feedback (RLHF).

## Core Components

### Agent
The agent is the entity that makes decisions. It observes the current state of the environment and selects an action from its action space.

### Environment
The environment is everything the agent interacts with. After receiving an action, it transitions to a new state and emits a reward. In robotics, the environment is the physical world; in game-playing, it is the game engine.

### State (s)
A state is a representation of the environment at a given time step. States can be fully observable (the agent sees everything relevant) or partially observable (the agent receives only noisy or incomplete observations).

### Action (a)
An action is a decision the agent can take. Action spaces can be discrete (e.g., {up, down, left, right} in a grid world) or continuous (e.g., joint torques in a robotic arm).

### Reward (r)
A scalar signal the environment emits after each action. The agent's goal is to maximise the sum of future rewards, usually discounted by a factor γ ∈ [0,1] to weight near-term rewards more heavily:

    G_t = r_{t+1} + γ*r_{t+2} + γ²*r_{t+3} + ...

### Policy (π)
The policy is the agent's strategy — a mapping from states to actions. A deterministic policy maps each state to a single action; a stochastic policy maps states to probability distributions over actions.

### Value Function
The value function V^π(s) estimates the expected return starting from state s when following policy π. The action-value (Q) function Q^π(s, a) estimates the expected return when taking action a in state s then following π thereafter.

## The Bellman Equation

The value function satisfies the Bellman equation, which expresses the recursive relationship between a state's value and its successor's:

    V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [r(s,a,s') + γ * V^π(s')]

Solving the Bellman equation exactly is only feasible in small, tabular environments. In large or continuous state spaces, value functions are approximated by neural networks.

## Q-Learning and Deep Q-Networks (DQN)

Q-learning is a model-free, off-policy algorithm that learns the optimal action-value function Q*(s,a) directly from experience. The update rule is:

    Q(s,a) ← Q(s,a) + α * [r + γ * max_{a'} Q(s',a') - Q(s,a)]

Deep Q-Networks (DQN), introduced by DeepMind in 2013, approximate Q(s,a) with a neural network. DQN achieved human-level performance on 49 Atari games using only raw pixel inputs. Two key techniques stabilise training:

1. **Experience replay**: transitions (s, a, r, s') are stored in a replay buffer and sampled randomly for updates, breaking temporal correlations.
2. **Target network**: a periodically updated copy of the Q-network is used to compute target values, preventing oscillations.

## Policy Gradient Methods

Policy gradient methods directly optimise the policy by computing the gradient of the expected return with respect to the policy parameters θ:

    ∇_θ J(θ) = E_π [∇_θ log π_θ(a|s) * Q^π(s,a)]

REINFORCE is the simplest policy gradient algorithm: it uses the actual return G_t as an estimate of Q^π(s,a). A baseline (often the state value function) is subtracted to reduce variance.

## Actor-Critic Methods

Actor-critic methods combine a policy (the actor) with a value function (the critic). The actor updates the policy based on advantage estimates from the critic, while the critic updates its value estimate using TD errors. This reduces the variance of policy gradient estimates compared to pure REINFORCE.

**Proximal Policy Optimisation (PPO)** is a widely used actor-critic algorithm that clips the policy gradient update to prevent large policy changes:

    L_CLIP = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

where r_t(θ) = π_θ(a|s) / π_{θ_old}(a|s) is the probability ratio and A_t is the advantage estimate. PPO is used to train the policy in RLHF pipelines.

## Reinforcement Learning from Human Feedback (RLHF)

RLHF is the technique used to align large language models with human preferences. The process has three stages:

1. **Supervised fine-tuning**: the base language model is fine-tuned on high-quality demonstrations.
2. **Reward model training**: human annotators rank pairs of model responses; a reward model is trained to predict human preference scores.
3. **RL fine-tuning**: the language model is optimised using PPO to maximise the reward model's scores while a KL-divergence penalty prevents it from drifting too far from the supervised baseline.

ChatGPT, Claude, and Gemini all use variants of RLHF to improve helpfulness, harmlessness, and honesty.

## Exploration vs. Exploitation

A central challenge in RL is the exploration–exploitation trade-off. The agent must balance exploiting known high-reward actions with exploring new actions that might yield higher rewards.

Common exploration strategies include:
- **ε-greedy**: with probability ε, select a random action; otherwise select the greedy action.
- **Softmax / Boltzmann**: select actions proportional to their Q-values exponentiated by an inverse-temperature parameter.
- **Upper Confidence Bound (UCB)**: add an exploration bonus proportional to uncertainty to action values.
- **Intrinsic motivation**: reward the agent for visiting novel states (used in environments with sparse external rewards).

## Multi-Agent Reinforcement Learning

In multi-agent RL (MARL), multiple agents interact in a shared environment. Agents may cooperate (maximise a shared reward), compete (zero-sum), or operate in mixed settings. OpenAI Five and AlphaStar (StarCraft II) are examples of MARL systems trained through self-play, where agents improve by competing against copies of themselves.
