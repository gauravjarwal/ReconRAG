# Neural Network Training

## Gradient Descent

Neural networks are trained by minimising a loss function L(θ) with respect to the model parameters θ. The primary optimisation algorithm is gradient descent: parameters are updated in the direction of the negative gradient of the loss.

    θ ← θ - η * ∇_θ L(θ)

where η is the learning rate. Computing the gradient over the full dataset (batch gradient descent) is expensive for large datasets. In practice, stochastic gradient descent (SGD) uses a single sample or mini-batch of B samples per update.

## Backpropagation

Backpropagation is the algorithm used to efficiently compute gradients of the loss with respect to all parameters. It applies the chain rule of calculus in reverse, propagating error signals from the output layer back through each layer to compute the gradient at each weight.

For a composition of functions f = f_L ∘ f_{L-1} ∘ ... ∘ f_1, the gradient of the loss with respect to the input of layer l is:

    δ^l = ((W^{l+1})^T δ^{l+1}) ⊙ σ'(z^l)

where σ' is the derivative of the activation function and z^l is the pre-activation value. The weight gradient is then δ^l * (a^{l-1})^T.

## Activation Functions

Activation functions introduce non-linearity, enabling networks to approximate complex functions.

**ReLU (Rectified Linear Unit)**: f(x) = max(0, x). ReLU is the most widely used activation in deep networks. It avoids the vanishing gradient problem for positive inputs and is computationally efficient. However, neurons can "die" if they always receive negative inputs.

**Leaky ReLU**: f(x) = max(αx, x) where α is a small constant (e.g., 0.01). Allows a small gradient for negative inputs, preventing dead neurons.

**GELU (Gaussian Error Linear Unit)**: f(x) = x * Φ(x), where Φ is the standard normal CDF. Used in BERT and GPT models. Smoother than ReLU.

**Sigmoid**: f(x) = 1 / (1 + e^{-x}). Squashes outputs to (0,1). Suffers from vanishing gradients for large |x|. Used in binary classification output layers.

**Softmax**: Converts a vector of logits to a probability distribution. Used in multi-class classification output layers.

## Optimisers

**SGD with Momentum**: adds a velocity term that accumulates gradient directions, dampening oscillations and accelerating convergence in the consistent gradient direction.

    v ← μv - η∇L;  θ ← θ + v

**Adam (Adaptive Moment Estimation)**: maintains per-parameter first moment (mean) m and second moment (uncentred variance) v estimates of the gradient. It is widely used in practice and typically requires less learning rate tuning than SGD.

    m ← β₁m + (1-β₁)∇L
    v ← β₂v + (1-β₂)(∇L)²
    θ ← θ - η * m̂ / (√v̂ + ε)

where m̂ and v̂ are bias-corrected estimates. Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8.

**AdamW**: Adam with decoupled weight decay. Weight decay is applied directly to the parameters rather than through the gradient, correcting a subtle flaw in the original Adam implementation. AdamW is the standard optimiser for training large language models.

## Learning Rate Schedules

A constant learning rate is rarely optimal. Common schedules:

- **Warmup + cosine decay**: the learning rate linearly increases from 0 to peak over the first few thousand steps (warmup), then follows a cosine decay curve. Used in GPT-3 and most modern LLM training runs.
- **Linear decay**: learning rate decreases linearly from peak to 0 over training.
- **Step decay**: learning rate is multiplied by a factor (e.g., 0.1) every fixed number of epochs.
- **Cyclical learning rates**: learning rate oscillates between a minimum and maximum, allowing the model to escape sharp minima.

## Regularisation

**L2 regularisation (weight decay)**: adds λ||θ||² to the loss, penalising large weights and preventing over-fitting.

**Dropout**: during training, each neuron is randomly set to zero with probability p (typically 0.1–0.5). At inference, all neurons are active and their outputs are scaled by (1-p). Dropout acts as an ensemble of exponentially many sub-networks.

**Batch normalisation**: normalises the activations of each layer to zero mean and unit variance within each mini-batch, then applies learned scale and shift parameters. Stabilises training and allows higher learning rates. Used heavily in CNNs but less so in Transformers.

**Layer normalisation**: normalises across the feature dimension rather than the batch dimension. Preferred for Transformers and recurrent networks because it is independent of batch size.

**Early stopping**: monitor validation loss during training; stop when it stops improving. Prevents over-fitting to the training set.

## Initialisation

Poor weight initialisation can cause gradients to vanish or explode during the first few backward passes, preventing training from starting.

**Xavier / Glorot initialisation**: initialises weights from a uniform or normal distribution with variance 2 / (n_in + n_out). Designed to maintain the variance of activations and gradients for tanh and sigmoid activations.

**He / Kaiming initialisation**: initialises weights with variance 2 / n_in. Designed for ReLU activations to prevent the variance from halving at each layer.

## Batch Size and Gradient Accumulation

Larger batch sizes produce lower-variance gradient estimates, enabling larger learning rates and faster convergence in terms of wall-clock time. However, very large batches can hurt generalisation (sharp minima).

When GPU memory is insufficient for a large batch, gradient accumulation allows accumulating gradients over multiple forward–backward passes before performing a single parameter update, simulating a larger effective batch size.

## Mixed Precision Training

Modern GPUs (e.g., NVIDIA A100) have dedicated hardware for 16-bit floating point (FP16 or BF16) computations that are 2–8× faster than FP32. Mixed precision training keeps a master copy of weights in FP32 for numerical stability but performs forward and backward passes in FP16. A loss scaling factor prevents FP16 underflow in gradients.

BF16 (Brain Float 16) has the same exponent range as FP32 (reducing the risk of overflow) but fewer mantissa bits than FP16. BF16 has become the default precision for LLM training.

## Gradient Clipping

Exploding gradients — where gradient norms grow very large — are a common problem in deep or recurrent networks. Gradient clipping caps the gradient norm to a maximum value (typically 1.0) before applying the parameter update:

    if ||g|| > threshold:
        g ← g * threshold / ||g||

This is standard practice when training Transformers.

## Common Training Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN from step 1 | Learning rate too high or bad init | Lower LR, use He/Xavier init |
| Loss plateaus immediately | Learning rate too low | Increase LR or add warmup |
| Training loss low, val loss high | Over-fitting | Add dropout, weight decay, or reduce model size |
| Training loss oscillates wildly | Batch size too small | Increase batch size or reduce LR |
| Gradients vanish | Deep network with sigmoid/tanh | Switch to ReLU, add batch norm, use residual connections |
| Gradients explode | Deep network or RNN | Apply gradient clipping |
