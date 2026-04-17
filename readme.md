<!--- 
One-paragraph summary: 
- What problem, what you contribute, what the key result is. 
--->
# SDEmatching
This repo implements the 2025 paper SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations, by Bartosh, Vetrov, and Naesseth [[1]](#ref-1). The paper shows that it is possible to train a Stochastic Differential Equation (SDE) to fit data in a simulation free manner, by score matching between the trained model and a variational data approximation.

<!--- 
Key idea (theory) in 1–2 screens:
- Definitions/assumptions (minimal).
- Main result(s) as a theorem/claim + intuition.
- A short “why it matters / when it fails” section.
- One figure if you have it.
--->
## Score matching SDEs against variational approximation

The following is paraphrased from the paper [[1]](#ref-1).
We assume the existence of an SDE on the form

$$dz_t = f(z_t, t) dt + \sigma(z_t, t) dW_t$$

with prior distribution $p_0(z_0)$ and emission distribution $p_e(x\vert z_t)$. Here $dW_t$ is a Wiener process. The function $f(z_t, t)$ is called the *drift* and $\sigma(z_t, t)$ is the *diffusion term*, which is function that returns a matrix.

This process generates a set of series of observation, each of which of the form $X=[x_{t_1}, x_{t_2}, ..., x_{t_N}]$. In the following, we just assume that there is one single sampled series of observations, but the implementation works with a set of samples.

The problem, which the paper address is how to estimate the drift $f_\theta$, diffusion $\sigma_\theta$, prior $q_0$ and emission $q_e$ distributions such that the SDE

$$ dz_t = f_\theta(z_t, t) dt + \sigma_\theta(z_t, t) dW_t$$

generates similar data.


This can be done by defining a variational conditional marginal distribution 
that correpsonds to the latent states:
$z_t = F_\phi(\varepsilon, t, X)$, where $\varepsilon\sim \mathcal{N}(0,I)$. 
This transformation implicitly defines the conditional distribution $q_\phi(z_t\vert X)$, 
where $z_t = F_\phi(\varepsilon, t, X)$ is a sample from it.

In this implementation $q_\phi$ is chosen to be Gaussian, conditional on the inputs $t$ and $X$, but this can be excanged with any distribution as long as $F$ is invertible in $\varepsilon$ and differentiable in $t$. Any normalizing flow conditional on $X$ and $t$ will do.

Now define the time derivative of $F_\phi$, given a fixed sample of $\varepsilon$:

$$\bar{f}_{\phi}(z_{t},t,X)=\frac{\partial F_{\phi}(\varepsilon,t,X)}{\partial t}\Big\vert_{\varepsilon=F_{\phi}^{-1}(z_{t},t,X)}.
$$

Starting from $z_t \sim q(z_t\vert X)$ and integrating the ODE 

$$
dz_t = \bar{f}_{\phi}(z_{t},t,X) dt
$$

we then get a sample from the variational marginal distribution $q_\phi(z_t\vert X)$.

We are now interested in minimizing the KL-divergence between $q_\phi(z_t\vert X)$ and $p_\theta(z_t)$ over path measures. By Girsanovs theorem, this is finite if both processes share the same diffusion term $\sigma_\theta$.

Let 
$\sigma^2_\theta(z_t, t)$ be a shorthand for $\sigma_\theta(z_t, t)\sigma_\theta(z_t, t)^\top$.
If we then define 

$$
f_\phi(z_t, t, X) = \bar{f}_{\phi}(z_{t},t,X) + \frac{1}{2}\sigma^2_\theta(z_t, t) \nabla_{z_t} \ln q_\phi(z_t\vert X) + 
\frac{1}{2}\nabla_{z_t}  \sigma^2_\theta(z_t, t),$$

then a result in [[4]](#ref-4) gives us that the SDE defiend by 

$$
dz_t = f_\phi(z_t, t, X) dt + \sigma_\theta(z_t, t) dW_t
$$

*also* has the marginal distribution $q_\phi(z_t\vert X)$, regardless of what $\sigma_\theta$ looks like.

Now, if we approximate $f_\phi(z_t, t, X)$ by a neural network $f_\theta(z_t, t)$, then we will in turn have an SDE which also approximately has the same marginal distribution as the variational marginal distribution $q_\phi(z_t\vert X)$.

This means that if we also get an approximation $q_e$ of the emission distribution, then we can couple everything together and calculate an Evidence Lower Bound as 

$$
ELBO(\theta) = \mathcal{L}_{\text{prior}} + \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{rec}},
$$ 

where 

$$\mathcal{L}_{\text{prior}} = D_{KL}(q_\phi(z_0|X) \| p_\theta(z_0))$$
 
$$\mathcal{L}_{\text{rec}} = -\log p_\theta(x_{t_i}|z_{t_i})$$

$$\mathcal{L}_{\text{diff}} = \tfrac{1}{2}\|\sigma_\theta^{-1}(z_t, t)(f_\theta(z_t, t) - f_\phi(z_t, t, X))\|^2$$

## Implementation
<!--- 
Results:
- Table/plot with the headline outcome.
- Link to results/ artifacts.
--->


## Ideas for further research
As explained above, the main obstacle with this model is how to construct the variational process. I have tried with a transformer model, which works for non-hidden states, but collapses when working with hidden latent dimensions. 

In contrast, I have used a Gaussian Process, which generates an estimate of the mean and variance, but also generates an estimate of the derivative of the mean, and the variance of the derivative of the mean. This works very well for a simple physics system with restrictions to the drift matrix.

However, none of these generalizes. In the original paper, the authors use the ODE-RNN introduced by Rubanova et al. in [2](#ref-2). I personally do not like this approach, since it breaks with the simulation free spirit of the SDE matching method.

My suggestion for further research would be to use a 1D convolutional network that maps from `observation_dim * series_length` to `latent_dim * series_length` and then perhaps use a deep kernel [3](#ref-3) for further refinement.

<!--- 
Repo layout
- What’s in src/, evaluation/, etc.
--->

## How to run
<!--- 
How to run:
- Installation
- Minimal command to reproduce a quick run
- Full reproduction (optional)
- Hardware/time caveats (brief, factual)
--->
In order to use the SDEMatching package do the following:

1) Move to the folder where you want the code
2) Clone this repository: ```git clone https://github.com/simoneiriksson/SDEMatching.git```
3) If you prefer, create new python environment: ```python -m venv .venv```
4) And activate the new python environment: ```source .venv/bin/activate```
5) Install the package into your active environment: ```pip install -e .```

## References
<!--- 
Citation / attribution
- BibTeX or a short citation line if it maps to a paper/report.
--->

1. <a id="ref-1"></a> 
Bartosh, G., Vetrov, D. &amp; Naesseth, C.A.. (2025). SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations. <i>Proceedings of the 42nd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 267:3054-3070 Available from https://proceedings.mlr.press/v267/bartosh25a.html.


2. <a id="ref-2"></a> 
Yulia Rubanova, Ricky T. Q. Chen, and David Duvenaud. 2019. Latent ODEs for irregularly-sampled time series. Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, Article 478, 5320–5330.


3. <a id="ref-3"></a> 
Wilson, A.G., Hu, Z., Salakhutdinov, R. &amp; Xing, E.P.. (2016). Deep Kernel Learning. <i>Proceedings of the 19th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 51:370-378 Available from https://proceedings.mlr.press/v51/wilson16.html.

4. <a id="ref-4"></a> 
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.
