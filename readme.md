<!--- 
One-paragraph summary: 
- What problem, what you contribute, what the key result is. 
--->
# SDEmatching
This repo implements the 2025 paper SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations, by Bartosh, Vetrov, and Naesseth [1](#ref-1). The paper shows that it is possible to train a Stochastich Differential Equation (SDE) to fit data in a simulation free manner, by scor-ematching between the trained model and a variational data approximation.

<!--- 
Key idea (theory) in 1–2 screens:
- Definitions/assumptions (minimal).
- Main result(s) as a theorem/claim + intuition.
- A short “why it matters / when it fails” section.
- One figure if you have it.
--->
## Score matchin SDEs against variational approximation


## Implementation
<!--- 
Results:
- Table/plot with the headline outcome.
- Link to results/ artifacts.
--->


## Ideas for further research
As explained above, the main obstacle with this model is how to construct the variational process. I have tried with a transformer model, which works for non-hidden states, but collapses when working with hidden latent dimensions. 

In contrast, I have used a Gaussian Process, which generates an estimate of the mean and variance, but also generates an estimate of the derivative of the mean, and the variance of the derivative of the mean. This works very well for a simple physics system with restrictions to the drift matrix.

However, none of these generalizes. In the original paper, the authors use the ODE-RNN introduced by Rubanova et al. in [2](#ref-2).

My suggestion for further research would be to use a 1D convolutional network that maps from `observation_dim * series_length` to `latent_dim * series_length` and then perhaps use a deep kernel [3](#ref-3) for further refinement.

<!--- 
How to run:
- Installation
- Minimal command to reproduce a quick run
- Full reproduction (optional)
- Hardware/time caveats (brief, factual)
--->

<!--- 
Repo layout
- What’s in src/, evaluation/, etc.
--->

<!--- 
Citation / attribution
- BibTeX or a short citation line if it maps to a paper/report.
--->
In order to use the SDEMatching package do the following:

1) Move to the folder where you want the code
2) Clone this repository: ```git clone https://github.com/simoneiriksson/SDEMatching.git```
3) If you prefer, create new python environment: ```python -m venv .venv```
4) And activate the new python environment: ```source .venv/bin/activate```
5) Install the package into your active environment: ```pip install -e .```

Now you should be able to run any of the ```.py``` and ```.ipynb``` notebooks in the folder named ```doodles``` or ```to_Ole```. Entry at own risk.


## References

1. <a id="ref-1"></a> 
Bartosh, G., Vetrov, D. &amp; Naesseth, C.A.. (2025). SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations. <i>Proceedings of the 42nd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 267:3054-3070 Available from https://proceedings.mlr.press/v267/bartosh25a.html.


2. <a id="ref-2"></a> 
Yulia Rubanova, Ricky T. Q. Chen, and David Duvenaud. 2019. Latent ODEs for irregularly-sampled time series. Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, Article 478, 5320–5330.


3. <a id="ref-3"></a> 
Wilson, A.G., Hu, Z., Salakhutdinov, R. &amp; Xing, E.P.. (2016). Deep Kernel Learning. <i>Proceedings of the 19th International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 51:370-378 Available from https://proceedings.mlr.press/v51/wilson16.html.

