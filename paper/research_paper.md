# Distribution-Free Uncertainty Quantification in Machine Learning: A Theoretical and Empirical Comparison of Conformal Prediction, Bayesian Credible Intervals, and Parametric Bootstrap in R

**Yuvraj Chandak**  
*School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India*  
*Register No.:* `25BCE2561` | *Email:* `yuvraj.chandak2025@vitstudent.ac.in`  

*Under the guidance of:*  
**Dr. Veermani S**  
*Project Supervisor & Professor, Department of Mathematics*  
*Vellore Institute of Technology, Vellore, Tamil Nadu, India* | *Email:* `veermani.s@vit.ac.in`  

---

## Abstract
And point predictions made by statistical machine learning models get across no information regarding prediction variance, skewness, or tail risk, broadly speaking. Classical uncertainty quantification frameworks, including Ordinary Least Squares (OLS) prediction intervals, parametric bootstrap resamples, and conjugate Bayesian Normal-Inverse-Gamma trustworthy sets, obtain predictive distributions under the assumptions of homoscedasticity, correct likelihood specification, and thin-tailed Gaussian mismatches, in real-world scenarios. Diverse from that, distribution-free conformal prediction guarantees finite-sample slight coverage, under the solitary need of exchangeability. Indeed, this paper is comparing five predictive interval methods in R, namely parametric OLS, parametric bootstrap, Bayesian Normal-Inverse-Gamma, Split Conformal Prediction, and Conformalized Quantile Regression (CQR). When error distributions stray from Gaussian homoscedasticity, parametric intervals over-cover ($92.7\%, 93.0\%$ at token $90\%$) due to sample variance inflation from outliers. In fact, an examination of the theoretical mechanics of each approach was did, explaining why fixed-scale bootstrap simulations come together to normal rather than Student-$t$ predictive quantiles and assessing the conservative finite-sample conformal bound $1 - \alpha \le \mathbb{P}(Y_{n+1} \in \hat{C}_n) \le 1 - \alpha + \frac{1}{n_2 + 1}$. Across $M = 100$ Monte Carlo replications ($N = 1,500$) under Gaussian, heavy-tailed Student-$t_3$, and heteroscedastic noise, alongside the historical Boston housing benchmark, we gauge tested coverage calibration, interval width, and proper Winkler scores. Indeed, even as a modest width overhead is incurred, under Gaussian noise by constant-width conformal prediction, CQR modulates interval width with local conditional dispersion, decreasing proper Winkler scores under the heteroscedasticity. When the dust settles up, concrete trade-offs are tackled entailing sample-splitting budgets, cross-split dependence in repeated benchmark partitions, and the assumptions needed for valid coverage under the covariate shift. But empirical coverage consistent with the small $90\%$ target, across all regimes is upheld by conformal methods.

**Index Terms:** Conformal Prediction, Exchangeability, Bayesian Inference, Parametric Bootstrap, Uncertainty Quantification, Finite-Sample Coverage, Non-Parametric Statistics, Machine Learning in R, in the main.

**MSC 2020 Subject Classifications:** Primary 62G15. Secondary 62F15, 68T05, 62J02.

---

## 1. Introduction & Problem Formulation

In applied machine learning, regularized estimators such as Random Forests, Gradient Boosted Decision Trees, and neural networks are widely used for predictive modeling. Given an observed sample $\mathcal{D}_n = \{(X_i, Y_i)\}_{i=1}^n$ drawn from an unknown joint probability measure $\mathcal{P}_{X,Y}$ on $\mathcal{X} \times \mathbb{R}$, where $\mathcal{X} \subseteq \mathbb{R}^d$, typical training minimizes empirical risk:
$$
\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n \mathcal{L}\big(Y_i, f(X_i)\big),
$$
for a prescribed loss function $\mathcal{L}: \mathbb{R} \times \mathbb{R} \to [0, \infty)$. Although empirical risk minimization yields estimators with small generalization error, scalar point forecasts $\hat{Y}_{n+1} = \hat{f}_n(X_{n+1})$ convey no information regarding prediction variance, skewness, or tail risk. Uncertainty quantification (UQ) addresses this limitation by constructing a data-driven prediction interval $\hat{C}_n(X_{n+1}) \subset \mathbb{R}$ that satisfies, for a target significance level $\alpha \in (0, 1)$:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \ge 1 - \alpha.
$$

### 1.1 Aleatoric and Epistemic Uncertainty Decomposition
Formally, total predictive variability decomposes into two orthogonal components:
$$
\text{Var}(Y \mid X = x) = \underbrace{\sigma^2(x)}_{\text{Aleatoric Uncertainty}} + \underbrace{\text{Var}_{\mathcal{D}_n}\big(\hat{f}_n(x)\big)}_{\text{Epistemic Uncertainty}}.
$$
1. Aleatoric Uncertainty: Reflects irreducible stochastic deviation in the data-generating system. In turn, aleatoric noise can't be lowered by increasing sample size, at least empirically, and when $\text{Var}(\epsilon \mid X = x) = \sigma^2$ is uniform across $\mathcal{X}$ likely, the model is typically homoscedastic. When $\sigma^2(x)$ differs with $x$, it's heteroscedastic, in actual application. Aleatoric variability corresponds to the conditional error distribution $\epsilon \mid X \sim \mathcal{P}_{\epsilon \mid X}$ in regression models $Y = \mu(X) + \epsilon$ with $\mathbb{E}[\epsilon \mid X] = 0$.
2. Epistemic Uncertainty: What epistemic uncertainty includes is parameter estimation variance, model misspecification, and regularization bias, commonly decaying at rate $\mathcal{O}_p(n^{-1/2})$ under the parametric regularity, at least for now. Represents reducible variance resulting from finite sample size and incomplete parameter knowledge.

Standard inferential frameworks, such as Ordinary Least Squares (OLS) prediction intervals, parametric bootstrap resamples, and Bayesian posterior predictive distributions, quantify both variance components under the joint assumption that the parametric likelihood, error family, and variance structures are correctly recommended, at least in principle. When data show heavy tails, asymmetry, or heteroscedasticity, these classic methods suffer from systematic coverage miscalibration, in many respects.

### 1.2 Scope and Objectives
While foundational works established the theory of split conformal prediction (Papadopoulos et al., 2002. Lei et al., 2018) and Conformalized Quantile Regression (Romano et al., 2019), existing literature often checks conformal methods in isolation or against uncalibrated baseline quantiles. The goal of this paper isn't to introduce new conformal theorems, but to transparently gauge in R across the five established UQ paradigms under the same data splits:

1. Comparative Framework: All i mean is it's, We apply and compare parametric OLS, parametric bootstrap resampling, conjugate Bayesian Normal-Inverse-Gamma inference, Split Conformal Random Forests, and CQR within a integrated R workflow, at least in practice.
2. Stress-Testing Under Misspecification: Plain and simple, the breakdown behavior of standard parametric and Bayesian intervals under the heavy-tailed Student-$t_3$ errors ($\kappa = \infty$) is assessed and asymmetric heteroscedastic noise ($\sigma(X_1) = 1 + 2|X_1|$), explaining how sample variance inflation warps coverage, all things considered.
3. Empirical Calibration and Scoring: We assess coverage calibration across $M = 100$ independent Monte Carlo trials using replication-level Student's $t$-tests, paired with strictly proper Winkler scores to penalize uninformative interval width and boundary violations simultaneously.
4. Sample-Splitting Budget Analysis: What's more, the operational trade-off is checked between training budget $n_1$ and calibration budget $n_2$, and clearly account for the differing training sample sizes available to full-sample parametric models versus split-conformal models, to a significant degree.

We test whether empirical coverage rates stray from token 90% via $H_{0, \text{cov}}: \mathbb{E}[\text{ECR}] = 1 - \alpha$ versus $H_{1, \text{cov}}: \mathbb{E}[\text{ECR}] \ne 1 - \alpha$ for the check of these methods, and test paired differences in Winkler scores $S_\alpha$ against the parametric baseline, on the whole.

The remainder of this paper is organized as follows. Section 7 is concluding the paper. Section 2 introduces the mathematical foundations of parametric, bootstrap, Bayesian and conformal uncertainty quantification, including a step-by-step proof of Theorem 1, in the main. Section 4 is offering the self-contained R implementation script, and section 5 details the tested findings, statistical tests, and graphical investigation, at least for now. Section 6 theoretically is examining on sample partitioning, conditional coverage limits, and covariate shift. And section 3 is outlining the experimental method, noise regimes, real-world benchmark, and proper scoring rules.

---

## 2. Mathematical & Probabilistic Foundations

Let $(X_1, Y_1), \dots, (X_n, Y_n) \in \mathcal{X} \times \mathbb{R}$ denote a sample of random pairs on probability space $(\Omega, \mathcal{F}, \mathbb{P})$. We consider the non-linear regression setting:
$$
Y_i = \mu(X_i) + \epsilon_i, \quad i = 1, \dots, n,
$$
where $\mu: \mathcal{X} \to \mathbb{R}$ is the true regression function and $\epsilon_i$ denotes disturbance with $\mathbb{E}[\epsilon_i \mid X_i] = 0$. We seek a prediction interval $\hat{C}_n(x_{n+1}) = [\hat{L}(x_{n+1}), \, \hat{U}(x_{n+1})]$ for a test query $X_{n+1} = x_{n+1}$ such that:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \ge 1 - \alpha, \quad \alpha \in (0, 1).
$$

---

### 2.1 Classical Ordinary Least Squares Prediction Intervals

The classical linear parametric model posits:
$$
Y = X \beta + \epsilon, \quad \epsilon \sim \mathcal{N}_n(0, \, \sigma^2 I_n),
$$
As well, $\epsilon$ is an independent Gaussian error vector. What's more, where $X \in \mathbb{R}^{n \times p}$ is a design matrix of full column rank ($\text{rank}(X) = p < n$), $\beta \in \mathbb{R}^p$ is an unknown coefficient vector, all things considered.

The Ordinary Least Squares (OLS) estimator is:
$$
\hat{\beta} = (X^\top X)^{-1} X^\top Y,
$$
with sampling distribution:
$$
\hat{\beta} \sim \mathcal{N}_p\big(\beta, \, \sigma^2 (X^\top X)^{-1}\big).
$$
Meaning, the unbiased estimator of $\sigma^2$ is the residual mean squared error:
$$
\hat{\sigma}^2 = \frac{1}{n - p} \|Y - X \hat{\beta}\|_2^2 = \frac{\text{RSS}}{n - p},
$$
fulfilling Cochran's Theorem:
$$
\frac{(n - p) \hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n-p},
$$
independently of $\hat{\beta}$.

For an unobserved query point $(X_{n+1}, Y_{n+1})$ where $X_{n+1} = x_{n+1} \in \mathbb{R}^p$ and $Y_{n+1} = x_{n+1}^\top \beta + \epsilon_{n+1}$ with $\epsilon_{n+1} \sim \mathcal{N}(0, \sigma^2)$ autonomous of $\mathcal{D}_n$ maybe, the point prediction is usually $\hat{Y}_{n+1} = x_{n+1}^\top \hat{\beta}$. The prediction error is:
$$
e_{n+1} = Y_{n+1} - \hat{Y}_{n+1} = \epsilon_{n+1} - x_{n+1}^\top (\hat{\beta} - \beta).
$$
Because $\epsilon_{n+1}$ and $\hat{\beta}$ are independent Gaussian variables, the conditional mean and variance of $e_{n+1}$ are:
$$
\mathbb{E}[e_{n+1} \mid X, x_{n+1}] = 0,
$$
$$
\begin{aligned}
\text{Var}(e_{n+1} \mid X, x_{n+1}) &= \text{Var}(\epsilon_{n+1}) + \text{Var}\big(x_{n+1}^\top \hat{\beta}\big) \\
&= \sigma^2 + x_{n+1}^\top \text{Cov}(\hat{\beta}) x_{n+1} \\
&= \sigma^2 \Big(1 + x_{n+1}^\top (X^\top X)^{-1} x_{n+1}\Big).
\end{aligned}
$$
Replacing $\sigma^2$ with $\hat{\sigma}^2$, the studentized prediction pivot is:
$$
T = \frac{Y_{n+1} - x_{n+1}^\top \hat{\beta}}{\hat{\sigma} \sqrt{1 + x_{n+1}^\top (X^\top X)^{-1} x_{n+1}}} \sim t_{n-p},
$$
in many respects. In response, the exact $(1 - \alpha)$ parametric prediction interval is:
$$
\hat{C}_{\text{param}}(x_{n+1}) = \left[ x_{n+1}^\top \hat{\beta} \pm t_{1-\alpha/2, \, n-p} \, \hat{\sigma} \sqrt{1 + h_{n+1}} \right],
$$
where the use of $x_{n+1}$ is meant by $h_{n+1} = x_{n+1}^\top (X^\top X)^{-1} x_{n+1}$.

---

### 2.2 Parametric Bootstrap Prediction Intervals

The parametric bootstrap offers a computational framework for building prediction intervals by copying from the fitted parametric distribution. Or of counting on analytical pivot distributions perhaps, it resamples synthetic outcomes directly from the estimated model:

1. Parameter Estimation: Fit the base model to get $\hat{\beta} = (X^\top X)^{-1} X^\top Y$ and residual variance $\hat{\sigma}^2 = \frac{1}{n-p} \|Y - X\hat{\beta}\|_2^2$.
2. Bootstrap Resampling: For $b = 1, \dots, B$ (where $B$ means the number of bootstrap replications):
   - Draw synthetic training innovations from the fitted Gaussian distribution:
     $$
     \epsilon^{*(b)} \sim \mathcal{N}_n(0, \, \hat{\sigma}^2 I_n).
     $$
   - Form bootstrap response vectors:
     $$
     Y^{*(b)} = X \hat{\beta} + \epsilon^{*(b)}.
     $$
   - Re-estimate the coefficient vector:
     $$
     \hat{\beta}^{*(b)} = (X^\top X)^{-1} X^\top Y^{*(b)}.
     $$
   - For test vector $x_{\text{test}}$, draw a new test innovation representing future aleatoric noise:
     $$
     \epsilon_{\text{test}}^{*(b)} \sim \mathcal{N}(0, \, \hat{\sigma}^2).
     $$
   - Compute the simulated test response:
     $$
     Y_{\text{test}}^{*(b)} = x_{\text{test}}^\top \hat{\beta}^{*(b)} + \epsilon_{\text{test}}^{*(b)}.
     $$
3. Quantile Interval Construction: The empirical $(1 - \alpha)$ bootstrap prediction interval is:
   $$
   \hat{C}_{\text{boot}}(x_{\text{test}}) = \left[ \hat{q}_{\alpha/2}\left( \{Y_{\text{test}}^{*(b)}\}_{b=1}^B \right), \quad \hat{q}_{1-\alpha/2}\left( \{Y_{\text{test}}^{*(b)}\}_{b=1}^B \right) \right].
   $$

#### Asymptotic Convergence of the Residual Bootstrap
As $\hat{\sigma}$ is held fixed across the bootstrap draws rather than re-estimated or sampled from a posterior, conditional on $(\hat{\beta}, \hat{\sigma})$ the imitated predictor is satisfying $\hat{\beta}^{*(b)} \sim \mathcal{N}_p(\hat{\beta}, \hat{\sigma}^2(X^\top X)^{-1})$ and $Y_{\text{test}}^{*(b)} \sim \mathcal{N}(x_{\text{test}}^\top \hat{\beta}, \hat{\sigma}^2(1 + h_{\text{test}}))$, all things considered. And thus, as $B \to \infty$, the tested percentiles are converging almost surely to the quantiles of this conditional Gaussian distribution: $[x_{\text{test}}^\top \hat{\beta} \pm z_{1-\alpha/2} \hat{\sigma} \sqrt{1 + h_{\text{test}}}]$, to a significant degree. When the dust settles, this fixed-scale procedure converges to the plug-in normal predictive interval rather than the exact Student-$t$ interval. And thus, getting the exact Student-$t$ distribution via bootstrap would need re-estimating $\hat{\sigma}^{*(b)}$ on each bootstrap sample and constructing studentized important statistics. Seeing as $z_{1-\alpha/2} < t_{1-\alpha/2, n-p}$, the bootstrap interval is slightly narrower and shows modest finite-sample under-coverage, under Gaussian errors (as recorded in Table 1).

When the true disturbance distribution violates normality:
- Heavy Tails: When $\epsilon \sim t_3$, drawing bootstrap errors from $\mathcal{N}(0, \hat{\sigma}^2)$ forces thin rapid tails, while $\hat{\sigma}^2$ is generally inflated by sample outliers probably, the Gaussian generative system crashes to encapsulate the true tail probability mass, guiding to unpredictable coverage, in many respects.
- Heteroscedasticity: The parametric bootstrap assumes homoscedastic variance $\hat{\sigma}^2$ across all $x$. Under variable variance $\sigma^2(x)$, drawing homoscedastic test errors misallocates uncertainty across $\mathcal{X}$.

---

### 2.3 Bayesian Posterior Predictive Credible Intervals

In the Bayesian formulation, uncertainty is quantified through the posterior predictive distribution. Let $\theta = (\beta, \sigma^2) \in \Theta = \mathbb{R}^p \times \mathbb{R}_+$ denote model parameters with prior density $\pi(\theta)$. Given observed data $\mathcal{D}_n = (X, Y)$, the Gaussian likelihood is:
$$
\mathcal{L}(\mathcal{D}_n \mid \theta) = (2\pi \sigma^2)^{-n/2} \exp\left( -\frac{1}{2\sigma^2} \|Y - X\beta\|_2^2 \right).
$$
The posterior parameter distribution follows from Bayes' Theorem:
$$
p(\theta \mid \mathcal{D}_n) \propto \mathcal{L}(\mathcal{D}_n \mid \theta) \, \pi(\theta).
$$
For an unobserved query $x_{n+1}$, the **Posterior Predictive Distribution** integrates over parameter uncertainty:
$$
p(y_{n+1} \mid x_{n+1}, \mathcal{D}_n) = \int_{\Theta} p(y_{n+1} \mid x_{n+1}, \theta) \, p(\theta \mid \mathcal{D}_n) \, d\theta,
$$
in real-world scenarios.

#### Analytical Derivation under Conjugate Normal-Inverse-Gamma Prior
Think about the conjugate Normal-Inverse-Gamma (NIG) prior:
$$
\beta \mid \sigma^2 \sim \mathcal{N}_p\big(\mu_0, \, \sigma^2 \Lambda_0^{-1}\big), \quad \sigma^2 \sim \text{Inv-Gamma}(a_0, \, b_0),
$$
with joint prior density:
$$
\pi(\beta, \sigma^2) \propto (\sigma^2)^{-(a_0 + p/2 + 1)} \exp\left( -\frac{1}{\sigma^2} \left[ b_0 + \frac{1}{2}(\beta - \mu_0)^\top \Lambda_0 (\beta - \mu_0) \right] \right),
$$
at least in practice. Multiplying by the likelihood and completing the quadratic form in $\beta$, the posterior distribution remains Normal-Inverse-Gamma:
$$
\beta \mid \sigma^2, \mathcal{D}_n \sim \mathcal{N}_p\big(\mu_n, \, \sigma^2 \Lambda_n^{-1}\big), \quad \sigma^2 \mid \mathcal{D}_n \sim \text{Inv-Gamma}(a_n, \, b_n),
$$
with modified hyperparameters:
$$
\begin{aligned}
\Lambda_n &= X^\top X + \Lambda_0, \\
\mu_n &= \Lambda_n^{-1} \big(X^\top Y + \Lambda_0 \mu_0\big), \\
a_n &= a_0 + \frac{n}{2}, \\
b_n &= b_0 + \frac{1}{2} \left( Y^\top Y + \mu_0^\top \Lambda_0 \mu_0 - \mu_n^\top \Lambda_n \mu_n \right),
\end{aligned}
$$
in many respects. Under standard objective Jeffreys prior limits ($\Lambda_0 \to 0, a_0 \to 0, b_0 \to 0$), the posterior moments reduce to the OLS values:
$$
\mu_n = \hat{\beta} = (X^\top X)^{-1} X^\top Y, \quad \Lambda_n = X^\top X, \quad a_n = \frac{n - p}{2}, \quad b_n = \frac{1}{2} \text{RSS},
$$
at least for now. Conditioning on $\sigma^2$:
$$
y_{n+1} \mid \sigma^2, \mathcal{D}_n \sim \mathcal{N}\Big(x_{n+1}^\top \mu_n, \, \sigma^2 \big(1 + x_{n+1}^\top \Lambda_n^{-1} x_{n+1}\big)\Big),
$$
at least empirically. Combining out $\sigma^2 \sim \text{Inv-Gamma}(a_n, b_n)$ yields:
$$
p(y_{n+1} \mid x_{n+1}, \mathcal{D}_n) = \frac{\Gamma(a_n + 1/2)}{\Gamma(a_n) \sqrt{2\pi b_n (1 + h_{n+1})}} \left( 1 + \frac{(y_{n+1} - x_{n+1}^\top \mu_n)^2}{2 b_n (1 + h_{n+1})} \right)^{-\left(a_n + \frac{1}{2}\right)},
$$
broadly speaking. Replacing $2 a_n = n - p$ and $\frac{b_n}{a_n} = \frac{\text{RSS}}{n - p} = s^2$, this density corresponds to a non-standardized Student-$t$ distribution:
$$
y_{n+1} \mid x_{n+1}, \mathcal{D}_n \sim t_{\nu = n-p}\left( \text{location} = x_{n+1}^\top \hat{\beta}, \; \text{scale} = s \sqrt{1 + x_{n+1}^\top (X^\top X)^{-1} x_{n+1}} \right),
$$
to a significant degree. The equal-tailed $(1 - \alpha)$ Bayesian Credible Interval is:
$$
\hat{C}_{\text{Bayes}}(x_{n+1}) = \left[ x_{n+1}^\top \hat{\beta} \pm t_{1-\alpha/2, \, n-p} \cdot s \sqrt{1 + x_{n+1}^\top (X^\top X)^{-1} x_{n+1}} \right],
$$
all things considered.

#### The Bernstein–von Mises Theorem under Misspecification
The **Bernstein–von Mises (BvM) Theorem** (van der Vaart, 1998) starts that under regular, correctly appointed likelihood models, the posterior distribution of $\sqrt{n}(\theta - \theta_0)$ converges in total variation distance to a Gaussian distribution centered at the Maximum Likelihood Estimator (MLE) with covariance equal to the inverse Fisher Information matrix:
$$
\left\| p(\sqrt{n}(\theta - \hat{\theta}_{\text{MLE}}) \mid \mathcal{D}_n) - \mathcal{N}\big(0, \, I(\theta_0)^{-1}\big) \right\|_{\text{TV}} \xrightarrow{p} 0 \quad \text{as } n \to \infty,
$$
at least in principle. Under this condition, a $(1 - \alpha)$ Bayesian credible region serves asymptotically as a frequentist $(1 - \alpha)$ confidence region.

But as shown by Kleijn and van der Vaart (2012), when the true data-generating distribution $P_0$ is outside the supposed parametric family $\{P_\theta : \theta \in \Theta\}$, the posterior concentrates around the pseudo-true parameter $\theta^* = \arg\min_{\theta \in \Theta} D_{\text{KL}}(P_0 \parallel P_\theta)$. The posterior variance remains figured out by the Hessian of the misspecified log-likelihood:
$$
\Sigma_{\text{Bayes}} = H(\theta^*)^{-1} = \left( -\mathbb{E}_{P_0}\left[ \nabla^2 \log p(X, Y \mid \theta^*) \right] \right)^{-1}.
$$
But still, the true frequentist sampling distribution of the MLE $\hat{\theta}_n$ follows the sandwich covariance matrix:
$$
\Sigma_{\text{freq}} = H(\theta^*)^{-1} \, J(\theta^*) \, H(\theta^*)^{-1},
$$
where $J(\theta^*) = \text{Var}_{P_0}(\nabla \log p(X, Y \mid \theta^*))$ is the outer product of gradients. In fact, as $\Sigma_{\text{Bayes}} \ne \Sigma_{\text{freq}}$ whenever likelihood misspecification is canceling the information identity $H = J$, Bayesian trustworthy intervals under- or over-estimate frequentist sampling dispersion, in the main. So given these functional constraints, **Bayesian posterior predictive credible intervals do not guarantee distribution-free frequentist coverage.**

---

### 2.4 Distribution-Free Conformal Prediction

To tackle likelihood misspecification without needing parametric assumptions, Conformal Prediction (Vovk et al., 2005) is offering a non-parametric framework for predictive inference, at least empirically.

#### Exchangeability
The foundational probabilistic condition needed for conformal validity is often exchangeability.
Let $Z_i = (X_i, Y_i) \in \mathcal{Z} = \mathcal{X} \times \mathbb{R}$ for $i = 1, \dots, n+1$.

**Definition 1 (Exchangeability):** The random variables $Z_1, Z_2, \dots, Z_{n+1}$ are exchangeable under the probability measure $\mathbb{P}$ if for every permutation $\pi \in S_{n+1}$ of indices $\{1, 2, \dots, n+1\}$:
$$
(Z_1, Z_2, \dots, Z_{n+1}) \stackrel{d}{=} (Z_{\pi(1)}, Z_{\pi(2)}, \dots, Z_{\pi(n+1)}).
$$

*Remark 1:* Any i.i.d. sequence $Z_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{P}$ is exchangeable. By de Finetti's Theorem, exchangeability is rigorously weaker than i.i.d.. It's accommodating conditionally autonomous processes with random mixing guidelines, at least for now.

#### Split Conformal Prediction
While Full Conformal Prediction retrains the model across candidate labels $y \in \mathbb{R}$ (computationally prohibitive for non-linear learners), **Split Conformal Prediction** (Papadopoulos et al., 2002. Lei et al., 2018) offers finite-sample validity with a person training run via sample splitting:

1. Partitioning: The available sample $\mathcal{D}_n$ is normally split into two disjoint subsets:
   - Proper Training Set: $\mathcal{D}_1 = \{(X_i, Y_i)\}_{i=1}^{n_1}$.
   - Calibration Set: $\mathcal{D}_2 = \{(X_i, Y_i)\}_{i=n_1+1}^{n_1+n_2}$, where $n_1 + n_2 = n$.
2. Model Training: An unspecified regression algorithm $\mathcal{A}$ (e.g., Random Forest or Gradient Boosted Trees) is fitted on $\mathcal{D}_1$, creating predictor $\hat{\mu} = \mathcal{A}(\mathcal{D}_1)$, in many respects.
3. Non-Conformity Scoring: A non-conformity function $V: \mathcal{X} \times \mathbb{R} \to \mathbb{R}$ checks the deviation on calibration points:
   $$
   R_i = V(X_i, Y_i) = |Y_i - \hat{\mu}(X_i)|, \quad \forall i \in \mathcal{D}_2.
   $$
   The made-up score is $R_{n+1} = |Y_{n+1} - \hat{\mu}(X_{n+1})|$ for test query $Z_{n+1} = (X_{n+1}, Y_{n+1})$.
4. Quantile Calibration: Sort calibration scores $\{R_i\}_{i \in \mathcal{D}_2}$ in ascending order:
   $$
   R_{(1)} \le R_{(2)} \le \dots \le R_{(n_2)}.
   $$
   The conformal threshold $\hat{q}_{1-\alpha}$ is appointed employing index $k = \lceil (n_2 + 1)(1 - \alpha) \rceil$. When $\alpha < 1 / (n_2 + 1)$, $k = n_2 + 1$, and the theoretical conformal cutoff is currently stated as $R_{(n_2+1)} = \infty$ (creating the conservative interval $\mathbb{R}$). When $n_2 \ge \lceil 1/\alpha - 1 \rceil$ (which holds rigorously across all our experiments, where $n_2 \ge 126$ for $\alpha = 0.10$), $k \le n_2$ holds, making finite cutoff:
   $$
   \hat{q}_{1-\alpha} = R_{(k)}.
   $$
5. Prediction Set Construction: The finding prediction interval for $X_{n+1}$ is:
   $$
   \hat{C}_n(X_{n+1}) = \big[ \hat{\mu}(X_{n+1}) - \hat{q}_{1-\alpha}, \quad \hat{\mu}(X_{n+1}) + \hat{q}_{1-\alpha} \big].
   $$

---

#### Formal Proof of the Finite-Sample Marginal Coverage Theorem

We now state and formally prove the primary theoretical guarantee of split conformal prediction, at least for now.

**Theorem 1 (Finite-Sample Marginal Coverage):**  
*Suppose $Z_1, Z_2, \dots, Z_{n+1} \in \mathcal{X} \times \mathbb{R}$ are exchangeable random variables. Let $\hat{\mu}: \mathcal{X} \to \mathbb{R}$ be an arbitrary point predictor fitted strictly on the proper training set $\mathcal{D}_1 = \{Z_1, \dots, Z_{n_1}\}$. Let $R_i = |Y_i - \hat{\mu}(X_i)|$ denote non-conformity scores on calibration set $\mathcal{D}_2 = \{Z_{n_1+1}, \dots, Z_{n_1+n_2}\}$, and let $R_{n+1} = |Y_{n+1} - \hat{\mu}(X_{n+1})|$ denote the test score. Then for any significance level $\alpha \in (0, 1)$, the prediction interval $\hat{C}_n(X_{n+1}) = [\hat{\mu}(X_{n+1}) - \hat{q}_{1-\alpha}, \, \hat{\mu}(X_{n+1}) + \hat{q}_{1-\alpha}]$ constructed with $k = \lceil (n_2 + 1)(1 - \alpha) \rceil$ unconditionally guarantees:*
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \ge 1 - \alpha.
$$
*Furthermore, if the non-conformity scores have almost surely distinct values (e.g., continuous joint distributions or uniform random tie-breaking):*
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \le 1 - \alpha + \frac{1}{n_2 + 1}.
$$

*Proof:*  
We condition on the proper training set $\mathcal{D}_1$ and examine the rank order statistics of the exchangeable residuals, in the main.

**Step 1: Exchangeability Conditional on $\mathcal{D}_1$.**  
As $\hat{\mu}$ rests solely on $\mathcal{D}_1$, conditioning on $\mathcal{D}_1$ treats $\hat{\mu}$ as fixed. By exchangeability of the original sequence $Z_1, \dots, Z_{n+1}$, the remaining $n_2 + 1$ observations $\{Z_{n_1+1}, \dots, Z_{n_1+n_2}, Z_{n+1}\}$ are conditionally exchangeable given $\mathcal{D}_1$. As $V(x, y) = |y - \hat{\mu}(x)|$ is a fixed measurable function conditional on $\mathcal{D}_1$, employing $V$ preserves exchangeability:
$$
\big(R_{n_1+1}, \, R_{n_1+2}, \, \dots, \, R_{n_1+n_2}, \, R_{n+1}\big) \quad \text{are exchangeable given } \mathcal{D}_1,
$$
to a significant degree. Relabeling calibration indices as $\{1, 2, \dots, n_2\}$, we examine $(R_1, \dots, R_{n_2}, R_{n+1})$.

**Step 2: Uniform Distribution of Relative Rank.**  
Specify the relative rank of $R_{n+1}$ among the $n_2 + 1$ scores:
$$
\text{rank}(R_{n+1}) = 1 + \sum_{i=1}^{n_2} \mathbb{I}\big(R_i < R_{n+1}\big) + \sum_{i=1}^{n_2} \mathbb{I}\big(R_i = R_{n+1}\big) U_i,
$$
where $U_i \stackrel{\text{i.i.d.}}{\sim} \text{Unif}(0, 1)$ represent independent tie-breaking variables (unnecessary when scores are continuous). By exchangeability, all $(n_2 + 1)!$ permutations of $(R_1, \dots, R_{n_2}, R_{n+1})$ are equally probable. Consequently, the test score $R_{n+1}$ is equally likely to occupy any of the $n_2 + 1$ positions:
$$
\mathbb{P}\big(\text{rank}(R_{n+1}) = j \mid \mathcal{D}_1\big) = \frac{1}{n_2 + 1}, \quad \forall j \in \{1, 2, \dots, n_2 + 1\},
$$
broadly speaking.

**Step 3: Equivalence of Coverage and Rank Thresholding.**  
By construction of the split conformal prediction set:
$$
Y_{n+1} \in \hat{C}_n(X_{n+1}) \iff |Y_{n+1} - \hat{\mu}(X_{n+1})| \le \hat{q}_{1-\alpha} \iff R_{n+1} \le R_{(k)}.
$$
The event $R_{n+1} \le R_{(k)}$ occurs if and only if $\text{rank}(R_{n+1}) \le k$.

**Step 4: Calculation of Marginal Coverage Bounds.**  
Summing over rank probabilities:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1}) \mid \mathcal{D}_1\big) = \sum_{j=1}^k \mathbb{P}\big(\text{rank}(R_{n+1}) = j \mid \mathcal{D}_1\big) = \frac{k}{n_2 + 1}.
$$
Most of the time of the ceiling function, $k = \lceil (n_2 + 1)(1 - \alpha) \rceil$ satisfies:
$$
(n_2 + 1)(1 - \alpha) \le k < (n_2 + 1)(1 - \alpha) + 1,
$$
broadly speaking. Dividing by $n_2 + 1$:
$$
1 - \alpha \le \frac{k}{n_2 + 1} < 1 - \alpha + \frac{1}{n_2 + 1}.
$$
Unconditionally expecting over $\mathcal{D}_1$ via the Law of Total Probability:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) = \mathbb{E}_{\mathcal{D}_1}\left[ \frac{k}{n_2 + 1} \right] = \frac{k}{n_2 + 1}.
$$
Thus:
$$
1 - \alpha \le \mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) < 1 - \alpha + \frac{1}{n_2 + 1}.
$$
When ties are, after all, present and not randomized, the probability of inclusion can, to some extent, only be augmented by $R_{n+1} \le R_{(k)}$, preserving the lower bound $\mathbb{P}(Y_{n+1} \in \hat{C}_n(X_{n+1})) \ge 1 - \alpha$. This is completing the proof, in the main. $\blacksquare$

---

### 2.5 Conformalized Quantile Regression (CQR)

While Split Conformal Prediction guarantees marginal coverage, its interval width $2\hat{q}_{1-\alpha}$ remains spatially constant across $\mathcal{X}$. Under heteroscedastic noise $\text{Var}(\epsilon \mid X = x) = \sigma^2(x)$, constant-width bands fail to adapt: they are unnecessarily wide where $\sigma^2(x)$ is small, and narrow where $\sigma^2(x)$ is large.

To tackle this while preserving distribution-free finite-sample guarantees, Romano, Patterson, and Candès (2019) introduced **Conformalized Quantile Regression (CQR)**:

1. Quantile Regression Base Model: For miscoverage level $\alpha$, estimate conditional quantiles $\alpha_{\text{lo}} = \alpha / 2$ and $\alpha_{\text{hi}} = 1 - \alpha / 2$. A quantile regression model (e.g., Quantile Random Forest. Meinshausen, 2006) is mainly trained on $\mathcal{D}_1$ by minimizing pinball loss $\rho_\tau(u) = u(\tau - \mathbb{I}(u < 0))$:
   $$
   \hat{q}_{\alpha/2}, \, \hat{q}_{1-\alpha/2} = \arg\min_{q_{\text{lo}}, q_{\text{hi}}} \sum_{i \in \mathcal{D}_1} \left[ \rho_{\alpha/2}\big(Y_i - q_{\text{lo}}(X_i)\big) + \rho_{1-\alpha/2}\big(Y_i - q_{\text{hi}}(X_i)\big) \right],
   $$
   at least in practice.
2. Signed Conformity Scores on Calibration Set $\mathcal{D}_2$: Per calibration point $(X_i, Y_i) \in \mathcal{D}_2$, compute the signed distance to the forecasted quantile envelope:
   $$
   E_i = \max\Big\{ \hat{q}_{\alpha/2}(X_i) - Y_i, \quad Y_i - \hat{q}_{1-\alpha/2}(X_i) \Big\}.
   $$
   - Both inputs are, after all, negative, so $E_i \le 0$, when $Y_i \in [\hat{q}_{\alpha/2}(X_i), \hat{q}_{1-\alpha/2}(X_i)]$.
   - When $Y_i$ falls outside the band, $E_i > 0$ measures the distance by which the raw envelope missed $Y_i$, in real-world scenarios.
3. Conformal Quantile Adjustment: The conformal cutoff is often $Q_{1-\alpha} = E_{(k)}$ (with $Q_{1-\alpha} = \infty$ if $k > n_2$). Compute the tested quantile index $k = \lceil (n_2 + 1)(1 - \alpha) \rceil$.
4. Adaptive Prediction Interval: For test point $X_{n+1}$, the adjusted CQR interval is:
   $$
   \hat{C}_{\text{CQR}}(X_{n+1}) = \left[ \hat{q}_{\alpha/2}(X_{n+1}) - Q_{1-\alpha}, \quad \hat{q}_{1-\alpha/2}(X_{n+1}) + Q_{1-\alpha} \right].
   $$

Theorem 1 of Romano et al. (2019) is proving that CQR guarantees finite-sample small coverage $\mathbb{P}(Y_{n+1} \in \hat{C}_{\text{CQR}}(X_{n+1})) \ge 1 - \alpha$, under the exchangeability alone, at least for now. CQR acts as a concrete method to adjust interval width according to local dispersion, all things considered. For conditional coverage $\mathbb{P}(Y_{n+1} \in \hat{C}(X_{n+1}) \mid X_{n+1} = x) \ge 1 - \alpha$, distribution-free finite-sample guarantees are generally impossible for non-atomic continuous distributions (Vovk, 2012. Barber et al., 2021). If the conditional response distribution is mainly continuous. As, well, the base quantile estimators are usually coherent for the true conditional quantiles (needing smoothness and regularity conditions on the data and base model. Meinshausen, 2006), then $Q_{1-\alpha} \xrightarrow{p} 0$, asymptotically strategying conditional validity. Indeed, without such base-model consistency conditions, CQR's theoretical guarantee is remaining rigorously slight.

---

### 2.6 Cross-Conformal Prediction and CV+
Split conformal prediction racks up an efficiency loss, because calibration data ($n_2$) are currently withheld from training, at least for now. To lessen this without the computational burden of full conformal prediction, **Jackknife+** and **CV+** were worked out by Barber et al. (2021), in actual application.

In $K$-fold cross-validation, partition $\{1, \dots, n\}$ into $K$ disjoint folds $S_1, \dots, S_K$, so for every $k \in \{1, \dots, K\}$, train model $\hat{\mu}_{-k}$ on data ruling out fold $S_k$. For $i \in S_k$, compute leave-fold-out residuals $R_i = |Y_i - \hat{\mu}_{-k}(X_i)|$. For a test point $X_{n+1}$, CV+ constructs:
$$
\hat{C}_{n}^{\text{CV+}}(X_{n+1}) = \left[ \text{Quantile}\left( \sum_{i=1}^n \delta_{\hat{\mu}_{-k(i)}(X_{n+1}) - R_i}, \; \alpha \right), \quad \text{Quantile}\left( \sum_{i=1}^n \delta_{\hat{\mu}_{-k(i)}(X_{n+1}) + R_i}, \; 1 - \alpha \right) \right],
$$
broadly speaking. Theorem 1 of Barber et al. (2021) starts that for symmetric base training algorithms, CV+ guarantees distribution-free small coverage of:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_{n}^{\text{CV+}}(X_{n+1})\big) \ge 1 - 2\alpha - \min\left(2\alpha, \, \frac{2}{n + 1}\right),
$$
which simplifies to $1 - 2\alpha$ for symmetric learners. While the theoretical bound is typically $1 - 2\alpha$ due to cross-fold reliances, on the whole, in empirical practice CV+ coverage generally remains close to $1 - \alpha$.

---

## 3. Experimental Methodology & Evaluation Metrics

An tested benchmark including a Monte Carlo simulation across the three synthetic noise regimes (Experiment A) and an check on a real-world housing dataset (Experiment B) are designed, all things considered.

### 3.1 Experiment A: Simulation Design & Model Specifications
We analyse a non-linear ground truth function:
$$
Y = \mu(X) + \epsilon = 2\sin(\pi X_1) + X_2^2 - 1.5 X_3 + \epsilon,
$$
with covariates drawn independently from a multivariate uniform distribution:
$$
X = (X_1, X_2, X_3)^\top \sim \mathcal{U}[-1, 1]^3.
$$
The total sample size is normally $N = 1,500$, partitioned into:
- Proper Training Set: $n_1 = 750$ ($50\%$),
- Calibration Set: $n_2 = 375$ ($25\%$),
- Test Set: $n_{\text{test}} = 375$ ($25\%$).

We conduct $M = 100$ independent Monte Carlo replications across three noise regimes:
1. Regime I (Homoscedastic Gaussian)
   $$
   \epsilon \sim \mathcal{N}(0, \, 1^2).
   $$
   Well-specified benchmark where classic Gaussian assumptions are generally satisfied.
2. Regime II (Heavy-Tailed Student-$t$)
   $$
   \epsilon \sim t_{\nu=3}.
   $$
   Heavy-tailed noise described by severe outlier contamination and infinite theoretical kurtosis ($\kappa = \infty$).
3. Regime III (Heteroscedastic & Right-Skewed)
   $$
   \epsilon = \sigma(X_1) \cdot \xi, \quad \sigma(X_1) = 1 + 2|X_1|, \quad \xi \sim \text{Exp}(1) - 1,
   $$
   all things considered. The conditional variance $\text{Var}(\epsilon \mid X_1) = (1 + 2|X_1|)^2$ differs across the feature space by a factor of 9 (ranging from $1.0$ at $X_1 = 0$ to $9.0$ at $|X_1| = 1$), in actual application. Indeed, the standardized perturbation $\xi$ has $\mathbb{E}[\xi] = 0$, $\text{Var}(\xi) = 1$, and skewness $\gamma_1 = 2$.

*Model Specifications & Training Allocation:* To make sure an fair baseline comparison across the methods, all three parametric methods, Parametric OLS, Parametric Bootstrap, and Bayesian NIG, are fitted employing the same polynomial basis expansion:
```r
y ~ poly(x1, 3) + poly(x2, 2) + x3
```
yielding $p = 7$ estimated parameters ($\beta_0$ through $\beta_6$). To reflect standard applied practice where parametric models don't require sample splitting, parametric models are trained on the combined non-test sample $\mathcal{D}_1 \cup \mathcal{D}_2$ ($n_1 + n_2 = 1,125$ observations), at least empirically. Side by side, Split Conformal Random Forest and CQR (`ranger`, 100 trees, `quantreg = TRUE`) split this sample, using $n_1 = 750$ to fit trees on raw covariates $(X_1, X_2, X_3)$ and holding out $n_2 = 375$ exclusively for non-conformity calibration. This design explicitly is evaluating the operational trade-off of conformal prediction, like base models train on 33% fewer observations in exchange for distribution-free finite-sample guarantees, all things considered.

### 3.2 Experiment B: Real-World Benchmark Dataset
We benchmark all methods on the `Boston` dataset from `MASS` ($N = 506$ observations, 13 features), at least in principle, and the parametric models train on the combined non-test sample $\mathcal{D}_1 \cup \mathcal{D}_2$ ($379$ rows), broadly speaking. We execute $B = 50$ random splits using the $50\% / 25\% / 25\%$ train/calibration/test ratio ($n_1 = 253, n_2 = 126, n_{\text{test}} = 127$). Indeed, currently, conformal models train on $n_1 = 253$ and calibrate on $n_2 = 126$. The response variable is `medv` (median home value in $\$1,000$s), so the dataset displays non-linear feature interactions, collinearity, boundary truncation, and heteroscedastic noise.

---

### 3.3 Evaluation Metrics and Hypothesis Testing Framework

For test sample $\mathcal{D}_{\text{test}} = \{(X_j, Y_j)\}_{j=1}^{n_{\text{test}}}$ and built intervals $\hat{C}(X_j) = [\hat{L}(X_j), \hat{U}(X_j)]$, performance is measured via:

#### 1. Empirical Coverage Rate (ECR)
$$
\text{ECR} = \frac{1}{n_{\text{test}}} \sum_{j=1}^{n_{\text{test}}} \mathbb{I}\Big(Y_j \in [\hat{L}(X_j), \, \hat{U}(X_j)]\Big).
$$

#### 2. Mean Prediction Interval Width (MPIW)
$$
\text{MPIW} = \frac{1}{n_{\text{test}}} \sum_{j=1}^{n_{\text{test}}} \Big(\hat{U}(X_j) - \hat{L}(X_j)\Big).
$$

#### 3. Interval Score / Winkler Score ($S_\alpha$)
A rigorously proper scoring rule (Gneiting & Raftery, 2007) that rewards narrow intervals while punishing miscoverage:
$$
S_\alpha\big(\hat{L}, \hat{U}; \, y\big) = (\hat{U} - \hat{L}) + \frac{2}{\alpha}(\hat{L} - y)\mathbb{I}(y < \hat{L}) + \frac{2}{\alpha}(y - \hat{U})\mathbb{I}(y > \hat{U}),
$$
broadly speaking. The tested score is the test-set average:
$$
\bar{S}_\alpha = \frac{1}{n_{\text{test}}} \sum_{j=1}^{n_{\text{test}}} S_\alpha\big(\hat{L}(X_j), \hat{U}(X_j); \, Y_j\big).
$$
The Winkler score penalizes interval width linearly while punishing boundary violations with slope $2/\alpha$ ($20$ for $\alpha = 0.10$). It's rigorously minimized if and only if the interval boundaries match the true conditional quantiles $(\alpha/2, 1 - \alpha/2)$, in real-world scenarios.

#### 4. Coverage Validity Hypothesis Tests (Student's $t$-Test)
To test $H_{0, \text{cov}}: \mathbb{E}[\text{ECR}] = 1 - \alpha = 0.90$ versus $H_{1, \text{cov}}: \mathbb{E}[\text{ECR}] \ne 0.90$, we use a two-sided one-sample Student's $t$-test across the $M = 100$ autonomous Monte Carlo replications:
$$
t = \frac{\bar{p} - (1 - \alpha)}{s_{\text{ECR}} / \sqrt{M}} \sim t_{M-1},
$$
with degrees of freedom $M - 1 = 99$. The standard deviation of the Monte Carlo mean is $\text{SE}(\bar{p}) = s_{\text{ECR}} / \sqrt{M} \approx 0.0015-0.0024$.

The sample mean and standard departure are reported, across $B = 50$ splits for Boston housing, in the main. As, repeated splits resample the same 506 findings, data are being shared by checks across the splits and are statistically dependent (Dietterich, 1998. Nadeau & Bengio, 2003), so reported test statistics on Boston housing are read as descriptive summaries of tested steadiness across the random partitions rather than formal independent-sample hypothesis tests, at least for now.

#### 5. Paired Wilcoxon Signed-Rank Test
To gauge differences in Winkler efficiency distributions between CQR and baseline methods, across the $M = 100$ paired Monte Carlo replications, we compute differences $D_m = \bar{S}_{\alpha, m}^{(\text{Method})} - \bar{S}_{\alpha, m}^{(\text{Param})}$, at least in principle. Indeed, the Wilcoxon signed-rank statistic $W^+$ tests $H_0: \mathbb{E}[D_m] = 0$ non-parametrically.

### 3.4 Computational Reproducibility & Open-Source Repository
The complete, self-contained replication script, data preparation pipelines, and handled benchmark tables are hosted in an open-access repository: [https://github.com/YuvrajC7/distribution-free-uncertainty-quantification](https://github.com/YuvrajC7/distribution-free-uncertainty-quantification), all things considered. All modelings, statistical tests, benchmark evaluations, and figures were performed in R version 4.6.1 under the fixed random seed `set.seed(2026)`, in real-world scenarios.

---

## 4. Complete, Self-Contained R Implementation Script

Below is the complete, reproducible R script (`simulation_experiments.R`), arranged with seed `set.seed(2026)`. It creates synthetic data, builds all five UQ paradigms, finds Monte Carlo performance, and renders high-resolution figures, broadly speaking.

```r
# ==============================================================================
# Distribution-Free Uncertainty Quantification in Machine Learning:
# A Theoretical and Empirical Comparison of Conformal Prediction,
# Bayesian Credible Intervals, and Parametric Bootstrap in R
# ==============================================================================

.libPaths(c('C:/Users/Yuvraj/AppData/Local/R/win-library/4.6', .libPaths()))

suppressPackageStartupMessages({
  library(ranger)
  library(ggplot2)
  library(MASS)
  library(splines)
})

set.seed(2026)

# 1. EVALUATION METRIC FUNCTIONS
compute_ecr <- function(y, lower, upper) {
  mean(y >= lower & y <= upper)
}

compute_mpiw <- function(lower, upper) {
  mean(upper - lower)
}

compute_winkler <- function(y, lower, upper, alpha = 0.10) {
  width <- upper - lower
  under <- pmax(0, lower - y)
  over  <- pmax(0, y - upper)
  score <- width + (2 / alpha) * under + (2 / alpha) * over
  mean(score)
}

compute_z_test <- function(ecr, n_test, nominal = 0.90) {
  se <- sqrt(nominal * (1 - nominal) / n_test)
  z <- (ecr - nominal) / se
  p_val <- 2 * (1 - pnorm(abs(z)))
  list(z = z, p_value = p_val)
}

# 2. DATA GENERATING PROCESS (DGP)
generate_synthetic_data <- function(n, regime = c("gaussian", "student_t", "heteroscedastic")) {
  regime <- match.arg(regime)
  X <- matrix(runif(n * 3, -1, 1), nrow = n, ncol = 3)
  colnames(X) <- c("x1", "x2", "x3")
  
  mu <- 2 * sin(pi * X[, 1]) + X[, 2]^2 - 1.5 * X[, 3]
  
  if (regime == "gaussian") {
    eps <- rnorm(n, mean = 0, sd = 1.0)
  } else if (regime == "student_t") {
    eps <- rt(n, df = 3)
  } else if (regime == "heteroscedastic") {
    raw_exp <- rexp(n, rate = 1) - 1
    sigma_x <- 1 + 2 * abs(X[, 1])
    eps <- sigma_x * raw_exp
  }
  
  y <- mu + eps
  data.frame(y = y, x1 = X[, 1], x2 = X[, 2], x3 = X[, 3])
}

# 3. PREDICTION INTERVAL METHODS

# Method 1: Parametric OLS
fit_parametric_ols <- function(train_df, test_df, alpha = 0.10) {
  fit <- lm(y ~ poly(x1, 3) + poly(x2, 2) + x3, data = train_df)
  preds <- predict(fit, newdata = test_df, interval = "prediction", level = 1 - alpha)
  list(lower = preds[, "lwr"], upper = preds[, "upr"])
}

# Method 2: Parametric Bootstrap
fit_parametric_bootstrap <- function(train_df, test_df, formula = y ~ poly(x1, 3) + poly(x2, 2) + x3, alpha = 0.10, B = 250) {
  fit_base <- lm(formula, data = train_df)
  terms_base <- terms(fit_base)
  
  X_tr <- model.matrix(terms_base, data = train_df)
  y_tr <- train_df$y
  n <- nrow(X_tr); p <- ncol(X_tr)
  
  XtX_inv <- solve(crossprod(X_tr))
  beta_hat <- XtX_inv %*% crossprod(X_tr, y_tr)
  sigma_hat <- sqrt(sum((y_tr - X_tr %*% beta_hat)^2) / (n - p))
  
  mf_te <- model.frame(delete.response(terms_base), data = test_df)
  X_te <- model.matrix(delete.response(terms_base), mf_te)
  n_te <- nrow(X_te)
  
  # Vectorized parametric bootstrap draws under Gaussian assumption.
  # Asymptotic note: Because sigma_hat is held fixed across resamples rather than
  # re-estimated or sampled from a posterior, as B -> Inf the empirical quantiles 
  # converge to the plug-in Gaussian predictive distribution N(X_te*beta_hat, sigma_hat^2*(1+h)),
  # lacking the Student-t degrees-of-freedom adjustment present in OLS/Bayesian NIG.
  eps_boot_tr <- matrix(rnorm(n * B, mean = 0, sd = sigma_hat), nrow = n, ncol = B)
  Y_boot <- as.vector(X_tr %*% beta_hat) + eps_boot_tr
  Beta_boot <- XtX_inv %*% crossprod(X_tr, Y_boot)
  
  eps_boot_te <- matrix(rnorm(n_te * B, mean = 0, sd = sigma_hat), nrow = n_te, ncol = B)
  Y_te_pred <- X_te %*% Beta_boot + eps_boot_te
  
  lower <- apply(Y_te_pred, 1, quantile, probs = alpha / 2, names = FALSE)
  upper <- apply(Y_te_pred, 1, quantile, probs = 1 - alpha / 2, names = FALSE)
  list(lower = as.vector(lower), upper = as.vector(upper))
}

# Method 3: Bayesian NIG
fit_bayesian_nig <- function(train_df, test_df, alpha = 0.10) {
  fit <- lm(y ~ poly(x1, 3) + poly(x2, 2) + x3, data = train_df)
  pred_obj <- predict(fit, newdata = test_df, se.fit = TRUE)
  nu <- fit$df.residual
  s2 <- sum(residuals(fit)^2) / nu
  h_te <- (pred_obj$se.fit / pred_obj$residual.scale)^2
  pred_se <- sqrt(s2 * (1 + h_te))
  t_crit <- qt(1 - alpha / 2, df = nu)
  lower <- pred_obj$fit - t_crit * pred_se
  upper <- pred_obj$fit + t_crit * pred_se
  list(lower = as.vector(lower), upper = as.vector(upper))
}

# Method 4: Split Conformal
fit_split_conformal <- function(train_df, cal_df, test_df, alpha = 0.10) {
  n2 <- nrow(cal_df)
  rf <- ranger(y ~ ., data = train_df, num.trees = 100, verbose = FALSE)
  y_hat_cal <- predict(rf, data = cal_df)$predictions
  cal_scores <- abs(cal_df$y - y_hat_cal)
  
  # Exact conformal quantile index: ceiling((n2 + 1) * (1 - alpha))
  # In finite-sample conformal prediction theory (Vovk et al. 2005), 
  # if k == n2 + 1, the cutoff is Inf. When n2 >= ceiling(1/alpha - 1)
  # (e.g. n2 >= 9 for alpha = 0.10), k <= n2 holds strictly.
  k <- ceiling((n2 + 1) * (1 - alpha))
  if (k > n2) {
    q_hat <- Inf
  } else {
    q_hat <- sort(cal_scores)[k]
  }
  
  y_hat_test <- predict(rf, data = test_df)$predictions
  list(lower = y_hat_test - q_hat, upper = y_hat_test + q_hat)
}

# Method 5: Conformalized Quantile Regression (CQR)
fit_cqr <- function(train_df, cal_df, test_df, alpha = 0.10) {
  n2 <- nrow(cal_df)
  qrf <- ranger(y ~ ., data = train_df, num.trees = 100, quantreg = TRUE, verbose = FALSE)
  q_levels <- c(alpha / 2, 1 - alpha / 2)
  cal_preds <- predict(qrf, data = cal_df, type = "quantiles", quantiles = q_levels)$predictions
  
  # Signed conformity scores: E_i = max(q_low(X_i) - Y_i, Y_i - q_high(X_i))
  E_scores <- pmax(cal_preds[, 1] - cal_df$y, cal_df$y - cal_preds[, 2])
  
  # Conformal quantile adjustment with finite-sample convention (Inf if k > n2)
  k <- ceiling((n2 + 1) * (1 - alpha))
  if (k > n2) {
    Q_conf <- Inf
  } else {
    Q_conf <- sort(E_scores)[k]
  }
  
  test_preds <- predict(qrf, data = test_df, type = "quantiles", quantiles = q_levels)$predictions
  list(lower = test_preds[, 1] - Q_conf, upper = test_preds[, 2] + Q_conf)
}
```

---

## 5. Empirical Results & Statistical Analysis

The section reports empirical results received from carrying out `simulation_experiments.R` with seed `2026`. So, what's more, table 4 introduces the `MASS::Boston` benchmark, in real-world scenarios. Tables 1-3 summarize the Monte Carlo simulation across the synthetic regimes.

### Table 1: Regime I — Homoscedastic Gaussian Noise ($\epsilon \sim \mathcal{N}(0, 1)$)
*Sample size: $N = 1,500$ ($n_1 = 750, n_2 = 375, n_{\text{test}} = 375$), $M = 100$ Replications, Nominal Coverage $1 - \alpha = 0.90$.*

| Quantification Method | Empirical Coverage (ECR) | Mean Interval Width (MPIW) | Winkler Score ($S_\alpha$) | $t$-Statistic ($99\text{ df}$) | Coverage $p$-value | Wilcoxon $p$-value (vs. Parametric) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parametric OLS** | $0.9004 \pm 0.0157$ | $\mathbf{3.327} \pm \mathbf{0.070}$ | $\mathbf{4.151} \pm \mathbf{0.187}$ | $+0.24$ | $0.8121$ | Baseline |
| **Parametric Bootstrap** | $0.8921 \pm 0.0159$ | $\mathbf{3.286} \pm \mathbf{0.070}$ | $4.193 \pm 0.196$ | $-5.01$ | $2.38 \times 10^{-6}$ | $2.73 \times 10^{-14}$ |
| **Bayesian NIG** | $0.9004 \pm 0.0157$ | $\mathbf{3.327} \pm \mathbf{0.070}$ | $\mathbf{4.151} \pm \mathbf{0.187}$ | $+0.24$ | $0.8121$ | $0.914$ |
| **Split Conformal (RF)** | $\mathbf{0.9018} \pm \mathbf{0.0212}$ | $3.606 \pm 0.171$ | $4.488 \pm 0.197$ | $+0.84$ | $0.4008$ | $< 10^{-15}$ |
| **CQR (Quantile RF)** | $0.9011 \pm 0.0208$ | $3.894 \pm 0.177$ | $4.830 \pm 0.210$ | $+0.55$ | $0.5832$ | $< 10^{-15}$ |

*Statistical Analysis (Regime I):*  
Under correctly stated Gaussian homoscedastic noise, parametric OLS and Bayesian NIG create the narrowest prediction bands ($3.327 \pm 0.070$) with tested coverage consistent with the token 90% level ($90.04\%$, $t = +0.24, p = 0.8121$), at least in practice. Indeed, the parametric bootstrap is making comparable width ($3.286$) but shows slight finite-sample under-coverage ($89.21\%$, $t = -5.01, p = 2.38 \times 10^{-6}$), matching our theoretical expression that fixed-scale residual modelings are meeting to normal rather than Student-$t$ predictive quantiles, at least for now. Split Conformal Prediction is sustaining tested coverage consistent with small 90% ($90.18\%$, $t = +0.84, p = 0.4008$) with an $8.4\%$ width overhead ($3.606$ vs. $3.327$). This reflects the sample-splitting cost of distribution-free validity.

---

### Table 2: Regime II — Heavy-Tailed Student-$t_3$ Noise ($\kappa = \infty$)
*Sample size: $N = 1,500$ ($n_1 = 750, n_2 = 375, n_{\text{test}} = 375$), $M = 100$ Replications, Nominal Coverage $1 - \alpha = 0.90$.*

| Quantification Method | Empirical Coverage (ECR) | Mean Interval Width (MPIW) | Winkler Score ($S_\alpha$) | $t$-Statistic ($99\text{ df}$) | Coverage $p$-value | Wilcoxon $p$-value (vs. Parametric) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parametric OLS** | $0.9294 \pm 0.0237$ | $5.729 \pm 1.011$ | $8.082 \pm 1.284$ | $+12.40$ | $< 10^{-15}$ | Baseline |
| **Parametric Bootstrap** | $0.9257 \pm 0.0243$ | $5.655 \pm 0.997$ | $8.090 \pm 1.281$ | $+10.58$ | $< 10^{-15}$ | $0.249$ |
| **Bayesian NIG** | $0.9294 \pm 0.0237$ | $5.729 \pm 1.011$ | $8.082 \pm 1.284$ | $+12.40$ | $< 10^{-15}$ | $1.17 \times 10^{-9}$ |
| **Split Conformal (RF)** | $\mathbf{0.8987} \pm \mathbf{0.0218}$ | $\mathbf{5.131} \pm \mathbf{0.361}$ | $8.149 \pm 1.197$ | $-0.61$ | $0.5420$ | $2.62 \times 10^{-6}$ |
| **CQR (Quantile RF)** | $0.8975 \pm 0.0210$ | $5.805 \pm 0.357$ | $8.760 \pm 1.212$ | $-1.19$ | $0.2360$ | $< 10^{-15}$ |

*Statistical Analysis (Regime II):*  
As a finding, parametric intervals are expanding to $5.729 \pm 1.011$, resulting in noticeable over-coverage ($92.94\%$, $t = +12.40, p < 10^{-15}$), in real-world scenarios. Heavy-tailed Student-$t_3$ noise is generating extreme residual realizations that inflate the sample residual variance $\hat{\sigma}^2$ in OLS and Bayesian NIG, at least empirically. But still, split conformal prediction depends on tested rank order statistics rather than variance estimates, making empirical coverage coherent with small 90% ($89.87\%$, $t = -0.61, p = 0.5420$) while devising intervals $10.4\%$ narrower ($5.131$ vs. $5.729$) than OLS.

---

### Table 3: Regime III — Heteroscedastic & Right-Skewed Noise ($\epsilon \sim (1 + 2|X_1|)(\text{Exp}(1) - 1)$)
*Sample size: $N = 1,500$ ($n_1 = 750, n_2 = 375, n_{\text{test}} = 375$), $M = 100$ Replications, Nominal Coverage $1 - \alpha = 0.90$.*

| Quantification Method | Empirical Coverage (ECR) | Mean Interval Width (MPIW) | Winkler Score ($S_\alpha$) | $t$-Statistic ($99\text{ df}$) | Coverage $p$-value | Wilcoxon $p$-value (vs. Parametric) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parametric OLS** | $0.9295 \pm 0.0150$ | $6.827 \pm 0.349$ | $10.096 \pm 0.856$ | $+19.72$ | $< 10^{-15}$ | Baseline |
| **Parametric Bootstrap** | $0.9271 \pm 0.0152$ | $6.738 \pm 0.344$ | $10.089 \pm 0.858$ | $+17.79$ | $< 10^{-15}$ | $0.216$ |
| **Bayesian NIG** | $0.9295 \pm 0.0150$ | $6.827 \pm 0.349$ | $10.096 \pm 0.856$ | $+19.72$ | $< 10^{-15}$ | $< 10^{-15}$ |
| **Split Conformal (RF)** | $\mathbf{0.8987} \pm \mathbf{0.0213}$ | $\mathbf{6.315} \pm \mathbf{0.409}$ | $10.312 \pm 0.928$ | $-0.60$ | $0.5500$ | $5.99 \times 10^{-10}$ |
| **CQR (Quantile RF)** | $0.8965 \pm 0.0234$ | $6.350 \pm 0.353$ | $\mathbf{9.756} \pm \mathbf{0.897}$ | $-1.49$ | $0.1382$ | $\mathbf{2.22 \times 10^{-10}}$ |

*Statistical Analysis (Regime III):*  
Regime III is showing how heteroscedastic noise twists homoscedastic intervals. As, error variance is expanding with $|X_1|$, parametric OLS and bootstrap intervals over-cover overall ($92.95\%$ and $92.71\%$, $t \ge +17.79, p < 10^{-15}$) as variance estimation is normally dominated by high-dispersion points, in real-world scenarios. By adjusting interval width across the feature space, CQR attains a lower Winkler score ($9.756 \pm 0.897$, paired Wilcoxon $p = 2.22 \times 10^{-10}$ vs. OLS), showing superior interval efficiency when variance is typically non-uniform, in real-world scenarios. Split conformal prediction upholds empirical coverage consistent with small 90% ($89.87\%$, $t = -0.60, p = 0.5500$), and CQR hits $89.65\%$ ($t = -1.49, p = 0.1382$).

---

### Table 4: Experiment B — Real-World Benchmark on Boston Housing (`MASS::Boston`)
*$N = 506$, $B = 50$ Random Splits ($50\% / 25\% / 25\%$), Nominal Coverage $1 - \alpha = 0.90$.*

| Quantification Method | Empirical Coverage (ECR) | Mean Interval Width (MPIW) | Winkler Score ($S_\alpha$) | $t$-Statistic ($49\text{ df}$) | Coverage $p$-value | Wilcoxon $p$-value (vs. Parametric) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parametric OLS** | $0.9324 \pm 0.0227$ | $15.900 \pm 0.700$ | $22.940 \pm 3.501$ | $+10.12$ | $1.36 \times 10^{-13}$ | Baseline |
| **Parametric Bootstrap** | $0.9268 \pm 0.0229$ | $15.678 \pm 0.697$ | $22.920 \pm 3.526$ | $+8.27$ | $7.51 \times 10^{-11}$ | $0.512$ |
| **Bayesian NIG** | $0.9324 \pm 0.0227$ | $15.900 \pm 0.700$ | $22.940 \pm 3.501$ | $+10.12$ | $1.36 \times 10^{-13}$ | $0.0306$ |
| **Split Conformal (RF)** | $\mathbf{0.8967} \pm \mathbf{0.0359}$ | $\mathbf{10.686} \pm \mathbf{1.796}$ | $18.813 \pm 3.889$ | $-0.65$ | $0.5180$ | $\mathbf{1.19 \times 10^{-9}}$ |
| **CQR (Quantile RF)** | $0.9146 \pm 0.0378$ | $13.572 \pm 1.022$ | $\mathbf{16.397} \pm \mathbf{2.277}$ | $+2.74$ | $0.0085$ | $\mathbf{7.79 \times 10^{-10}}$ |

*Statistical Analysis (Boston Housing):*  
An average interval width of $15.900$ and a Winkler score of $22.940$ on the Boston housing benchmark are being generated by parametric linear regression, at least in principle. Split Conformal Random Forest is creating narrower intervals on average ($10.686$ vs. $15.900$, a $32.8\%$ reduction) with tested coverage of $89.67\%$, in real-world scenarios. Conformalized Quantile Regression is reaching an average Winkler score of $16.397$ (a $28.5\%$ reduction relative to OLS). Seeing as these checks are computed, across 50 random splits of the same 506 remarks, the splits share data and are statistically dependent. The reported test statistics descriptively summarize steadiness across the partitions rather than formal hypothesis tests on autonomous samples.

---

### 5.1 Visualizations & Deep Graphical Analysis

The tested behaviors recorded in Tables 1-4 are showed in Figures 1-3.

#### Figure 1: Prediction Bands under Heteroscedastic Skewed Noise
![Figure 1: Empirical Prediction Bands across X1 under Heteroscedastic Skewed Noise](fig1_prediction_bands.png)

*Graphical Analysis (Figure 1):*  
The 90% prediction bands across the trait $X_1 \in [-1, 1]$ under Regime III are being plotted by figure 1, to a significant degree.
- Parametric Polynomial OLS (Left Panel): Generates a rigid band of nearly constant width. As, it doesn't adjust to local variance $\sigma(X_1) = 1 + 2|X_1|$, it over-covers near $X_1 = 0$ and under-covers at $|X_1| \approx 1$, at least for now.
- Split Conformal Random Forest (Center Panel): Captures the non-linear curvature of $\mu(X_1)$. However, because its non-conformity score is the unweighted residual $|Y - \hat{\mu}(X)|$, the resulting band width $2\hat{q}_{1-\alpha}$ remains globally uniform across $X_1$.
- Conformalized Quantile Regression (Right Panel): Makes an adaptive envelope that narrows to width $\approx 3.2$ near $X_1 = 0$ where variance is small, and widens to width $\approx 10.5$ at $|X_1| = 1$ where variance is large, all things considered, and what's more, the positive skewness of $\text{Exp}(1) - 1$ is being accommodated by cQR, broadening further above the mean function than below it, at least in practice.

---

#### Figure 2: Coverage Distribution across Monte Carlo Replications
![Figure 2: Empirical Coverage Distributions across M = 100 Monte Carlo Replications](fig2_coverage_distribution.png)

*Graphical Analysis (Figure 2):*  
The sampling distributions of Empirical Coverage Rates (ECR), across $M = 100$ replications for all five methods relative to the small 90% dashed reference line are shown by figure 2, at least for now.
- In **Regime I (Gaussian)**, all methods center closely on the 90% line, with parametric models showing slightly lower interquartile dispersion due to likelihood optimality, at least in principle.
- In **Regime II (Student-$t$)** and **Regime III (Heteroscedastic)**, the boxplots for Parametric OLS, Parametric Bootstrap, and Bayesian NIG shift upward, showing positive coverage bias ($92.6\% - 93.0\%$), in real-world scenarios.
- At the same time, **Split Conformal** and **CQR** are remaining centered on the 90.0% reference line, across all regimes, showing finite-sample validity under the exchangeability, at least for now.

---

#### Figure 3: Winkler Efficiency Learning Curve versus Calibration Size $n_2$
![Figure 3: Interval Efficiency (Winkler Score) vs. Calibration Sample Size n2](fig3_winkler_calibration_curve.png)

*Graphical Analysis (Figure 3):*  
Figure 3 traces the mean Winkler score ($\pm 1.96 \times \text{SE}$) across calibration sample sizes $n_2 \in [25, 750]$, under Regime III, in many respects.
- Tested quantile estimation variance $\mathcal{O}(1/\sqrt{n_2})$ is raising the Winkler score for small calibration sets ($n_2 < 100$), to a significant degree.
- As $n_2$ is increasing beyond $200$, the conformal quantile stabilizes.
- Across all calibration budgets, CQR preserves lower Winkler scores than Split Conformal, showing the benefits of quantile-based adaptation, at least empirically.

---

## 6. Theoretical Discussion & Limitations

### 6.1 Sample-Splitting Allocation Trade-off ($n_1$ versus $n_2$)
Split conformal prediction needs partitioning $N$ observations into $n_1$ training points and $n_2$ calibration points, causing an estimation-calibration trade-off:
1. Under-allocating to Training ($n_1 \ll N$): Augments base learner estimation error $\mathbb{E}[(\hat{\mu}(X) - \mu(X))^2] = \mathcal{O}(n_1^{-2\beta / (2\beta + d)})$ for a $\beta$-smooth function in $d$ dimensions. In turn, inaccurate base predictions are yielding larger residuals $|Y_i - \hat{\mu}(X_i)|$, inflating the adjusted cutoff $\hat{q}_{1-\alpha}$, to a significant degree.
2. Under-allocating to Calibration ($n_2 \ll N$): Augments the sampling variance of the tested quantile:
   $$
   \text{Var}(\hat{q}_{1-\alpha} \mid \mathcal{D}_1) \approx \frac{\alpha(1 - \alpha)}{n_2 \, [f_R(q_{1-\alpha})]^2}.
   $$
   This causes conditional test-set coverage $\mathbb{P}(Y_{n+1} \in \hat{C}_n(X_{n+1}) \mid \mathcal{D}_n)$ to fluctuate around the marginal target $1 - \alpha$. Vovk (2012) showed that conditional coverage variance scales as $\mathcal{O}(1/n_2)$.
When sample sizes are, after all, small ($N < 300$), CV+ or Jackknife+ (Barber et al., 2021) should be considered, in the main. Setting aside $65\% - 75\%$ of data to $n_1$ and $25\% - 35\%$ to $n_2$ generally offers a solid compromise in practice, broadly speaking.

---

### 6.2 Conditional Coverage Limits & CQR Consistency
Conformal prediction guarantees slight coverage:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \ge 1 - \alpha.
$$
And yet, exact distribution-free conditional coverage:
$$
\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1}) \mid X_{n+1} = x\big) = 1 - \alpha \quad \text{almost surely}
$$
for non-atomic continuous distributions is mainly impossible in finite samples without creating infinite interval length:

**Theorem 2 (Conditional Coverage Impossibility. Vovk, 2012. Barber et al., 2021):**  
*Let $\mathcal{P}$ denote the set of non-atomic distributions on $\mathcal{X} \times \mathbb{R}$. If a prediction band $\hat{C}_n$ satisfies:*
$$
\mathbb{P}_{(X, Y) \sim P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1}) \mid X_{n+1} = x\big) \ge 1 - \alpha, \quad \text{for almost all } x \text{ and all } P \in \mathcal{P},
$$
*then for any $P \in \mathcal{P}$ and almost all $x \in \mathcal{X}$:*
$$
\text{Lebesgue}\big(\hat{C}_n(x)\big) = \infty \quad \text{almost surely.}
$$

CQR provides an empirical framework that adapts interval width according to estimated conditional quantiles. When base quantile regression estimators are consistent for the true conditional quantiles (which requires structural smoothness conditions on the base learner and data distribution. Meinshausen, 2006), the calibration adjustment $Q_{1-\alpha}$ converges in probability to zero ($Q_{1-\alpha} \xrightarrow{p} 0$), asymptotically recovering conditional validity. In the absence of base-model consistency, CQR provides guaranteed marginal coverage, with interval width responding empirically to estimated local conditional dispersion.

---

### 6.3 Violation of Exchangeability: Weighted Conformal Prediction under Covariate Shift
Theorem 1 needs exchangeability of $(Z_1, \dots, Z_{n+1})$. Where it happens, this condition can fail under:
1. Autoregressive Correlation: Adaptive conformal inference (Gibbs & Candès, 2021) adjusts $\alpha_t$ dynamically to uphold coverage, under the temporal drift, and serial correlation $\text{Cov}(Y_t, Y_{t-k}) \ne 0$ violates permutation invariance in time-series forecasting.
2. Covariate Shift: Arises when the feature distribution drifts between training and test areas. In turn, in the meantime, the conditional response distribution remains invariant:
   $$
   P_{\text{train}}(X) \ne P_{\text{test}}(X), \quad P_{\text{train}}(Y \mid X) = P_{\text{test}}(Y \mid X).
   $$

Tibshirani et al. (2019) showed that **Weighted Conformal Prediction** restores exact finite-sample coverage under the covariate shift, provided that the exact covariate likelihood ratio $w(x) = \frac{dP_{\text{test}}(x)}{dP_{\text{train}}(x)}$ is known. The normalized weights are:
$$
p_i(X_{n+1}) = \frac{w(X_i)}{\sum_{j=1}^{n_2} w(X_j) + w(X_{n+1})}, \quad i \in \{1, \dots, n_2\},
$$
$$
p_{n+1}(X_{n+1}) = \frac{w(X_{n+1})}{\sum_{j=1}^{n_2} w(X_j) + w(X_{n+1})}.
$$
If the density ratio $w(x)$ is, more precisely, unknown. And more, must be estimated from finite samples (e.g., via logistic regression or kernel density estimation), the exact finite-sample coverage guarantee is usually relaxed to an asymptotic guarantee or bounded by the estimation error $\|\hat{w} - w\|$.

---

### 6.4 Empirical Limitations & Dataset Considerations
The findings of this study should be interpreted alongside several methodological considerations:
1. Monte Carlo Replication Budget: Standard mismatches around conformal coverage would further be tightened by larger replication counts ($M \ge 1,000$). While $M = 100$ Monte Carlo replications are offering enough statistical power to detect over-coverage, under the parametric misspecification ($p < 10^{-15}$), in real-world scenarios.
2. Sample-Splitting Overhead: Split conformal methods are needing partitioning data into the training and calibration subsets, lowering the sample size available for training the predictive model, at least for now.
3. Cross-Split Dependence in Benchmarking: Repeated random splits on the `MASS::Boston` dataset ($N = 506$) share data across the partitions, meaning checks across the splits are statistically dependent (Dietterich, 1998. Nadeau & Bengio, 2003). Findings on this historical benchmark serve as descriptive illustrations of steadiness rather than formal independent-sample hypothesis tests, broadly speaking. Finding larger modern high-dimensional tabular datasets represents a useful direction for future research.

---

## 7. Conclusion

This study checked standard parametric, bootstrap, Bayesian, and distribution-free conformal methods for uncertainty quantification in machine learning, at least in principle. Through Monte Carlo simulations and empirical checks in R, we showed while parametric OLS, bootstrap, and Bayesian Normal-Inverse-Gamma models are reaching best efficiency under Gaussian homoscedasticity, they are exhibiting systematic positive coverage bias under the likelihood misspecification due to outlier inflation of sample variance, at least in principle, but still, finite-sample slight coverage guarantees under the exchangeability, are being offered by Split Conformal Prediction and CQR upholding tested coverage consistent with small 90%, across all noise regimes. Conformalized Quantile Regression further adapts interval width to local heteroscedasticity, lessening proper Winkler scores relative to fixed-width conformal bands.

---

## 8. References

1. Angelopoulos, A. N. & Bates, S. (2023). Conformal prediction, like A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591.
2. Barber, R. F. Candès, E. J. Ramdas, A. & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *The Annals of Statistics*, 49(1), 486-507.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
4. Bürkner, P. C. (2017). brms: An R package for Bayesian multilevel models employing Stan. *Journal of Statistical Software*, 80(1), 1-28.
5. Chen, T. & Guestrin, C. (2016). XGBoost, like A adjustable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
6. Gibbs, C. & Candès, E. (2021). Adaptive conformal inference under the distribution shift. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 1660-1672.
7. Gneiting, T. & Raftery, A. E. (2007). Rigorously proper scoring rules, prediction and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.
8. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
9. Kleijn, B. J. K. & van der Vaart, A. W. (2012). The Bernstein-von-Mises theorem under the misspecification. *Electronic Journal of Statistics*, 6, 354-381.
10. Kuhn, M., & Wickham, H. (2020). *Tidymodels: a collection of packages for modeling and machine learning using tidyverse principles*. Posit / RStudio.
11. Kuhn, M., Vaughan, D., & Ruiz, E. (2025). *probably: Tools for Post-Processing Predicted Values*. R package version 1.2.0, CRAN.
12. Lei, J. G'Sell, M. Rinaldo, A. Tibshirani, R. J. & Wasserman, L. (2018), and distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094-1111.
13. Meinshausen, N. (2006). Quantile random forests. *Journal of Machine Learning Research*, 7, 983-999.
14. Papadopoulos, H. Proedrou, K. Vovk, V. & Gammerman, A. (2002). In fact, inductive confidence machines for regression. *European Conference on Machine Learning (ECML)*, 345-356. Springer.
15. Romano, Y. Patterson, E. & Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 3543-3553.
16. Tibshirani, R. J. Foygel Barber, R. Candès, E. & Ramdas, A. (2019). Indeed, conformal prediction under the covariate shift. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 2530-2540.
17. van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge University Press.
18. Vovk, V. (2012). Conditional validity of inductive conformal predictors. *Machine Learning*, 92(2), 349-376.
19. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer Science & Business Media.
