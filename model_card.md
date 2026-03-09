# Model Card: Adaptive GP-UCB Optimizer

## Overview
* **Name:** Adaptive Gaussian Process Optimizer (GP-UCB)
* **Type:** Bayesian Optimization Surrogate Model
* **Version:** 1.0

## Intended Use
This model is highly suitable for optimizing continuous, expensive-to-evaluate black-box functions in low-to-moderate dimensions (up to 8D) where query budgets are strictly limited. It should be avoided for high-frequency trading, real-time control systems, or optimization tasks exceeding 20 dimensions due to computational scaling limits.

## Details
The strategy utilizes a scikit-learn Gaussian Process Regressor equipped with a dynamically bounded Matern kernel ($\nu$ = 1.5 or 2.5). Over ten rounds, the approach evolved from static exploration to highly adaptive, function-specific tuning. The exploration parameter, beta, was manually adjusted weekly—ranging from aggressive exploration ($\beta = 5.0$) on flat landscapes to hyper-exploitation ($\beta = 0.01$) on confirmed peaks.



## Performance
Success was measured by the maximum scalar output achieved per function. The model performed exceptionally well on functions with distinct gradients, securing massive peaks on Function 5 (4089.90) and Function 8 (9.94). It struggled on flat or highly deceptive landscapes, such as Function 1.

## Assumptions and Limitations
The core assumption is that the underlying black-box functions are continuous and relatively smooth. The primary limitation is the curse of dimensionality; the strict 10-query limit means the model cannot robustly map the 6D and 8D spaces, relying heavily on early lucky gradients. Furthermore, the hyper-exploitation strategy guarantees local minima entrapment if a taller, unmapped peak exists elsewhere.

## Ethical Considerations and Transparency
By explicitly logging the weekly adjustments to the beta parameter, this model card ensures full reproducibility of the decision-making process. This transparency helps future practitioners understand the real-world trade-offs between exploration costs and exploitation risks when working with incomplete information.
