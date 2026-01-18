# captstone-project

# Black-Box Optimization (BBO) Capstone Project

## Section 1: Project Overview
**Goal:** The goal of this project is to optimize a series of eight unknown "black-box" functions where the internal mathematical structure (formula, gradients, or convexity) is hidden. The objective is to find the input coordinates that maximize the output of each function using a limited budget of queries.

**Relevance:** In real-world Machine Learning, we often face expensive optimization problems where we cannot simply calculate a derivative (gradient descent). Examples include hyperparameter tuning for neural networks, A/B testing in marketing, or drug discovery simulations. This project simulates that constraint: we must make intelligent decisions based on incomplete data rather than brute-force guessing.

**Career Impact:** Mastering Bayesian Optimization allows me to tackle problems where data labeling is expensive or time-consuming. It demonstrates the ability to build "data-efficient" AI agents that can learn optimal strategies with minimal trial and error, a highly valued skill in automated machine learning (AutoML) and operations research.

## Section 2: Inputs and Outputs
The model interacts with a server that hosts eight distinct functions ranging from 2 dimensions to 8 dimensions.

* **Inputs:** A set of continuous coordinates within the unit hypercube `[0, 1]^d`.
    * *Format:* `x1-x2-...-xd` (e.g., `0.523412-0.123958` for a 2D function).
    * *Constraint:* All values must be bounded between 0 and 1.
* **Outputs:** A single scalar value (float) representing the score of that point.
    * *Example:* Input `0.5-0.5` → Output `2788.27`.
    * *Constraint:* The output scale varies wildly between functions (from `1e-65` to `2000+`).

## Section 3: Challenge Objectives
**Primary Goal:** Maximize the scalar output for all eight functions.

**Constraints & Challenges:**
1.  **Limited Budget:** We are restricted to a small batch of queries per week. We cannot simply map the entire space.
2.  **Varying Dimensions:** The problem set includes low-dimensional (2D) functions which are easy to visualize, and high-dimensional (8D) functions that suffer from the "Curse of Dimensionality."
3.  **Noisy/Sparse Data:** Initial outputs can be extremely sparse (e.g., `1e-42`), making it difficult for standard models to distinguish signal from noise without normalization.

## Section 4: Technical Approach
My core strategy relies on **Bayesian Optimization** using a **Gaussian Process (GP)** surrogate model. This probabilistic approach allows me to model both the *predicted value* (mean) and the *uncertainty* (standard deviation) of the unseen landscape.

### Strategies Implemented (Weeks 1-3)

**1. Surrogate Modeling**
* I utilize a **Matern Kernel** (nu=2.5 or nu=1.5) to model non-linear functions. This kernel allows the GP to adapt to rougher terrains compared to a standard RBF kernel.
* **Normalization:** Inputs are scaled to `[0,1]`, and outputs are normalized to handle the extreme variance in magnitude across functions.

**2. Acquisition Function (UCB)**
* I use the **Upper Confidence Bound (UCB)** heuristic: `Score = Mean + (Beta * Std)`.
* This balances **Exploitation** (trusting the Mean) and **Exploration** (trusting the Uncertainty/Std).

**3. Adaptive Exploration Strategy**
* **Week 1-2:** Used a static `beta = 1.96` (95% confidence interval) to establish a baseline.
* **Week 3 (Current):** Implemented **Function-Specific Hyperparameters**.
    * **Exploitation:** For functions where a high peak was found (e.g., Function 5), I lowered `beta` to **0.5** to refine the maximum.
    * **Aggressive Exploration:** For functions stuck in local minima or flatlines (e.g., Function 1), I increased `beta` to **5.0** to force the model to jump to unexplored regions.
    * **High Dimensions:** For 8D functions, I increased the number of random candidate samples (150k+) to ensure adequate coverage of the search space.

This iterative approach ensures that resources are not wasted exploring regions we already know are poor, while preventing the model from getting stuck in local optima early in the process.
