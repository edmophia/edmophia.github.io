---

layout: post
title: Notes on RKHS-WIP
date: 2018-10-31
tags: RKHS
mathjax: true

  - Reinforcement Learning

---

- [Notes on RKHS](#notes-on-rkhs)
  * [What is Kernel?](#what-is-kernel-)
  * [Three views on RKHS](#three-views-on-rkhs)
    + [Predict, Experience and Backpropagate](#predict--experience-and-backpropagate)

# Notes on RKHS

## What is Kernel?

> **Definition** Kernel
>
> A function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is a **positive semidefinite (PSD) kernel** (or more simply, a **kernel**) if and only if for every finite set of points $x_1, \cdots, x_n \in \mathcal{X}$, the **kernel matrix** $K \in \mathbb{R}^{n \times n}$ defined by $K_{ij} = k(x_i, x_j)$ is positive semidefinite.

* Note: It is a generalization of positive semidefinite (PSD) matrix in the sense that *every finite observation* of kernel is a PSD matrix.

## Three views on RKHS

So from now on, we would like to understand this diagram completely!!!

<img src="/Users/SungYub/Google Drive/Notes/three_view.png" height="150px">

* Feature map ($\phi : \mathcal{X}\rightarrow \mathbb{R}^d$): Yeah, that **feature map** which we encounter in machine learning everyday.
* Kernel : We already define what kernel means mathematically. Although this is enough to progress our story, stop here and check what situation do we meet kernel in detailed setting.

### Predict, Experience and Backpropagate

Now assume that we **predict** the output $y \in \mathbb{R}$ by
$$
\hat{y} = \big< w, \phi(x) \big>
$$
where $w \in \mathbb{R}^d$ is *weight vector*.

Now given answer of the predictions $y_1, \cdots, y_n$ for inputs $x_1, \cdots, x_n$ , our model **experience** loss by
$$
L(w) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2}\big( y_i - \big< w, \phi(x_i) \big> \big)^2
$$
and the **gradient** of this loss is
$$
\begin{eqnarray}
\nabla_w L(w) &= \frac{1}{n} \sum_{i=1}^n (y_i - \big< w, \phi(x_i) \big>)\phi(x_i)\\
&= \frac{1}{n} \sum_{i=1}^n \alpha_i \phi(x_i)
\end{eqnarray}
$$
where $\alpha_i = y_i - \big< w, \phi(x_i) \big>$ is a kind of *prediction error*.

If our optimization algorithm use only gradient information, the *approximate optimal solution* would be
$$
w^{OPT} = \sum_{t=1}^T\sum_{i=1}^n \alpha^t_i \phi(x_i)
$$


where $\alpha_i^t$ means prediction error of $i$-th data in $t$-th prediction.

So back to our *approximate optimal solution* the prediction can be denoted as
$$
\hat{y} = \big<w^{OPT}, \phi(x)\big> = \big< \sum_{t=1}^T\sum_{n=1}^n \alpha_i^t\phi(x_i), \phi(x)\big> = \sum_{t=1}^T\sum_{n=1}^n \alpha_i^t \big< \phi(x_i), \phi(x) \big>
$$
So if we define $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ as 
$$
k(x_i,x_j) \dot{=} \big< \phi(x_i),\phi(x_j)\big>
$$
the equation (6) is 
$$
\hat{y} =\sum_{t=1}^T\sum_{n=1}^n \alpha_i^t k(x_i, x).
$$

* Note: Soon we will see that $k(\cdot, \cdot)$ in equation (7) is a PSD kernel in the very first definition.



Now back to the three views on RKHS, we define our main subject **Reproducing Kernel Hilbert Space (RKHS)**.



> **Definition** Hilbert space
>
> A Hilbert space $\mathcal{H}$ is an complete vector space with an inner product $\big<\cdot, \cdot, \big>: \mathcal{H} \times \mathcal{H} \rightarrow \mathbb{R}$ that satisfies the following properties
>
> 1. **(Symmetry)** $\big< x, y \big> = \big< y, x \big>$,  $\forall x, y \in \mathcal{H}$.
> 2. **(Linearity)** $\big<c_1x_1, c_2x_2, y\big> = c_1 \big< x_1, y\big> + c_2 \big< x_2, y\big>$,  $\forall x_1,x_2,y \in \mathcal{H}, \forall c_1, c_2 \in \mathbb{F}$.
> 3. **(Positive Definite)** $\big< x, x \big> \ge 0$ and equality holds only if $x = 0$.

* Note1: The inner product of a Hilbert space $\mathcal{H}$ gives a natural norm $\|x\|_{\mathcal{H}} = \sqrt{\big<x,x\big>}$.
* Note2: The natural norm in Note1 gives a natural metric $d_{\mathcal{H}}(x,y) = \|x-y\|_{\mathcal{H}}$. So the complete vector space in above definition actually means *complete metric space*, any cauchy sequence converges. 

In Hilbert space, we call a function $L$ which maps an element of $\mathcal{H}$ to $\mathbb{R}$ (or more generally field), **functional**.

Moreover in most time, we are interested in the functional  $L$ is linear, i.e. $L(cx + y)  = cL(x) + L(y), \forall x, y \in \mathcal{H}, \forall c \in \mathbb{R}$.

Now take an example of linear functional.

> **Definition** Evaluation functional
>
> Let $\mathcal{H}$ is an Hilbert space consisting of functions $f: \mathcal{X} \rightarrow \mathbb{R}$. For each $x \in \mathcal{X}$, we can define the **evaluation functional** $L_x : \mathcal{H} \rightarrow \mathbb{R}$ as
> $$
> L_x(f) \dot{=} f(x)
> $$
> Note that the evaluation functional is a linear functional, i.e. $L_x(cf + g) = c L_x(f) + L_x(g), \forall f,g \in \mathcal{H}, \forall c \in \mathbb{R}$.



If a linear functional is **bounded** (or equivalently **continuous**), we can do various analysis about that functional.

> **Definition** Bounded functional
>
> Given a Hilbert space $\mathcal{H}$, a functional $L : \mathcal{H} \rightarrow \mathbb{R}$ is bounded if and only if there exists an $M < \infty$ suct that 
> $$
> | L(f) | \le M \|f\|_{\mathcal{H}}, \forall f \in \mathcal{H}
> $$
>

* Note1: If our functional $L$ is **linear**,  **boundedness** is a necessary and sufficient condition of  **continuity** .
* Note2: By Riesz, any bounded linear functional can be represented as a inner product between an element of dual space.

So our main subject **Reproducing Kernel Hilbert Space (RKHS)** demand that *any evaluation functional $L_x$ is a bounded linear functional.*

> **Definition** Reproducing Kernel Hilbert Space (RKHS)
>
> A Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}$ is a Hilbert space over functions $f : \mathcal{X} \rightarrow \mathbb{R}$ such that for each $x \in \mathcal{X}$, the evaluation functional $L_x$ is bounded.

