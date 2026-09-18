# delta_reasoner

From-scratch reproduction of **∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space** (ICLR 2026, [arXiv:2603.04948](https://arxiv.org/abs/2603.04948)) on 2 x Tesla T4.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/delta_reasoner/blob/main/nb/delta_reasoner.ipynb)

**Read [REPORT.md](REPORT.md) for the findings.** The notebook is the deliverable.

## The idea

Every other test-time scaling method is zeroth-order: sample candidates, score them, keep the best. But the reward model is differentiable, so you can ask a richer question — not "is this good?" but "which direction makes it better?" ∇-Reasoner does gradient descent on the text at inference time. The latent space is the **pre-softmax output logits**, not hidden states.

## Verdicts

| claim | verdict |
|---|---|
| C1 bidirectional gradient (Prop. C.1) | **SPLIT** — exact to 1.3e-14, but worth only +1.3σ in practice |
| C2 confident tokens are un-optimisable (Eq. 25) | **CONFIRMED + extended** — also dies at the *uniform* end, which the paper misses |
| C3 lower rejection rate (Tab. 3) | **CONFIRMED** — 49.2% vs 81.9% baseline |
| C4 test-time descent = KL-regularised RL (Thm. 4.1) | **CONFIRMED** to the noise floor (1.18x) |
| C5 beats BoN/SC at equal or fewer calls | **NOT REPRODUCED** — BoN wins at 26.6x less compute |
| C6 accelerations skip most work (App. D.2) | **CONFIRMED** — 84.2% vs 89.2% |

Plus: **gradient caching (Alg. 4) is exact, not an approximation** — the paper understates it.

## The substrate

2-digit multiplication with exact ground truth. The answer space is only 10,000, so we brute-force the **exact global optimum of Eq. 2** over every possible answer and measure everything against it. The paper cannot do this (their space is 151936^1024).

```
FLOOR   random answer                    0.01%
        greedy decoding                 28.90%
        exact optimum of Eq. 2          39.50%   <- CEILING
        reward-only argmax              17.00%   <- reward hacking limit
```

That last line justifies Eq. 2's design empirically, which the paper never does: dropping the log-likelihood term costs **22.5 accuracy points**.

## Layout

```
nb/delta_reasoner.ipynb   the whole thing, 11 rungs, runs top to bottom
REPORT.md                 claims, what broke, results, verdicts
out/results.json          every number in the report
```

Section 3 of the report ("What broke") is the part worth reading. Eight instrumentation bugs, two of which **flipped a conclusion's sign**.
