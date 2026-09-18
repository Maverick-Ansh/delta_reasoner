# ∇-Reasoner, reproduced from scratch

A from-scratch reproduction of **∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space** (Wang, Cai, Wang, Mei, Liu, Li, Wang, ICLR 2026, arXiv:2603.04948), run on 2 x Tesla T4.

Everything is in one notebook, `nb/delta_reasoner.ipynb`. Every number below is produced by a cell in it, and every claim is checked by an assertion that prints its own verdict.

---

## 1. What the paper claims, stated so it can come out false

| # | Claim | Where | Verdict |
|---|---|---|---|
| **C1** | Gradient flows **bidirectionally** along the sequence. Later tokens send signal back to earlier ones. Prior gradient-decoding work detaches this term. | Prop. C.1, Remark C.2 | **SPLIT** |
| **C2** | Gradient magnitude on logit `z_i` is proportional to its own probability `x_i`, so confident tokens cannot be moved | Eq. 25, App. C.3 | **CONFIRMED and extended** |
| **C3** | DTO's refined policy has a much lower rejection rate than blind resampling | Tab. 3 | **CONFIRMED** |
| **C4** | Inference-time gradient descent on Eq. 2 is dual to KL-regularised RL | Thm. 4.1 | **CONFIRMED** |
| **C5** | Beats greedy, BoN and SC at **equal or fewer model calls** | Tab. 1, Fig. 3, Fig. 4 | **NOT REPRODUCED** |
| **C6** | The three accelerations skip most of the work | App. D.2 | **CONFIRMED** |

C1 to C4 are mechanism claims. On a small substrate they can be checked **exactly**, to machine precision, with no benchmark noise to hide in. That is where most of the value of this reproduction sits. C5 is the headline and it is the one most at risk from resizing.

---

## 2. How it was resized, and what that cost

The paper uses Qwen-2.5-7B and Llama-3.1-8B policies with a 4B reward model on MATH-500, AIME and AMC. None of that fits on 2 x T4, and shrinking it would produce a noisy number on a hard task from which nothing could be concluded.

Instead the substrate is **2-digit multiplication** with exact ground truth:

```
prompt  "47*58="      answer  "2726"        vocabulary 13, answer length 4
```

The point of this choice is that the answer space is only 10,000, so we can score **every possible answer** under both the reward model and the language model and compute the **exact global optimum of Eq. 2** by brute force. The paper cannot do this, because their space is 151936^1024. That exact optimum is what every result below is measured against.

### Deviations

| paper | here | why this is still a test of the claim |
|---|---|---|
| Qwen-2.5-7B / Llama-3.1-8B | 0.8M-param, 4-layer transformer | C1 to C4 are statements about gradients through an autoregressive model plus a reward model. They do not depend on scale, and at this scale they are exactly checkable. |
| Skywork-Reward-V2-Qwen3-4B | 2.7M-param outcome reward model trained here | the paper requires the RM to share the policy's vocabulary. Ours does by construction. Its quality is measured, not assumed. |
| MATH-500 / AIME / AMC | 2-digit multiplication | exact ground truth, no grader, no parser, no annotator noise |
| answer length up to 1024 | **4 tokens** (and 13 in the length experiment) | **the material deviation.** Tested directly in section 6. |
| lambda not reported | 87.4, chosen from the exact ceiling curve | principled rather than guessed |
| learning rate 0.01 | 0.03 used, 0.01 also reported everywhere | our logit scale differs. Both are shown side by side. |
| T=20, N_max=8, temp 0.5, top-p 0.95, eps_ent 0.25, eps_grad 8, AdamW, cosine to 0.001 | identical | - |

The policy is trained with **label smoothing 0.1**. This is not cosmetic. Without it the model drives training loss to 1e-4, its softmax saturates, and section 3 shows the entire DTO gradient dies. Real LLMs see far more data than they can memorise and sit in the un-saturated regime.

---

## 3. What broke

This is the section nobody writes and it is the most useful one. Eight bugs, all caught by instrumentation rather than by luck, and two of them **changed a conclusion's sign**.

### 3.1 The reward model was a good classifier and a useless ranker

Version 1 of the RM drew its negatives from digit corruptions, near-misses and the policy's own wrong samples. It scored **93.3% balanced accuracy**, which looks fine.

Then the bracket cell asked it to rank all 10,000 candidate answers. Its argmax was correct **2.0%** of the time. The exact ceiling of Eq. 2 came out at 28.0% against greedy decoding's 27.7%, meaning **there was no headroom at all** and the entire experiment would have measured nothing.

The cause is that 10,000-wide ranking queries the model about candidates far outside the negative distribution it was trained on. The fix was to add **uniform negatives over the whole answer space**, which is the distribution the metric actually asks about. Ceiling went from 28.0% to **39.5%**, headroom from 0.2 points to **10.6 points**.

The general lesson: a metric that runs through a learned component must have its ceiling and floor measured before any compute is spent. Balanced accuracy on the training distribution predicted nothing about ranking accuracy off it.

### 3.2 The policy metric was measuring sharpening, not reasoning

The natural metric for "did DTO improve the policy" is the change in the probability of the correct next token, `delta p`. Measured naively across all token slots it gave clean, many-sigma answers, and they were wrong.

The giveaway was a control arm. `delta_reward detached` removes the reward gradient, so it optimises `-log pi` alone and cannot know anything about correctness. **It scored best of all three arms.**

The reason is that maximising log-likelihood **sharpens** the distribution toward its own argmax, and the greedy token is already correct on 72.9% of slots. So sharpening mechanically raises `p(correct)` on 72.9% of slots and lowers it on the rest, netting a large positive number with zero information about the answer.

Two fixes, both applied:

1. a **matched-KL null arm** that only sharpens (`z -> c*z`), with `c` tuned so it moves the distribution by the same KL as the DTO arm it is compared against. It has an identical movement budget and no gradient information. Anything real must beat it.
2. **split by whether the greedy token was already correct.** On already-correct slots sharpening is free. The discriminating slots are the ones where greedy is wrong, which is also exactly the subset test-time scaling exists to fix.

The null arm scored **+0.0426**, beating every DTO arm's naive score of **+0.0127 to +0.0340**. The metric was almost entirely sharpening.

After the fix, on the 3,249 discriminating slots:

| arm | vs matched null | verdict |
|---|---|---|
| DTO full | **+7.8 sigma** | beats the null |
| delta_postfix detached | +6.1 sigma | beats the null |
| delta_reward detached | **-6.4 sigma** | **loses to the null**, as it must |

The third row is the instrument validating itself. And the conclusion about C1 **flipped sign**: the naive metric said `delta_postfix` hurts by -0.0012, the corrected metric says it helps by +0.0013.

### 3.3 Theorem 4.1 took three attempts, and the first two failures were instrument failures

- **v1.** Total variation measured on a 220x220 grid with 200k particles. That is 48k bins for 200k samples, so the **Poisson noise floor alone was TV = 0.111**. The comparison could not resolve anything. Fixed by measuring TV on a coarse 64x64 grid, which dropped the floor to 0.023 and then **exposed a real bias of 0.196**.
- **v2.** The real bias had two causes: Euler-Maruyama discretisation error, and a `clamp()` that **pinned overshooting particles onto the domain boundary** instead of reflecting them. Fixed with a Metropolis accept/reject step (MALA), which is asymptotically exact.
- **v3.** A residual of 0.164 remained at 99.5% acceptance. Near-100% acceptance is a **symptom, not a success**: the proposal was so timid that nothing was ever rejected because nothing ever moved. Raising the step size to dt=0.05 (94% acceptance) brought TV to **0.0256 against a floor of 0.0217**.

Only at that point was the theorem actually being tested.

### 3.4 A device-side assert that killed the CUDA context

The policy can emit `*` or `=` in an answer slot. A naive base-10 decode of those token ids overflows 9999, `one_hot` indexes out of bounds, and on CUDA that is a device-side assert which **poisons the context for the entire process**. Ten minutes of reward-model training were lost.

Fixed with a single guarded `answer_index()` that every conversion goes through, plus disk checkpointing of every trained model so a crash costs seconds rather than minutes.

### 3.5 Asserting the wrong objective decreases

The obvious smoke test, "DTO should decrease the loss it is minimising", fails. With straight-through the forward pass sees a hard one-hot, so `L(hard y)` is **piecewise constant** in `z`: it changes only when an argmax flips, and an individual flip can go either way.

DTO descends the **relaxed** objective, which falls monotonically from -46.95 to -52.10 while the hard-y value barely moves. This is the same fact as "DTO almost never flips a token", seen from the loss side.

### 3.6 The length experiment's first design confounded two variables

Moving to chain-of-thought answers raised greedy accuracy from 28.90% to **94.52%**, because decomposing the product into steps makes the task far easier. That changes **difficulty as well as length**, so the comparison would have tested both and isolated neither.

Fixed by holding architecture, optimiser and format fixed and shrinking the training set until greedy accuracy matched: **N_train = 2100 gives 33.36%** against the short world's 28.90%. The transition is sharp (500 rows gives 3.4%, 5000 gives 94.5%), so the search had to bracket it.

---

## 4. Results

### The bracket, which everything is read against

```
FLOOR   random 4-digit answer                 0.01%
        greedy decoding                      28.90%   <- what must be beaten
        MAP sequence under pi_LLM            27.50%   <- best possible WITHOUT reward
        exact optimum of Eq. 2, lambda=87.4  39.50%   <- CEILING
        reward-only argmax (lambda -> inf)   17.00%   <- reward hacking limit
```

The last line is worth pausing on. It is Eq. 2's design justified empirically, which the paper never does: **dropping the log-likelihood term costs 22.5 accuracy points.** The lambda curve is a clean inverted U, rising from 27.5% to 39.5% and collapsing to 17.0%. That curve *is* the reward-hacking tradeoff, measured.

### Main comparison, 2000 held-out problems, 3 seeds

| method | accuracy | model calls | rejection rate |
|---|---|---|---|
| greedy decoding | 28.80% | 4 | - |
| Self-Consistency (N=8) | 28.72% +/- 0.23 | 32 | - |
| **Best-of-N (N=8)** | **33.00% +/- 0.04** | 40 | 81.9% |
| ∇-Reasoner (Alg. 1, basic) | 30.08% +/- 0.24 | 258 | 54.4% |
| ∇-Reasoner (Alg. 3, accelerated) | 28.37% +/- 0.24 | 266 | 49.2% |
| ... with delta_postfix detached | 28.38% +/- 0.19 | 266 | 50.6% |

### Test-time scaling, matched on model calls

| calls | BoN | SC | | calls | ∇-Reasoner |
|---|---|---|---|---|---|
| 10 | **29.95** | 27.00 | | 12 | 26.80 |
| 20 | 31.85 | 27.65 | | 264 | 28.30 |
| 40 | 33.25 | 27.70 | | 266 | 28.20 |
| 80 | 34.45 | 27.75 | | | |
| 160 | 35.35 | 28.25 | | | |
| 280 | 35.70 | 28.20 | | | |

**BoN at 10 calls beats ∇-Reasoner at 266 calls.** That is 26.6x cheaper. The lambda and learning-rate sweep (15 configurations) did not change this: the best ∇-Reasoner configuration found was lambda=87.4, lr=0.1 at 28.20%.

Note also that **BoN never saturates** here, climbing steadily toward the 39.5% ceiling. The paper's Fig. 4 shows BoN flattening while ∇-Reasoner keeps improving, and the entire comparison depends on that flattening.

---

## 5. Verdict on each claim

**C1, bidirectional gradient. SPLIT.**

The mechanism is real and exact:
- Prop. C.1's three-way split reproduces autograd to **1.3e-14**
- Eq. 4's closed form matches autograd to **exactly 0.0**, so `delta_prefix` really is `-log Cat(pi(.|y_<l))` and not merely "some gradient"
- `delta_postfix` is non-zero at every non-final token, is largest at the **first** token (which has the most future), and is **exactly zero** at the last token, which has no future

But the practical benefit Remark C.2 implies is not detectable here. On the discriminating subset it is worth **+0.0013 +/- 0.0010, which is +1.3 sigma**. It carries about **2.1%** of projected gradient norm. With 4-token answers there is almost no future for it to sum over, which is exactly what section 6 tests.

**The finding that matters more:** on the saturated policy `delta_postfix` is **6.8 million times smaller**, and the *entire* gradient collapses (projected `delta_prefix` of 1e-5 against the calibrated policy's 0.43). A model trained to near-zero loss is completely immune to DTO. This is Eq. 25 operating at the sequence level, and it is not mentioned in the paper.

**C2, Eq. 25. CONFIRMED, and extended.**

Reproduces autograd exactly. Gradient norm collapses to **<0.01%** of peak in the confident limit, confirming App. C.3.

Two things the paper does not say:
1. the gradient is maximal in the **middle** of the entropy range (peak at H = 3.07 nats) and **also collapses to 2.2% at the near-uniform end**, because `x_i` multiplies every component and a near-uniform token has every `x_i ≈ 1/|V|`. The `eps_ent` criterion guards only one end. Nothing in Alg. 3 checks the other.
2. the `eps_ent = 0.25` threshold is **not free**. Right at the threshold the gradient is still **12.6% of peak**. It sits on the shoulder of the collapse, not past it. It is a good trade, but it is a trade.

**C3, rejection rate. CONFIRMED.**

| | paper | here |
|---|---|---|
| baseline (BoN) | 65.9% | 81.9% |
| ∇-Reasoner | 32.8% | 49.2% |
| relative reduction | 50% | **40%** |

Our BoN baseline exceeds the theoretical 66.0% because the theory assumes N distinct draws. At temperature 0.5 the policy produces only **2.17 distinct answers out of 8**, and a duplicate can never set a new reward maximum. The direction and rough magnitude of the paper's claim hold.

**C4, Theorem 4.1. CONFIRMED to the noise floor.**

Three independent computations of the same distribution on a 2-D space:

| comparison | TV | meaning |
|---|---|---|
| A closed form vs B argmin of L_PPO | **0.00023** | Eq. 3 was read correctly |
| B argmin of L_PPO vs C Langevin | **0.02563** | **the theorem** |
| two runs of the same law | 0.02166 | the noise floor |
| pi_LLM vs rho* | 0.45230 | how far reward moves it |

Langevin closes **94.3%** of the gap and lands at **1.18x the noise floor**. Path B is gradient descent over a 65,536-dimensional simplex and knows nothing about Langevin dynamics. Path C is 400,000 independent noisy particles. They agree.

The per-mode masses confirm what has to happen: reward more than doubles mode 1's share (0.345 to 0.782) and guts mode 2 (0.206 to 0.057). That is not local reshaping, whole particles must **cross between modes**.

**And that is the practical limitation.** Thm. 4.1 is asymptotic. It promises the destination, never the rate, and the rate is exponentially bad exactly when reward wants mass moved between distant modes. Token sequences are the extreme case of separated modes. Measured directly on the real algorithm: DTO shifts logits by about 0.1 against a top-1/top-2 gap of **3.16**, so only **1.7%** of tokens have a gap it could close. It re-weights within a mode and essentially never migrates between them. That is precisely why Alg. 1 must wrap DTO in resampling and rejection with N_max = 8 rollouts. The rollouts supply the global moves the gradient cannot.

**C5, beats baselines at equal or fewer calls. NOT REPRODUCED.**

BoN at 10 calls (29.95%) beats the best ∇-Reasoner configuration at 266 calls (28.20%). See section 6 for the mechanism.

**C6, accelerations. CONFIRMED.**

**84.2%** of token-optimisation steps skipped against the paper's 89.2%, and 7,760 rollouts reused.

One surprise: the split between the two criteria is nothing like even. The **entropy** criterion fired on **1 slot out of 8,000**. The **gradient-norm** criterion did all 84.2% of the work. In this setup `eps_ent` is inert and `eps_grad` is the whole acceleration.

**Gradient caching (Alg. 4), a result the paper understates.**

The paper presents caching as an acceleration, which invites reading it as a speed-for-accuracy trade. It is not a trade. Both models are frozen and the prefix is fixed, so `L` is a function of `y` alone. If `y` is unchanged then `dL/dy` is not approximately the same, it is **the same number**, and the surrogate is built so that the recovered gradient is exact while the `dz/dy` factor is recomputed fresh each step. Verified: cached and uncached trajectories agree to floating-point precision.

---

## 6. The length experiment

C5 failed, and a reproduction owes a mechanism rather than a verdict. The hypothesis comes straight out of the earlier rungs: **everything ∇-Reasoner does well scales with answer length, and ours are 4 tokens against the paper's 1024.**

- `delta_postfix` sums over tokens *after* position l. With 4 tokens there is almost nothing to sum.
- the cost argument rests on one gradient step updating all tokens at once while autoregression emits one per call. That advantage is proportional to length. At length 4 it is worth almost nothing.
- BoN is unusually strong here and never saturates, and the paper's comparison depends on BoN saturating.

So the short-answer regime is close to the worst case for the method and the best case for the baseline.

The test keeps everything fixed and makes the answer longer, from a bare product to a chain of thought with real intermediate structure, where earlier tokens constrain later ones through the partial products:

```
short world (4 tokens)  :  47*58=  ->  2726
long  world (13 tokens) :  47*58=  ->  329+2350=2679
```

Greedy accuracy is matched across worlds (33.36% vs 28.90%) by shrinking the long world's training set, so **length is the only variable**. Policy, reward model and ceiling are all retrained from scratch for the new format.

**Status: the matched long world is built and validated. The final head-to-head measurement did not finish before the session ended.** The cell that runs it is in the notebook and is the first thing to execute on resume. The quantity it reports is not either world's absolute accuracy, it is how the **∇-Reasoner-minus-BoN gap moves** between them. If the gap shrinks, the length hypothesis is supported. If it does not move, the hypothesis is wrong and the failure is about something else.

---

## 7. What was not tested

Being exhaustive and unglamorous about this, because the gaps matter:

- **Real LLMs.** No Qwen, no Llama, no Skywork. Every claim here is tested on a 0.8M-parameter model. C1 to C4 are scale-independent statements and are checked exactly. C5 is not, and its failure here is explicitly scoped to this regime.
- **Real benchmarks.** No MATH-500, AIME, AMC or GSM8K.
- **The length experiment's conclusion.** Built and validated, not yet measured. This is the single most important open item.
- **The attribution-graph section (Rung 10).** Transcoders and the frozen-attention linearisation are implemented and the replacement model is defined, but it was not run end to end, so no circuit claims are made.
- **Process rewards.** Only outcome reward, as in the paper's main experiments. App. B.2's generalisation is implemented in the loss but not evaluated.
- **The wall-clock claim (Tab. 5).** The paper's own numbers show ∇-Reasoner doing 26x the arithmetic of BoN (2.46e17 vs 9.54e15 FLOPs) while claiming comparable wall-clock through better hardware utilisation. That is a serving-stack claim and cannot be tested on 2 T4s. We count model calls the paper's way, which is generous to the method, and say so.
- **Reward hacking under the rejection loop.** We measured that reward-only maximisation costs 22.5 points, but not whether accepted proposals specifically are reward-hacked. The rejection test uses the same reward model DTO is exploiting, so it is structurally unable to detect hacking. This is a real concern that deserves its own experiment.
- **Only 2 to 3 seeds**, and one substrate. Seed spread is reported everywhere and is small relative to the C5 effect, but not to the C1 effect.

## 8. Reproducing

`nb/delta_reasoner.ipynb` runs top to bottom on a 2 x T4 Kaggle or Colab GPU session. Trained weights are cached to `out/*.pt`, so a re-run after the first is fast. `out/results.json` carries every number in this report.

The notebook is the deliverable and is meant to be read in order. Each rung builds one concept from scratch, checks it numerically, and only then hands it to the next rung.
