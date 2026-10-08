# Machine Learning Task — 12: Collaborative Filtering & Matrix Factorization

> **Goal:** Move past content-based filtering's limits by learning to recommend using *other users'* behavior — first with intuitive memory-based collaborative filtering (user-based and item-based), then with the industry-standard technique: matrix factorization, trained via SGD and WALS.

> **Note:** This continues the Recommendation Systems track (Task 11 → Task 12). Task 11 used only item features. Task 12 drops that requirement entirely — collaborative filtering can recommend based purely on the `users × items` feedback matrix, discovering latent taste patterns no human engineered.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What You Will Learn](#2-what-you-will-learn)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Phase 1 — Collaborative Filtering & the Embedding Intuition](#phase-1--collaborative-filtering--the-embedding-intuition)
5. [Phase 2 — Memory-Based CF: User-Based Filtering](#phase-2--memory-based-cf-user-based-filtering)
6. [Phase 3 — Memory-Based CF: Item-Based Filtering](#phase-3--memory-based-cf-item-based-filtering)
7. [Phase 4 — Matrix Factorization Fundamentals](#phase-4--matrix-factorization-fundamentals)
8. [Phase 5 — Weighted Matrix Factorization: SGD vs. WALS](#phase-5--weighted-matrix-factorization-sgd-vs-wals)
9. [Phase 6 — Implicit Feedback & Alternating Least Squares (ALS)](#phase-6--implicit-feedback--alternating-least-squares-als)
10. [Phase 7 — Bayesian Personalized Ranking (BPR)](#phase-7--bayesian-personalized-ranking-bpr)
11. [Phase 8 — Strengths, Weaknesses & When Matrix Factorization Breaks Down](#phase-8--strengths-weaknesses--when-matrix-factorization-breaks-down)
12. [Exercises (Must Do)](#exercises-must-do)
13. [Deliverables Checklist](#deliverables-checklist)
14. [Reference: Key Concepts Glossary](#reference-key-concepts-glossary)
15. [Resources](#resources)

---

## 1. Project Overview

### The Scenario

Content-based filtering (Task 11) can only ever recommend things *like* what a user already likes. Collaborative filtering (CF) does something more powerful: it finds users with similar taste and recommends what *they* liked — even items that look nothing alike on paper. Task 12 builds CF from the ground up, from the simplest "find similar users" approach to matrix factorization, the technique that won the $1M Netflix Prize and still underlies most production recommenders today.

### Why This Matters

| Concept | Where It Shows Up in Practice |
|---|---|
| User-based / item-based CF | The "customers who bought this also bought" pattern everywhere in e-commerce |
| Matrix factorization | The Netflix Prize-winning technique; still the default baseline at most companies |
| Embeddings | The same user/item embedding idea reappears in Task 14's deep models and Task 15's two-tower retrieval |
| SGD vs. WALS | A real engineering tradeoff: flexibility and side-feature support (SGD) vs. speed and native unobserved-data handling (WALS) |
| Implicit feedback / ALS | How you factorize a matrix when all you have is clicks and watches, not star ratings |
| BPR | The standard loss function when you care about *ranking* items correctly, not predicting exact ratings |

---

## 2. What You Will Learn

By completing this task, you will be able to:

- Explain collaborative filtering's core idea: use similarities *between users and items*, discovered from the feedback matrix alone, to recommend
- Build intuition for embeddings via the 1D and 2D "child movie vs. adult movie" / "blockbuster vs. arthouse" examples, and explain why the dot product of a user embedding and item embedding approximates their feedback-matrix entry
- Implement user-based CF: compute user-user similarity (Pearson correlation or cosine) and predict a rating as a similarity-weighted average of similar users' ratings
- Implement item-based CF: compute item-item similarity (adjusted cosine) and predict a rating from the user's own ratings on similar items
- Explain why item-based CF is generally preferred over user-based CF in production (item similarity is far more stable over time than user similarity)
- Write matrix factorization's objective function: minimize `Σ(i,j)∈obs (A_ij - ⟨U_i, V_j⟩)²` over user embedding matrix `U` and item embedding matrix `V`
- Explain why summing only over *observed* entries is insufficient (collapses to a trivial solution), and how Weighted Matrix Factorization fixes this by also summing a down-weighted term over unobserved entries
- Compare SGD and WALS as optimization algorithms: SGD is flexible (easy to add side features and other loss functions) but can be slow and tricky to tune; WALS converges faster and handles unobserved entries natively but is specialized to the squared-loss objective
- Explain why implicit feedback (clicks/watches, no explicit dislikes) needs a different factorization treatment, and implement it with Alternating Least Squares (ALS) via the `implicit` library
- Write the Bayesian Personalized Ranking (BPR) pairwise objective and explain why it directly optimizes ranking order rather than predicting individual scores
- List matrix factorization's limitations (cold start for new users/items, difficulty folding in side features, popularity bias with dot-product similarity) — motivating Task 14's deep learning approaches

---

## 3. Prerequisites & Setup

### Required Knowledge

- Task 11 (feedback matrix, explicit vs. implicit feedback)
- Task 3 (Pearson correlation, covariance)
- Task 10 (SVD, from PCA) — matrix factorization reuses this directly

### Environment Setup

```bash
pip install implicit scikit-surprise --break-system-packages
python3 -c "import numpy, pandas, scipy, implicit"
```

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
```

### Dataset

Continue with MovieLens (`ratings.csv`, `movies.csv`) from Task 11.

---

## Phase 1 — Collaborative Filtering & the Embedding Intuition



### Tasks to Perform

1. Explain collaborative filtering's core idea in one sentence: use similarities between users *and* items, mined from the feedback matrix, to recommend — with no item-feature engineering required.
2. State the key advantage over content-based filtering: serendipity — CF can recommend item A to user X based on the taste of a similar user Y, even if A looks nothing like anything X has rated before.
3. Walk through the 1D embedding intuition: imagine assigning every movie a single scalar in `[-1, 1]` for "kids' movie" (-1) vs. "adult movie" (+1), and every user a scalar for their taste on the same axis — the product of the two approximates whether that user will like that movie.
4. Walk through the 2D extension: add a second axis for "blockbuster" vs. "arthouse," and note that now the *dot product* of the user and movie's 2D embeddings approximates the feedback value.
5. State the key insight: in practice, these embeddings are not hand-designed — they are *learned automatically* from the feedback matrix alone. This automatic learning is the power of collaborative filtering.
6. Explain why users and items can be represented in the *same* embedding space even though they're different entity types (the space is just a shared coordinate system for measuring affinity via similarity metrics).

**Deliverable for Phase 1:** A written walkthrough of the 1D → 2D embedding example in your own words, ending with a clear statement of what "learning embeddings automatically" means.

---

## Phase 2 — Memory-Based CF: User-Based Filtering



### Tasks to Perform

1. Explain user-based CF: to predict how user `u` will rate item `i`, find the `k` users most similar to `u` who have rated `i`, and take a similarity-weighted average of their ratings.
2. Write the Pearson correlation similarity formula between two users `u` and `v` (reuse Task 3's covariance/correlation formula, applied to their co-rated items only).
3. Build the user-user similarity matrix on a (small, for compute reasons) subset of MovieLens users using Pearson correlation.
4. Write the rating prediction formula: `pred(u,i) = r̄_u + (Σ_v sim(u,v)·(r_vi - r̄_v)) / Σ_v |sim(u,v)|`, and explain why it's centered on each user's mean rating (to correct for users who rate everything higher/lower on average).
5. Predict a rating for one (user, unrated-movie) pair by hand using this formula, and compare it to the user's actual rating if you held one out.
6. State user-based CF's core weakness: the user-user similarity matrix must be recomputed as new users/ratings arrive, and it scales poorly (`O(m²)` in the number of users).

### 2.1 Formula Reference

```
Pearson similarity(u,v) = Cov(r_u, r_v) / (σ_u · σ_v)     (over items both rated)

Predicted rating:
pred(u,i) = r̄_u + [Σ_v sim(u,v)·(r_vi - r̄_v)] / Σ_v |sim(u,v)|
```

**Deliverable for Phase 2:** The user-user similarity matrix for a subset of users, one fully hand-worked rating prediction, and a comparison against the true held-out rating.

---

## Phase 3 — Memory-Based CF: Item-Based Filtering



### Tasks to Perform

1. Explain item-based CF: to predict how user `u` will rate item `i`, find items similar to `i` that `u` has already rated, and take a similarity-weighted average of those ratings.
2. Write the adjusted cosine similarity formula between two items (cosine similarity computed after subtracting each user's mean rating — this corrects for different users' rating scales, unlike plain cosine).
3. Build the item-item similarity matrix for a subset of MovieLens movies.
4. Predict a rating for a (user, unrated-movie) pair using item-based CF and compare the result (and the computation cost) to Phase 2's user-based prediction.
5. State why item-based CF is generally preferred in production over user-based CF: item-item similarity is far more stable over time (a movie's relationship to other movies barely changes; a user's taste and your pool of "similar users" shifts constantly), so the similarity matrix can be precomputed offline and reused.

### 3.1 Formula Reference

```
Adjusted cosine(i,j) = Σ_u (r_ui - r̄_u)(r_uj - r̄_u) / [√Σ_u(r_ui-r̄_u)² · √Σ_u(r_uj-r̄_u)²]
   (sum over users u who rated both items i and j)
```

**Deliverable for Phase 3:** The item-item similarity matrix, one hand-worked prediction, and a written comparison of user-based vs. item-based CF (accuracy, stability, compute cost).

---

## Phase 4 — Matrix Factorization Fundamentals



### Tasks to Perform

1. State matrix factorization's goal: given the feedback matrix `A ∈ ℝ^(m×n)` (`m` users, `n` items), learn a user embedding matrix `U ∈ ℝ^(m×d)` and item embedding matrix `V ∈ ℝ^(n×d)` such that `U·Vᵀ` closely approximates `A`.
2. Write the basic objective function: `min(U,V) Σ(i,j)∈obs (A_ij - ⟨U_i, V_j⟩)²` — summed only over observed (non-missing) entries.
3. Explain why naively treating all unobserved entries as 0 and summing over the *entire* matrix is a bad idea (it would push the model toward predicting 0 — "not interested" — for everything, including items the user simply hasn't seen yet).
4. Explain why solving the observed-only objective via plain SVD is also problematic in practice (real feedback matrices are far too sparse for SVD to produce a good factorization directly).
5. Factorize a small dense toy ratings matrix using `np.linalg.svd` (reusing Task 10's SVD skills) and compare the reconstructed matrix to the original.
6. Pick a trained embedding dimension `d` (e.g., `d=20`) and explain the tradeoff: too small underfits (can't capture enough latent taste dimensions); too large overfits and is slower to train/serve.

### 4.1 Formula Reference

```
Given feedback matrix A ∈ ℝ^(m×n):
   Learn U ∈ ℝ^(m×d)  (user embeddings, row i = user i)
       V ∈ ℝ^(n×d)  (item embeddings, row j = item j)
   such that  U·Vᵀ ≈ A

Objective (observed-only):
   min(U,V)  Σ_{(i,j)∈obs} (A_ij - ⟨U_i, V_j⟩)²
```

**Deliverable for Phase 4:** The SVD-based factorization of a toy matrix with reconstruction comparison, and a written explanation of why observed-only SVD fails on real, sparse feedback matrices.

---

## Phase 5 — Weighted Matrix Factorization: SGD vs. WALS



### Tasks to Perform

1. Write the Weighted Matrix Factorization (WMF) objective, which splits the loss into two sums: one over observed entries, one (down-weighted by hyperparameter `w0`) over *all* unobserved entries.
2. Explain what `w0` controls: it balances how much the model is penalized for getting observed interactions wrong vs. how much it's nudged toward predicting "not interested" for unobserved pairs.
3. Explain Stochastic Gradient Descent (SGD) as a generic way to minimize this objective (reuse Task 7's SGD skills directly) — flexible enough to add other loss functions or side features, but can converge slowly and is sensitive to hyperparameters.
4. Explain Weighted Alternating Least Squares (WALS): since the objective is quadratic in `U` with `V` fixed (and vice versa), WALS alternates — fix `V`, solve for `U` in closed form; fix `U`, solve for `V` in closed form; repeat until convergence.
5. State WALS's key practical advantages: converges faster than SGD, and can be parallelized efficiently; its main limitation is that it's specialized to this particular squared-loss objective (not as easily extended to arbitrary loss functions).
6. Train a matrix factorization model on MovieLens using `scikit-surprise`'s `SVD` (which implements a regularized SGD-based factorization) and report the RMSE on a held-out test split.

### 5.1 Formula & Code Reference

```
WMF objective:
   min(U,V)  Σ_{(i,j)∈obs} (A_ij - ⟨U_i,V_j⟩)²  +  w0 · Σ_{(i,j)∉obs} (0 - ⟨U_i,V_j⟩)²
```

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)

algo = SVD(n_factors=20)
algo.fit(trainset)
predictions = algo.test(testset)
print(accuracy.rmse(predictions))
```

**Deliverable for Phase 5:** The trained SVD model's RMSE, and a written comparison table of SGD vs. WALS (speed, flexibility, handling of unobserved data).

---

## Phase 6 — Implicit Feedback & Alternating Least Squares (ALS)



### Tasks to Perform

1. Explain why rating-prediction matrix factorization (Phases 4–5) doesn't transparently apply to implicit feedback: there are no negative examples, only "observed" (watched/clicked) and "unknown" (which could mean "disliked" or simply "never shown").
2. Convert the MovieLens ratings into a binary/confidence-weighted implicit matrix (e.g., `confidence = 1 + α·rating_count` for watched items, as in the Hu, Koren & Volinsky implicit-ALS formulation).
3. Train an implicit-feedback ALS model using the `implicit` library's `AlternatingLeastSquares` class on the binary matrix.
4. Retrieve the top-N recommendations for a sample user and inspect whether they make intuitive sense given that user's watch history.
5. State the key conceptual difference from Phase 5: implicit ALS optimizes a *confidence-weighted* reconstruction of a binary preference matrix, rather than predicting an explicit star rating.

### 6.1 Code Reference

```python
import implicit
from scipy.sparse import csr_matrix

user_item_csr = csr_matrix((ratings["rating"], (ratings["userId"], ratings["movieId"])))

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=15)
model.fit(user_item_csr)

recommended = model.recommend(userid=5, user_items=user_item_csr[5], N=10)
print(recommended)
```

**Deliverable for Phase 6:** The trained implicit-ALS model's top-10 recommendations for 2 sample users with a sanity check against their watch history.

---

## Phase 7 — Bayesian Personalized Ranking (BPR)



### Tasks to Perform

1. Explain the core shift BPR makes: instead of predicting *how much* a user will like an item (a regression-style objective), optimize directly for *correct ranking order* between a positive item and a random negative item.
2. Write the BPR pairwise assumption: for a user `u`, an observed item `i`, and an unobserved item `j`, assume `u` prefers `i` over `j` (`score(u,i) > score(u,j)`).
3. Write the BPR loss conceptually: `-ln(σ(score(u,i) - score(u,j)))`, summed over sampled `(u,i,j)` triples, where `σ` is the sigmoid function.
4. Explain why this pairwise framing is often a better fit for implicit-feedback recommendation than pointwise squared-error loss: the actual production goal is almost always "rank the right items higher," not "predict the exact interaction value."
5. Train a BPR model (`implicit.bpr.BayesianPersonalizedRanking` or `lightfm` with `loss="bpr"`) on the same implicit matrix from Phase 6 and compare its top-N recommendations to the ALS model's.

### 7.1 Formula & Code Reference

```
BPR assumption:  for observed i, unobserved j:  score(u,i) > score(u,j)
BPR loss:        -ln(σ(score(u,i) - score(u,j)))   summed over sampled triples (u,i,j)
```

```python
import implicit

bpr_model = implicit.bpr.BayesianPersonalizedRanking(factors=20, iterations=100)
bpr_model.fit(user_item_csr)

recommended = bpr_model.recommend(userid=5, user_items=user_item_csr[5], N=10)
```

**Deliverable for Phase 7:** The trained BPR model's recommendations compared side-by-side with Phase 6's ALS recommendations, with a written note on where (and whether) they differ.

---

## Phase 8 — Strengths, Weaknesses & When Matrix Factorization Breaks Down



### Tasks to Perform

1. List matrix factorization's strengths: learns embeddings automatically (no feature engineering), generally more accurate than memory-based CF, scales reasonably well with WALS/ALS.
2. List its weaknesses: cold start for brand-new users/items (no interaction history means no learnable embedding), hard to incorporate side features (user demographics, item metadata) into the pure matrix-factorization formulation, and popularity bias — when using dot-product similarity, popular items (which tend to have larger embedding norms) get recommended to everyone.
3. Reproduce the popularity-bias effect: compare the embedding norms of a very popular movie vs. a niche movie in your trained model, and confirm the popular one tends to score higher against almost any user embedding.
4. State the natural next step (previewed for Task 14): deep neural network models can incorporate side features directly and use learned similarity functions instead of a fixed dot product, directly addressing these two weaknesses.

**Deliverable for Phase 8:** The strengths/weaknesses table, the embedding-norm popularity-bias demonstration, and a one-paragraph bridge into why Task 14 moves to deep learning.

---

## Exercises (Must Do)

| # | Exercise | Skills Practiced |
|---|----------|--------------------|
| 1 | Compare user-based CF, item-based CF, and matrix-factorization RMSE on the same held-out test split | Head-to-head model comparison |
| 2 | Vary the embedding dimension `d` (5, 20, 50, 100) in your SVD model and plot RMSE vs. `d` | Hyperparameter sensitivity |
| 3 | Implement the cold-start scenario: add a brand-new user with 2 ratings to the matrix — can your trained MF model produce any recommendation for them without retraining? | Cold-start limitation, hands-on |
| 4 | Compare ALS vs. BPR recommendations for 3 different users and categorize how often they agree on the top-5 | Pointwise vs. pairwise comparison |
| 5 | Implement the adjusted cosine similarity from Phase 3 from scratch (no library) for a 5x5 toy user-item matrix | Manual similarity derivation |

---

## Deliverables Checklist

- [ ] **Phase 1:** Written walkthrough of the 1D/2D embedding intuition
- [ ] **Phase 2:** User-user similarity matrix + hand-worked rating prediction
- [ ] **Phase 3:** Item-item similarity matrix + hand-worked prediction + user-vs-item CF comparison
- [ ] **Phase 4:** SVD-based toy factorization + explanation of observed-only SVD's failure on sparse data
- [ ] **Phase 5:** Trained SVD model RMSE + SGD-vs-WALS comparison table
- [ ] **Phase 6:** Implicit ALS model + top-10 recommendations for 2 users
- [ ] **Phase 7:** BPR model + side-by-side comparison with ALS
- [ ] **Phase 8:** Strengths/weaknesses table + popularity-bias demonstration
- [ ] **Exercises:** All 5 exercises attempted
- [ ] **Code:** Clean notebook that runs top to bottom without errors

---

## Reference: Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Collaborative Filtering** | Recommending using similarities between users and/or items, mined from the feedback matrix alone |
| **User-Based CF** | Predicts a rating from similar *users'* ratings on the same item |
| **Item-Based CF** | Predicts a rating from the same user's ratings on similar *items* |
| **Adjusted Cosine Similarity** | Cosine similarity computed after centering each user's ratings, correcting for different rating scales |
| **Matrix Factorization** | Learns user embedding matrix `U` and item embedding matrix `V` such that `U·Vᵀ ≈ A` |
| **Weighted Matrix Factorization (WMF)** | Matrix factorization whose objective also down-weights a term over unobserved entries, avoiding the trivial all-zero solution |
| **SGD (for MF)** | Generic gradient-based optimizer for the MF objective; flexible, supports side features and custom losses |
| **WALS** | Weighted Alternating Least Squares; alternates closed-form solves for `U` and `V`; faster convergence, specialized to squared loss |
| **Implicit Feedback ALS** | A confidence-weighted matrix factorization variant for binary/implicit interaction data (Hu, Koren & Volinsky) |
| **Bayesian Personalized Ranking (BPR)** | A pairwise ranking loss that directly optimizes for correct relative ordering of a positive vs. a sampled negative item |
| **Popularity Bias** | The tendency of dot-product-similarity models to over-recommend popular items to everyone |

---

## Resources

| Resource | Link |
|----------|------|
| Google Recommendation Systems — Collaborative Filtering | [developers.google.com/machine-learning/recommendation/collaborative/basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics) |
| Google Recommendation Systems — Matrix Factorization | [developers.google.com/machine-learning/recommendation/collaborative/matrix](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) |
| `implicit` library documentation | [benfred.github.io/implicit](https://benfred.github.io/implicit/) |
| `scikit-surprise` documentation | [surpriselib.com](https://surpriselib.com/) |
| Hu, Koren & Volinsky, "Collaborative Filtering for Implicit Feedback Datasets" (2008) | Search via Google Scholar |
| Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (2009) | Search via Google Scholar |
| Task 11 reference document | `Task_11_Recsys_Foundations_and_Content_Based_Filtering.md` |

---

## Suggested Timeline

| Date | Phases |
|------|--------|
| 08/10/2026 | Phase 1–8, Exercises, Demo |

---
**NOTE:** As in Task 11, attend daily standups/wrap-ups and keep the tracking sheet updated.