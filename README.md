# Collaborative Filtering & Matrix Factorization

Task 12 of the Recommendation Systems track, on the [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/) dataset
(610 users · 9,724 rated movies · 100,836 ratings · 98.3% of the user × movie matrix empty).

[Task 11](https://github.com/Balaji-krish09/Recommendation-Systems) recommended movies from their features. This task uses **no item features at all**: every model learns only from the users × movies rating matrix.

**Notebook:** [`Task12_CollaborativeFiltering_MatrixFactorization.ipynb`](Task12_CollaborativeFiltering_MatrixFactorization.ipynb)
**Slides:** [`Task12_CollaborativeFiltering_Presentation.pptx`](Task12_CollaborativeFiltering_Presentation.pptx)
**Task spec:** [`task12-readme.md`](task12-readme.md)

| Phase | What was done | Key result |
|---|---|---|
| 1. Embedding intuition | Hand-made 1-D → 2-D user/movie embeddings, score = dot product | The Dark Knight, users C / D: 0.63 / 0.56 in 1-D → 1.35 / −0.25 in 2-D |
| 2. User-based CF | Pearson similarity on a 150 × 300 core (verified against pandas), mean-centred prediction | Held-out 3.5★ rating predicted as 3.91 |
| 3. Item-based CF | Adjusted cosine item-item matrix, same held-out rating | 3.96; The Matrix's nearest neighbour is Office Space (shared audience, not genre) |
| 4. MF fundamentals | SVD on a dense toy; zero-/mean-filled SVD vs observed-only factorization on hidden cells | Hidden-cell RMSE: zero-filled SVD 2.74–3.34, observed-only MF (λ = 0.5) 0.49 |
| 5. WMF: SGD vs WALS | Both optimizers written from scratch on one objective, w0 study, `surprise.SVD` | SVD test RMSE 0.8780 vs 0.8785 for biases alone |
| 6. Implicit ALS | Confidence matrix 1 + α·rating, α tuned on held-out interactions | precision@10 0.275 (α = 1) vs 0.157 for popularity |
| 7. BPR | Pairwise ranking loss, compared with ALS on the same split | precision@10 0.186: ALS ranks better on this dataset |
| 8. Limits | Embedding norm vs popularity, dot-product vs cosine scoring | Spearman 0.944; 58.8% of ALS recommendation slots go to the top-1% movies |

Plus all 5 exercises:

| # | Exercise | Result |
|---|---|---|
| 1 | User-CF vs item-CF vs MF on one split | RMSE 0.838 / 0.852 / 0.836 (bias-only baseline 0.841) |
| 2 | RMSE vs embedding size d = 5, 20, 50, 100 | Test flat (0.877–0.881) while train falls 0.813 → 0.635: overfitting |
| 3 | Cold start: new user with 2 ratings | surprise SVD returns the same μ + b_i list for everyone; ALS folds the user in without retraining; BPR can't |
| 4 | ALS vs BPR top-5 agreement | 1.42 of 5 titles shared on average across 610 users |
| 5 | Adjusted cosine from scratch (5 × 5 toy) | Matches the vectorised version; M1–M2: plain cosine 0.956 → adjusted −0.988 |

### How to run
1. Install: `pip install numpy pandas scipy matplotlib implicit scikit-surprise jupyter`
2. The `ml-latest-small/` folder (`ratings.csv`, `movies.csv` and the MovieLens license `README.txt`) sits next to the notebook.
3. Open the notebook and run all cells top to bottom (about 1–2 minutes). Seeds are fixed and BPR runs single-threaded, so the numbers reproduce.

### Fixes made to the task's reference code
- Phase 6 indexed the sparse matrix by raw IDs, building a 611 × 193,610 matrix with 183,886 empty columns. IDs are now mapped to contiguous rows/columns (610 × 9,724).
- Phase 6 passed raw star ratings as ALS confidence. Confidence is now built explicitly as `1 + α·rating` (a `rating_count` is always 1 in MovieLens).
- `--break-system-packages` is a Linux-only pip flag; on Windows plain `pip install` works.
- Added a fixed `random_state` everywhere; BPR runs with `num_threads=1` because multi-threaded (Hogwild) BPR gives different results on every run.

## Data license
MovieLens data © GroupLens Research, University of Minnesota. It is redistributed here under the same license conditions
(see `ml-latest-small/README.txt`), for non-commercial use only.
