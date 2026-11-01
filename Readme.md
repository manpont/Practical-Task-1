# Practical 1 – Industry Classification from Business Descriptions

Foundations of Data Science, UZH

## What this is
In this practical we try to predict a company's industry (25 classes) from a text
embedding of its business description. The dataset has about 31,700 companies,
each described by a 768-dimensional embedding.

## What we did
1. **Explored the data**: converted the embeddings from text to numbers, checked for
   missing values and duplicates, and looked at the class distribution
   (very imbalanced: Materials is ~26 % of all companies). I also found ~4,500
   companies sharing one identical embedding, almost all Materials, probably a
   placeholder description. I kept them in the data but flagged them, since the model classifies them correctly by recognising the repeated vector rather than by learning from a real description.
   **Visualised it with PCA**: projected the embeddings to 2D to check whether industries form separate groups. Several do (e.g. Banks, Pharma, Real Estate),
   so there is clear signal.
3. **Compared two classifiers**: logistic regression and k-NN. I tuned C and k on a
   separate split of the training data and evaluated both on the validation set.
4. **Analysed the errors**: per-industry scores, confusion matrices, and whether
   both models make the same mistakes.

## Results (validation set)
| Model | Accuracy | F1 weighted | F1 macro |
|---|---|---|---|
| Baseline (always "Materials") | 0.25 | 0.10 | 0.02 |
| Logistic regression (C = 10) | 0.78 | 0.77 | 0.71 |
| k-NN (k = 10) | 0.77 | 0.77 | 0.70 |

- The 78 % includes the template companies, which the model always gets right. If I
  leave them out and only look at companies with a real description, accuracy drops
  to about 74 %, which I think is the more honest number.

- Both models are about 3× better than the baseline.
- The difference between them is not statistically significant (McNemar's test).
- Errors depend more on how similar industries are (e.g. REITs vs. Real Estate,
  Semiconductors vs. Tech Hardware) than on how many examples an industry has.


