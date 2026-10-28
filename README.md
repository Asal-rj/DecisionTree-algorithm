# Decision Tree — Drug Classification

A decision tree classifier that predicts which drug to prescribe for a patient,
based on the patient's features.

> Based on the drug-classification lab from IBM's *Machine Learning with Python*
> course on Coursera. I re-implemented it myself as practice, and wrote this up
> in my own words so I can explain every step of the pipeline.

## The dataset

- 200 patients
- Features: Age, Sex, BP (blood pressure), Cholesterol, Na_to_K
  (sodium-to-potassium ratio in the blood)
- Target: Drug (drugA, drugB, drugC, drugX, drugY)

## What I did

1. **Encoded the categorical columns** (Sex, BP, Cholesterol) as numbers so
   sklearn could work with them. The target column, Drug, was kept aside.
2. **Mapped the drugs to numbers for correlation analysis:**
   drugA = 0, drugB = 1, drugC = 2, drugX = 3, drugY = 4.
3. **Checked correlations.** Na_to_K has the strongest correlation with the
   drug (around 0.5): the higher the number assigned to the drug (toward
   drugY = 4), the higher the Na_to_K ratio tends to be. This makes sense
   later, once the tree is built — if a patient's Na_to_K is greater than
   14.027, the prescribed drug is drugY.
4. **Plotted the count of each drug** in the dataset.
5. **Ran the standard ML pipeline:** preprocess the data, split it into
   train and test sets, create the decision tree, fit it on the training
   set, predict on the unseen test data, evaluate the predictions, and plot
   the tree.

## The model

- Decision tree classifier with `max_depth = 4`, meaning at most 4 questions
  are asked along any path from the root of the tree to a leaf.
- The tree keeps splitting nodes until the leaves are pure (entropy = 0),
  with each leaf belonging to a single drug class.

## Decision rules

Reading the tree from the root down, a patient is classified as:

- **DrugA:** Na_to_K ≤ 14.027, BP = High, Age ≤ 50.5
- **DrugB:** Na_to_K ≤ 14.027, BP = High, Age > 50.5
- **DrugC:** Na_to_K ≤ 14.027, BP = Low, Cholesterol = High
- **DrugX:** Na_to_K ≤ 14.027, and either (BP = Low or Normal with
  Cholesterol = Normal) or (BP = Normal with Cholesterol = High)
- **DrugY:** Na_to_K > 14.027

Every combination of BP and Cholesterol below the Na_to_K threshold lands on
exactly one drug — nothing overlaps, nothing is missing.

## Results

- **Accuracy on the test set: 98.3%** — very strong performance. 100% would mean perfect prediction on X_test which then I would check for overfitting or data leakage.
