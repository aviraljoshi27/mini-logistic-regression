# Data

`breast_cancer.csv` is the Breast Cancer Wisconsin (Diagnostic) dataset.
I saved it once from scikit-learn's `load_breast_cancer(as_frame=True)`,
so every notebook reads the same file.

- 569 rows, one per tissue sample
- 30 numeric feature columns (measurements of cell nuclei from microscope images)
- 1 `target` column

## Target encoding

I flipped the target compared to scikit-learn:

- `1` = malignant (cancer), 212 rows
- `0` = benign, 357 rows

scikit-learn uses 0 = malignant. I flipped it because the model predicts
the probability of class 1, and the question I care about is
"how likely is this to be cancer?"

## Source

Wolberg, Mangasarian, Street and Street (1993), Breast Cancer Wisconsin
(Diagnostic), UCI Machine Learning Repository. License: CC BY 4.0.
