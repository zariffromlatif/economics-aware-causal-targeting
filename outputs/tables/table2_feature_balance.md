### Table 2: Pre-Treatment Covariate Balance (Standardized Mean Differences)

|    | Feature            |   Mean Ctrl |   Mean Mens |   SMD (Mens vs Ctrl) |   p-val (Mens) |   Mean Womens |   SMD (Womens vs Ctrl) |   p-val (Womens) |   Max SMD |
|----|--------------------|-------------|-------------|----------------------|----------------|---------------|------------------------|------------------|-----------|
|  0 | recency            |      5.7497 |      5.7736 |               0.0068 |         0.4807 |        5.7678 |                 0.0052 |           0.5925 |    0.0068 |
|  1 | history            |    240.8827 |    242.8359 |               0.0076 |         0.4320 |      242.5366 |                 0.0065 |           0.5012 |    0.0076 |
|  2 | mens               |      0.5532 |      0.5509 |               0.0046 |         0.6362 |        0.5489 |                 0.0086 |           0.3726 |    0.0086 |
|  3 | womens             |      0.5476 |      0.5514 |               0.0076 |         0.4335 |        0.5501 |                 0.0049 |           0.6093 |    0.0076 |
|  4 | newbie             |      0.5020 |      0.5015 |               0.0009 |         0.9267 |        0.5032 |                 0.0026 |           0.7917 |    0.0034 |
|  5 | zip_code_Surburban |      0.4518 |      0.4459 |               0.0117 |         0.2255 |        0.4512 |                 0.0011 |           0.9104 |    0.0117 |
|  6 | zip_code_Urban     |      0.4009 |      0.4019 |               0.0020 |         0.8387 |        0.4001 |                 0.0018 |           0.8555 |    0.0037 |
|  7 | channel_Phone      |      0.4378 |      0.4337 |               0.0083 |         0.3930 |        0.4420 |                 0.0086 |           0.3730 |    0.0169 |
|  8 | channel_Web        |      0.4399 |      0.4454 |               0.0110 |         0.2556 |        0.4374 |                 0.0051 |           0.5948 |    0.0162 |

**Omnibus Randomization Test**: 5-Fold Cross-Validated ROC-AUC = 0.4991 (Expected ~0.5000 under perfect randomization; confirms zero confounding).
**Max SMD across all features**: 0.0169 (Well below the 0.05 conservative imbalance threshold).
