Relative covariance distance to the real train data (lower = closer). PCA space k=64; full = 768-dim.

| kind | run_id | n | rel_dist_pca | rel_dist_full |
| --- | --- | --- | --- | --- |
| subset | cov_match_only_b490_s42 | 1657 | 0.348 | 0.663 |
| subset | low_rank_only_b490_s42 | 1762 | 0.789 | 1.312 |
| subset | proposed_knapsack_b490_s42 | 2715 | 0.459 | 0.759 |
| subset | quality_only_b490_s42 | 1472 | 0.429 | 0.719 |
| subset | random_b490_s42 | 1850 | 0.684 | 0.988 |
| reference | Whole low-rank pool | 33104 | 0.670 | 0.948 |
| reference | All 7,088 covariance-matched candidates | 7088 | 0.450 | 0.692 |
| reference | Covariance-matching top 1,772 | 1772 | 0.351 | 0.662 |
| reference | Covariance-matching top 3,544 | 3544 | 0.395 | 0.664 |
