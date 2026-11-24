| split | method | budget_sec | exact_match_mean | token_f1_mean | char_f1_mean | contains_gold_mean | actual_wall_clock_sec_mean | proxy_error_pct_mean | exact_match_std | token_f1_std | char_f1_std | contains_gold_std | actual_wall_clock_sec_std | proxy_error_pct_std | n_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | base_zero_shot |  | 8.52 | 30.76 | 47.17 | 38.07 |  |  |  |  |  |  |  |  | 1 |
| test | cov_match_only | 527.00 | 27.84 | 51.77 | 65.34 | 47.73 | 4320.93 | -75.62 |  |  |  |  |  |  | 1 |
| test | low_rank_only | 527.00 | 27.84 | 50.95 | 62.95 | 47.73 | 4778.23 | -77.95 |  |  |  |  |  |  | 1 |
| test | proposed_knapsack | 527.00 | 30.68 | 52.25 | 63.90 | 48.86 | 4409.68 | -76.10 |  |  |  |  |  |  | 1 |
| test | quality_only | 527.00 | 28.41 | 52.00 | 64.08 | 44.89 | 4757.44 | -77.85 |  |  |  |  |  |  | 1 |
| test | random | 527.00 | 27.27 | 52.77 | 64.97 | 45.45 | 4303.57 | -75.52 |  |  |  |  |  |  | 1 |
| test | real_only |  | 24.43 | 45.88 | 60.85 | 43.75 | 2169.89 | -75.72 |  |  |  |  |  |  | 1 |
| validation | base_zero_shot |  | 16.92 | 35.69 | 50.59 | 45.13 |  |  |  |  |  |  |  |  | 1 |
| validation | cov_match_only | 527.00 | 32.82 | 54.29 | 68.42 | 46.67 | 4320.93 | -75.62 |  |  |  |  |  |  | 1 |
| validation | low_rank_only | 527.00 | 31.28 | 52.73 | 66.40 | 45.13 | 4778.23 | -77.95 |  |  |  |  |  |  | 1 |
| validation | proposed_knapsack | 527.00 | 26.67 | 50.79 | 67.94 | 42.56 | 4409.68 | -76.10 |  |  |  |  |  |  | 1 |
| validation | quality_only | 527.00 | 29.23 | 49.75 | 66.32 | 42.56 | 4757.44 | -77.85 |  |  |  |  |  |  | 1 |
| validation | random | 527.00 | 29.23 | 51.10 | 68.97 | 42.56 | 4303.57 | -75.52 |  |  |  |  |  |  | 1 |
| validation | real_only |  | 28.72 | 49.27 | 65.96 | 42.56 | 2169.89 | -75.72 |  |  |  |  |  |  | 1 |
