"""HADL course project - pipeline integration, evaluation metrics and plotting.

Owner: Rahul Yadav (Person 6, Evaluation & Reporting).

The package glues together the work of the other team members:

* data preparation          -> data_and_scripts/odia_real_cleaned, odia_synth_cleaned
* low-rank selection         -> low_rank_method/odia_low_rank_run/final_weighted_train_draws.parquet
* covariance matching        -> covariance_matching/outputs/covariance_matched_n*.parquet
* time-proxy profiling       -> performance_profiling/time_proxy_model.json
* Qwen3-1.7B + LoRA training -> re-implemented in hadl_eval.training (same hyper-parameters)

and adds what was missing: real scores instead of placeholder scores, a 0/1 knapsack
solver, batched evaluation, QA metrics with confidence intervals, proxy-error
analysis, result aggregation and plots.
"""

__version__ = "1.0.0"
