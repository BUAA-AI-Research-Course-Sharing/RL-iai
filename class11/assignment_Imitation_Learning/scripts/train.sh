python run_behavior_cloning.py  --expert_policy_file ../policies/experts/Humanoid.pkl \
								--env_name Humanoid-v2 \
								--exp_name test_dagger_Humanoid \
								--n_iter 10 \
								--do_dagger \
								--expert_data ../expert_data/expert_data_Humanoid-v2.pkl \
								--video_log_freq 1