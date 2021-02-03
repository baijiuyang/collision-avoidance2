FOR /L %%A IN (0,1,10) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Cohen_movObst1 --approach_experiment Fajen_steer1a --subject %%A --t_start stimuli_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach2 --avoid_model cohen_avoid4_thres --training_model cohen_avoid4_thres"
)

pause