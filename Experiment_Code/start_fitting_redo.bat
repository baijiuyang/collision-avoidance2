set subjs[0]=0 1 2 3 4 6 7 8 9 10 11 12 13 14 15



FOR /L %%A IN (0, 1, 0) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Cohen_movObst1 --approach_experiment Fajen_steer1a --subject %%subjs[%%A]%% --t_start obst_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach --avoid_model cohen_avoid --training_model cohen_avoid"
)

pause