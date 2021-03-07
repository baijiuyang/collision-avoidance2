set subjs=-0 -1



FOR %%A IN (%subjs%) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Bai_movObst1 --approach_experiment Fajen_steer1a --subject %%A --t_start obst_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach --avoid_model cohen_avoid --training_model cohen_avoid"
)

pause