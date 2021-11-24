Rem Negative subject number means excluding this subject from training data
Rem for leave one out cross validataion
set subjs=-7 -9 -10 -11 -12 -13



FOR %%A IN (%subjs%) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Bai_movObst1 --approach_experiment Bai_movObst1b --subject %%A --t_start threshold_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach2 --avoid_model cohen_avoid3 --training_model cohen_avoid3"
)

pause