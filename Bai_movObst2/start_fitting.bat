Rem Negative subject number means excluding this subject from training data
Rem for leave one out cross validataion
set subjs=-8 -9 -10 -11 -12 -13 -14 -15



FOR %%A IN (%subjs%) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Bai_movObst1b --approach_experiment Bai_movObst1b --subject %%A --t_start stimuli_onset --t_end stimuli_out --ps trial --method differential_evolution --approach_model fajen_approach --avoid_model na --training_model fajen_approach"
)

pause