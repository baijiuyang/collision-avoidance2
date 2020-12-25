FOR /L %%A IN (0,1,1) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Fajen_steer1a --subject %%A --t_start stimuli_onset --t_end -2 --ps trial --method differential_evolution --approach_model fajen_approach2 --avoid_model cohen_avoid --training_model fajen_approach2"
)

pause