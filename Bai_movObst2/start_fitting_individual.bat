FOR /L %%A IN (0, 1, 6) DO (
  start cmd /k^
 "call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Bai_movObst1 --approach_experiment Bai_movObst1b --subject %%A --t_start obst_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach --avoid_model cohen_avoid --training_model cohen_avoid"
)

pause