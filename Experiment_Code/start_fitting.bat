FOR /L %%A IN (0,1,1) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Bai_movObst1 --subject %%A --t_start stimuli_onset --t_end obst_out --ps subj --method dual_annealing --approach_model fajen_approach --avoid_model cohen_avoid --training_model cohen_avoid"
)

pause