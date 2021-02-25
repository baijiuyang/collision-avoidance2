set subjs[0]=2 3 4 5 6 7 8 9 10 11 12 13 14 15
set subjs[1]=1 3 4 5 6 7 8 9 10 11 12 13 14 15
set subjs[2]=1 2 4 5 6 7 8 9 10 11 12 13 14 15
set subjs[3]=1 2 3 5 6 7 8 9 10 11 12 13 14 15
set subjs[4]=1 2 3 4 6 7 8 9 10 11 12 13 14 15
set subjs[5]=1 2 3 4 5 7 8 9 10 11 12 13 14 15
set subjs[6]=1 2 3 4 5 6 8 9 10 11 12 13 14 15
set subjs[7]=1 2 3 4 5 6 7 9 10 11 12 13 14 15
set subjs[8]=1 2 3 4 5 6 7 8 10 11 12 13 14 15
set subjs[9]=1 2 3 4 5 6 7 8 9 11 12 13 14 15
set subjs[10]=1 2 3 4 5 6 7 8 9 10 12 13 14 15
set subjs[11]=1 2 3 4 5 6 7 8 9 10 11 13 14 15
set subjs[12]=1 2 3 4 5 6 7 8 9 10 11 12 14 15
set subjs[13]=1 2 3 4 5 6 7 8 9 10 11 12 13 15
set subjs[14]=1 2 3 4 5 6 7 8 9 10 11 12 13 14

FOR /L %%A IN (0, 1, 14) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Cohen_movObst2 --approach_experiment Fajen_steer1a --subject %%subjs[%%A]%% --t_start stimuli_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach2 --avoid_model cohen_avoid4 --training_model cohen_avoid4"
)

pause