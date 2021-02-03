set subjs[0]=2 3 4 5 6 7 8 10 11 12 13 14 15
set subjs[1]=1 3 4 5 6 7 8 10 11 12 13 14 15
set subjs[2]=1 2 4 5 6 7 8 10 11 12 13 14 15
set subjs[3]=1 2 3 5 6 7 8 10 11 12 13 14 15
set subjs[4]=1 2 3 4 6 7 8 10 11 12 13 14 15
set subjs[5]=1 2 3 4 5 7 8 10 11 12 13 14 15
set subjs[6]=1 2 3 4 5 6 8 10 11 12 13 14 15
set subjs[7]=1 2 3 4 5 6 7 10 11 12 13 14 15
set subjs[8]=1 2 3 4 5 6 7 8 11 12 13 14 15
set subjs[9]=1 2 3 4 5 6 7 8 10 12 13 14 15
set subjs[10]=1 2 3 4 5 6 7 8 10 11 13 14 15
set subjs[11]=1 2 3 4 5 6 7 8 10 11 12 14 15
set subjs[12]=1 2 3 4 5 6 7 8 10 11 12 13 15
set subjs[13]=1 2 3 4 5 6 7 8 10 11 12 13 14

FOR /L %%A IN (0, 1, 13) DO (
  start cmd /k^
 "call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3"^
 "& activate avoidance"^
 "& python fitting.py --experiment_name Cohen_movObst1 --approach_experiment Fajen_steer1a --subject %%subjs[%%A]%% --t_start stimuli_onset --t_end obst_out --ps trial --method differential_evolution --approach_model fajen_approach2 --avoid_model cohen_avoid4_thres --training_model cohen_avoid4_thres"
)

pause