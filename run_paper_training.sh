#!/usr/bin/bash


########################## DEVELOPMENT

python RNN_experiment.py --behaviour crawl --env box_messy > logs/d1.txt 2>&1;
python RNN_experiment.py --behaviour walk --pretrained_behav crawl --env box_messy > logs/d2.txt 2>&1;
python RNN_experiment.py --behaviour run --pretrained_behav crawl,walk --env box_messy > logs/d3.txt 2>&1;
python RNN_experiment.py --behaviour adult --pretrained_behav crawl,walk,run --env box_messy > logs/d4.txt 2>&1;
## with grid cells
python RNN_experiment.py --behaviour adult --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/d9_gc.txt 2>&1;


##########################################
########## PARAMETER SWEEP ###############
##########################################


########################## TRAIN LONGER

python RNN_experiment.py --name_prefix "train_longer" --epochs 2500 --behaviour crawl --env box_messy > logs/tl1.txt 2>&1;
python RNN_experiment.py --name_prefix "train_longer" --epochs 2500 --behaviour walk --pretrained_behav crawl --env box_messy > logs/tl2.txt 2>&1;
python RNN_experiment.py --name_prefix "train_longer" --epochs 2500 --behaviour run --pretrained_behav crawl,walk --env box_messy > logs/tl3.txt 2>&1;
python RNN_experiment.py --name_prefix "train_longer" --epochs 2500 --behaviour adult --pretrained_behav crawl,walk,run --env box_messy > logs/tl4.txt 2>&1;
## with grid cells
python RNN_experiment.py --name_prefix "train_longer" --epochs 2500 --behaviour adult --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/tl9_gc.txt 2>&1;


########################## SMALL INPUT

python RNN_experiment.py --name_prefix "small_input" --behaviour crawl --env box_messy > logs/si1.txt 2>&1;
python RNN_experiment.py --name_prefix "small_input" --behaviour walk --pretrained_behav crawl --env box_messy > logs/si2.txt 2>&1;
python RNN_experiment.py --name_prefix "small_input" --behaviour run --pretrained_behav crawl,walk --env box_messy > logs/si3.txt 2>&1;
python RNN_experiment.py --name_prefix "small_input" --behaviour adult --pretrained_behav crawl,walk,run --env box_messy > logs/si4.txt 2>&1;
## with grid cells
python RNN_experiment.py --name_prefix "small_input" --behaviour adult --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/si9_gc.txt 2>&1;


########################## BIG INPUT

python RNN_experiment.py --name_prefix "big_input" --behaviour crawl --env box_messy > logs/bi1.txt 2>&1;
python RNN_experiment.py --name_prefix "big_input" --behaviour walk --pretrained_behav crawl --env box_messy > logs/bi2.txt 2>&1;
python RNN_experiment.py --name_prefix "big_input" --behaviour run --pretrained_behav crawl,walk --env box_messy > logs/bi3.txt 2>&1;
python RNN_experiment.py --name_prefix "big_input" --behaviour adult --pretrained_behav crawl,walk,run --env box_messy > logs/bi4.txt 2>&1;
## with grid cells
python RNN_experiment.py --name_prefix "big_input" --behaviour adult --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/bi9_gc.txt 2>&1;


########################## MORE HIDDEN UNITS

python RNN_experiment.py --latent_dim 625 --behaviour crawl --env box_messy > logs/mh1.txt 2>&1;
python RNN_experiment.py --latent_dim 625 --behaviour walk --pretrained_behav crawl --env box_messy > logs/mh2.txt 2>&1;
python RNN_experiment.py --latent_dim 625 --behaviour run --pretrained_behav crawl,walk --env box_messy > logs/mh3.txt 2>&1;
python RNN_experiment.py --latent_dim 625 --behaviour adult --pretrained_behav crawl,walk,run --env box_messy > logs/mh4.txt 2>&1;
## with grid cells
python RNN_experiment.py --latent_dim 625 --behaviour adult --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/mh9_gc.txt 2>&1;


########################## LESS HIDDEN UNITS

python RNN_experiment.py --latent_dim 375 --behaviour crawl --env box_messy > logs/lh1.txt 2>&1;
python RNN_experiment.py --latent_dim 375 --epochs 2000 --behaviour walk --pretrained_behav crawl --env box_messy > logs/lh2.txt 2>&1;
python RNN_experiment.py --latent_dim 375 --epochs 2000 --behaviour run --pretrained_behav crawl,walk --env box_messy > logs/lh3.txt 2>&1;
python RNN_experiment.py --latent_dim 375 --epochs 2000 --behaviour adult --pretrained_behav crawl,walk,run --env box_messy > logs/lh4.txt 2>&1;
## with grid cells
python RNN_experiment.py --latent_dim 375 --epochs 2000 --behaviour adult --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/lh9_gc.txt 2>&1;


#####################################################################
########## ALTERNATIVE HYPOTHESES CONTROL EXPERIMENTS ###############
#####################################################################


########################## RATE OF CHANGE

python RNN_experiment.py --behaviour crawl --env box_messy --stride 20 --pretrained_model_folder "/media/data/vrtopc/box/crawl/predictions/box_messy/vanilla/RNN_f1_w9_st10_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01" > logs/roc1.txt 2>&1;
python RNN_experiment.py --behaviour crawl --env box_messy --stride 25 --pretrained_model_folder "/media/data/vrtopc/box/crawl/predictions/box_messy/vanilla/RNN_ft_f1_w9_st20_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01" > logs/roc2.txt 2>&1;
python RNN_experiment.py --behaviour crawl --env box_messy --stride 30 --pretrained_model_folder "/media/data/vrtopc/box/crawl/predictions/box_messy/vanilla/RNN_ft_f1_w9_st25_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01" > logs/roc3.txt 2>&1;
## with grid cells
python RNN_experiment.py --behaviour crawl --env box_messy --stride 30 --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/roc9_gc.txt 2>&1;


########################## REVERSE DEVELOPMENT

python RNN_experiment.py --epochs 2500 --behaviour adult --env box_messy > logs/rd1.txt 2>&1;
python RNN_experiment.py --epochs 2500 --behaviour run --pretrained_behav adult --env box_messy > logs/rd2.txt 2>&1;
python RNN_experiment.py --epochs 2500 --behaviour walk --pretrained_behav adult,run --env box_messy > logs/rd3.txt 2>&1;
python RNN_experiment.py --epochs 2500 --behaviour crawl --pretrained_behav adult,run,walk --env box_messy > logs/rd4.txt 2>&1;
## with grid cells
python RNN_experiment.py --epochs 2500 --behaviour crawl --env box_messy --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/rt9_gc.txt 2>&1;


########################## CRAWL WITH MORE DATA

python RNN_experiment.py --behaviour crawl --env box_messy --moredata 1 --pretrained_model_folder "/media/data/vrtopc/box/crawl/predictions/box_messy/vanilla/RNN_f1_w9_st10_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01" > logs/m1.txt 2>&1;
python RNN_experiment.py --behaviour crawl --env box_messy --moredata 2 --pretrained_model_folder "/media/data/vrtopc/box/crawl/predictions/box_messy/vanilla/RNN_ft_moredata1_f1_w9_st20_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01" > logs/m2.txt 2>&1;
python RNN_experiment.py --behaviour crawl --env box_messy --moredata 3 --pretrained_model_folder "/media/data/vrtopc/box/crawl/predictions/box_messy/vanilla/RNN_ft_moredata2_f1_w9_st20_fss4_do[0,0,0]_lat500_nlsigmoid_hreg0.0_wreg0.0_s01" > logs/m3.txt 2>&1;
## with grid cells
python RNN_experiment.py --behaviour crawl --env box_messy --moredata 3 --reset_hidden_at 10 --n_gridcells 25 --gridcells_softmax --gridcells_modules 0.2,0.4,0.6 --gridcells_orientation 0.1 --lr 1e-4 > logs/m9_gc.txt 2>&1;
