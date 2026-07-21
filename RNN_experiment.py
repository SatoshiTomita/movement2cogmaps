import platform
import os
if 'mac' in platform.platform():
    BASE_DIR = "/Users/USER/vrtopc/"
    DATA_DIR = "/media/data/vrtopc"
else:
    BASE_DIR = "/home/USER/vr_to_pc/"
    DATA_DIR = "/media/data/vrtopc"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# データ保存先を、BASE_DIRの中にある "data" フォルダに設定
DATA_DIR = os.path.join(BASE_DIR, "data")
import sys
sys.path.append(BASE_DIR)

import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import time
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.trainer import RNNTrainer
from utils.activiter import RNNActiviter


def activity_part(trainer, exp_dir, dataloader_act, bptt_trainer):
    print()
    print(
        '''
        ######################################################################
        ########################### ACTIVITY PART ############################
        ######################################################################
        '''
    )

    model_name = trainer.get_model_name()
    activiter = RNNActiviter(trainer.get_args(), DATA_DIR, DEVICE, model_name, exp_dir)

    rnn, epoch, exp_dir_act = activiter.load_model()

    print(f"\n[+] Extracting latent activity from the model", flush=True)
    latent_activity, positions, thetas, vloss_dict = activiter.extract_latent_activity(
        rnn, dataloader_act, bptt_trainer
    )

    print("\n[*] Split the data to calculate cells stability later")
    (
        latent_activity_half1, positions_half1, thetas_half1,
        latent_activity_half2, positions_half2, thetas_half2
    ) = activiter.split_data(latent_activity, positions, thetas)

    print("\n[+] Calculating sRSA", end='', flush=True)
    s = time.time()
    sRSA = activiter.calculate_sRSA(latent_activity, positions)
    print(f" (time elapsed: {(time.time() - s)/60:.1f} minutes)", flush=True)

    # we now reshape everything because we don't care about seeds/experiments anymore
    latent_activity = latent_activity.reshape(-1, latent_activity.shape[-1])
    positions = positions.reshape(-1, positions.shape[-1])
    thetas = thetas.reshape(-1, thetas.shape[-1]).squeeze()
    print(f"\n[*] Reshaped data: {latent_activity.shape}, {positions.shape}, {thetas.shape}")

    print("\n[+] Plotting trajectory heatmap")

    print("\n[+] Extracting RNN place activity from hidden units", flush=True)
    rate_maps, si_r, indices_place_cells, n_fields, rm_stability, rm_single_field_dim, rm_vs_hd, rm_vs_hd_stability = activiter.rnn_place_activity(
        latent_activity, latent_activity_half1, latent_activity_half2,
        positions, positions_half1, positions_half2,
        thetas
    )

    print("\n[+] Extracting RNN head direction activity from hidden units", flush=True)
    polar_maps, si_d, rvl, rvangle, indices_hd_cells, pm_stability, pm_vs_place, pm_vs_place_stability = activiter.rnn_hd_activity(
        latent_activity, latent_activity_half1, latent_activity_half2,
        thetas, thetas_half1, thetas_half2,
        positions
    )

    print("\n[+] Calculating position and HD decoding errors", flush=True)
    pos_dec_err, thet_dec_err = activiter.pos_hd_decoding(
        latent_activity_half1, latent_activity_half2,
        positions_half1, positions_half2, thetas_half1, thetas_half2
    )

    print("\n[+] Analyzing selected units", flush=True)
    indices_place_cells, indices_hd_cells, indices_conjunctive_cells = activiter.selected_units_analysis(
        indices_place_cells, indices_hd_cells,
        rate_maps, rm_vs_hd, rm_vs_hd_stability,
        polar_maps, pm_vs_place, pm_vs_place_stability
    )
    
    print("\n[+] Plotting activity results")
    print("\tPlotting place-related activity")
    activiter.save_place_plots(
        rate_maps, si_r, indices_place_cells, indices_hd_cells, indices_conjunctive_cells)
    print("\tPlotting HD-related activity")
    activiter.save_hd_plots(
        polar_maps, si_d, rvl, rvangle, indices_place_cells, indices_hd_cells, indices_conjunctive_cells)

    print("\n[+] Summary\n")
    activiter.save_summary(
        vloss_dict, pos_dec_err, thet_dec_err, sRSA,
        indices_place_cells, n_fields, rm_stability, rm_single_field_dim, rm_vs_hd_stability,
        indices_hd_cells, pm_stability, pm_vs_place_stability,
        indices_conjunctive_cells
    )

    print("\n[*] Finished\n", flush=True)


def main(args):
    if not args.activity_only:
        print(
            '''
            ######################################################################
            ########################### TRAINING PART ############################
            ######################################################################
            '''
        )

    trainer = RNNTrainer(args, DATA_DIR, DEVICE)
    # 動画、角度、位置、速度、回転速度のシミュレーションデータを読み込む
    videos, thetas, positions, velocities, rot_velocities =\
        trainer.load_simulations()

    video_shape = trainer.check_shapes(videos)
    velocity_shape = trainer.check_shapes(velocities)
    rot_velocity_shape = trainer.check_shapes(rot_velocities)

    # 動画、速度、回転速度の特徴量の次元数を取得
    video_n_features = video_shape[-1]
    velocity_n_features = velocity_shape[-1]
    rot_velocity_n_features = rot_velocity_shape[-1]
    print("\n[*] Features shapes:")
    print(
        f"\tVideo: {video_n_features}\n\t"+
        f"Velocity: {velocity_n_features}\n\tRot velocity: {rot_velocity_n_features}"
    )

    print("\n[*] Preprocessing data")
    videos, velocities, rot_velocities = trainer.preprocess_data(
        videos, velocities, rot_velocities
    )

    (
        video_train, velocity_train, rot_velocity_train, positions_train, thetas_train,
        video_test, velocity_test, rot_velocity_test, positions_test, thetas_test,
        video_act, velocity_act, rot_velocity_act, positions_act, thetas_act
    ) = trainer.combine_videos(videos, velocities, rot_velocities, positions, thetas)

    print(f"\n[*] Defining dataloaders with {args.bptt_steps} BPTT steps and {args.n_future_pred} future predictions")
    if not args.activity_only:
        dataloader_train = trainer.generate_dataloader(
            video_train, velocity_train, rot_velocity_train, positions_train, thetas_train, verbose=True
        )
        dataloader_test = trainer.generate_dataloader(
            video_test, velocity_test, rot_velocity_test, positions_test, thetas_test
        )
    dataloader_act = trainer.generate_dataloader(
        video_act, velocity_act, rot_velocity_act, positions_act, thetas_act
    )

    scene_dim = video_n_features
    vel_dim = velocity_n_features + rot_velocity_n_features
    output_dim = video_n_features
    rnn, loss_fn, optimizer, lr_sched = trainer.define_training_objects(
        scene_dim, vel_dim, output_dim
    )

    exp_dir = trainer.get_exp_dir()

    bptt_trainer = trainer.define_bptt_trainer(optimizer, loss_fn)

    if not args.activity_only:
        print("\n[+] Saving untrained model and configuration file")
        torch.save(rnn, os.path.join(exp_dir, f"rnn_epoch0.pth"))

        config_wandb = {**{
            "architecture": args.architecture.upper(),
            "scene_dim": scene_dim,
            "vel_dim": vel_dim,
            "output_dim": output_dim,
            "loss": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": args.lr,
        }, **vars(args)}
        with open(os.path.join(exp_dir, "config.yaml"), 'w') as f:
            yaml.dump(config_wandb, f, default_flow_style=False)

        if args.wandb:
            print("\n[*] Starting Wandb project")
            wandb.init(
                project=f"iclr_{args.env}",
                name=trainer.get_model_name(),
                group=f"{args.architecture.upper()}_{args.name_prefix}",
                config=config_wandb
            )

        print("\n[*] Training model...")
        # 学習実行
        rnn = trainer.train(rnn, bptt_trainer, dataloader_train, dataloader_test, lr_sched)

        figs = bptt_trainer.plot_test_examples(
            rnn, dataloader_test, n_figures=5, n_examples=6*5,
            frame_dim = [args.frame_dim[0]//args.frame_subsampling, args.frame_dim[1]//args.frame_subsampling],
            truncate_scene = video_n_features,
        )
        for i, f in enumerate(figs) : f.savefig(os.path.join(exp_dir, f"test_example_{i}.png"))
        plt.close('all')

        if args.n_gridcells > 0:
            bptt_trainer.plot_single_test_examples(rnn, dataloader_test, n_examples=50,
                frame_dim = [args.frame_dim[0]//args.frame_subsampling, args.frame_dim[1]//args.frame_subsampling],
                exp_dir = exp_dir
            )

        if args.wandb:
            wandb.log({'test_example' : wandb.Image(figs[-1])})

        print("[*] Training complete!", flush=True)

    activity_part(trainer, exp_dir, dataloader_act, bptt_trainer)

    if args.wandb:
        wandb.finish()

    return exp_dir



def list_of_strings(arg):
    return arg.strip().split(',')
def list_of_floats(arg):
    return [float(n) for n in arg.strip().split(',')]

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # TRAINING PARAMETERS
    argparser.add_argument(
        '--behaviour', type=str, default=None, # required unless --curriculum is used
        help="Behaviour group of the agent (crawl, walk, run, adult)")
    argparser.add_argument(
        '--architecture', type=str, default='rnn', choices=['rnn', 'gru', 'lstm', 'rssm'],
        help="Recurrent architecture to use (rnn, gru, lstm, or rssm). Default is rnn.")
    argparser.add_argument(
        '--curriculum', type=list_of_strings, default=None,
        help="Comma-separated behaviours to train sequentially in a single run, "+\
        "carrying over weights from one step to the next. Example: crawl,walk,run. "+\
        "Each step is saved in its own directory. Overrides --behaviour.")
    argparser.add_argument(
        '--env', type=str, default='box_messy',
        help="Environment name (box_messy, circle_messy, ...)")
    argparser.add_argument(
        '--env_dim', type=float, default=0.635,
        help="Environment dimension on a single edge (box, circle: 0.635, triangle: 1). Default is 0.635")
    argparser.add_argument(
        '--pretrained_behav', type=list_of_strings, default=None,
        help="Set this parameter to fine-tune a pre-trained model. The list of behaviours the model was "+\
        "pre-trained on, separated by commas. Example: crawl,walk. Default is None, training from scratch")
    argparser.add_argument(
        '--name_prefix', type=str, default=None,
        help="Prefix to the model name. Default is None, i.e. no prefix.")
    argparser.add_argument(
        '--pretrained_model_folder', type=str, default=None,
        help="Set this parameter to fine-tune a pre-trained model specifying the exact directory.")
    argparser.add_argument(
        '--moredata', type=int, default=None,
        help="Positive integer to add more data to the training set, the number act as a "+\
        "counter for the seeds to be used. Default is None, i.e. no more data is added.")
    argparser.add_argument(
        '--n_gridcells', type=int, default=0,
        help="The number of grid cells to be used as input to the model. "+\
        "Default is zero, i.e. no grid cells input is given.")
    argparser.add_argument(
        '--gridcells_softmax', action=argparse.BooleanOptionalAction, default=False,
        help="Whether to softmax grid cells activity at each step. Default is False.")
    argparser.add_argument(
        '--gridcells_modules', type=list_of_floats, default=None,
        help="Grid cells modules. Default is None.")
    argparser.add_argument(
        '--gridcells_orientations', type=list_of_floats, default=None,
        help="Grid cells orientations. Default is None.")
    argparser.add_argument(
        '--n_future_pred', type=int, default=1,
        help="Number of future predictions to make. Default is 1")
    argparser.add_argument(
        '--stride', type=int, default=10,
        help="Stride to subsample the data, which is usually at 10 Hz. Default is 10")
    argparser.add_argument(
        '--reset_hidden_at', type=int, default=None,
        help="How often to reset the hidden state of the RNN. Default is None, never reset.")
    argparser.add_argument(
        '--bptt_steps', type=int, default=9,
        help="Number of steps to backpropagation through time. Default is 9")
    argparser.add_argument(
        '--latent_dim', type=int, default=500,
        help="Latent dimension of the RNN. For the RSSM this is the "+\
        "deterministic (GRU) state size. Default is 500")
    argparser.add_argument(
        '--stoch_dim', type=int, default=32,
        help="RSSM only: size of the stochastic latent z. The analysed latent "+\
        "activity then has dimension latent_dim + stoch_dim. Default is 32")
    argparser.add_argument(
        '--kl_scale', type=float, default=1.0,
        help="RSSM only: weight of the KL(posterior || prior) term. Default is 1.0")
    argparser.add_argument(
        '--free_nats', type=float, default=3.0,
        help="RSSM only: free-nats floor below which the KL is not penalised. Default is 3.0")
    argparser.add_argument(
        '--lr', type=float, default=5e-5,
        help="Learning rate. Default is 5e-5")
    argparser.add_argument(
        '--bias', action=argparse.BooleanOptionalAction,
        help="Whether to add bias to the RNN. Default is False.")
    argparser.add_argument(
        '--dropouts', type=list_of_floats, default=[0, 0, 0],
        help="List of dropouts (floats) to apply to the RNN. In the format: "+\
            "[in2hidden,hidden2hidden,hidden2out]. Default is [0,0,0] (no dropouts).")
    argparser.add_argument(
        '--nonlinearity', type=str, default='sigmoid',
        help="Non linearity of the RNN. Default is sigmoid")
    argparser.add_argument(
        '--hidden_reg', type=float, default=0.,
        help="Hidden state l2 regularization of the RNN. Default is 0")
    argparser.add_argument(
        '--weights_reg', type=float, default=0.,
        help="Weights regularization of the RNN. Default is 0")
    argparser.add_argument(
        '--epochs', type=int, default=1_500,
        help="Number of epochs to train the model. Default is 1_500")
    argparser.add_argument(
        '--seed', type=int, default=1,
        help="Random seed for reproducibility. Default is 1")
    argparser.add_argument(
        '--wandb', action=argparse.BooleanOptionalAction,
        help="Whether to log to wandb. Default is False.")
    
    # ACTIVITY PARAMETERS
    argparser.add_argument(
        '--behaviour_act', type=str, default=None,
        help="Behaviour group of the activity data (crawl, walk, run, adult). Default to argument \"behaviour\".")
    argparser.add_argument(
        '--activity_only', action=argparse.BooleanOptionalAction,
        help="Whether to only run the activity part of the code. Default to False, i.e. train and then run activity.")
    argparser.add_argument(
        '--ratemap_norm', type=str, default='minmax',
        help="How to normalize rate maps (minmax or sum). Default to minmax.")
    argparser.add_argument(
        '--activity_transform', type=str, default='identity',
        choices=['identity', 'softplus', 'minmax', 'halfshift'],
        help="Transform applied to latent activity before all spatial analyses. "
        "Use softplus for a non-negative activity representation, or minmax "
        "to scale each latent unit to [0, 1] across all trajectories and "
        "timesteps; default is identity.")
    argparser.add_argument(
        '--epoch_act', type=int, default=None,
        help="The epoch to load the model from. Default is None (last available epoch)")
    
       
    args = argparser.parse_args()

    if args.gridcells_modules is not None or args.gridcells_orientations is not None:
        if args.n_gridcells == 0:
            raise ValueError("Grid cells modules or orientations are defined but no grid cells are used")

    if args.activity_only and args.wandb:
        raise ValueError("Wandb logging is not supported in activity-only mode")

    if args.curriculum is None and args.behaviour is None:
        raise ValueError("Either --behaviour or --curriculum must be provided")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.curriculum is not None:
        # Train each behaviour sequentially in a single run, carrying over the
        # weights from the previous step. Each step saves its own checkpoints.
        prev_exp_dir = None
        for step_idx, behaviour in enumerate(args.curriculum):
            print(
                f"\n\n########## CURRICULUM STEP {step_idx+1}/{len(args.curriculum)}: "
                f"{behaviour} ##########\n"
            )
            args.behaviour = behaviour
            if step_idx == 0:
                # first step trains from scratch (unless user pinned a folder)
                args.pretrained_behav = args.pretrained_behav
            else:
                # subsequent steps fine-tune from the previous step's weights;
                # pretrained_behav sets the output sub-directory naming
                args.pretrained_behav = args.curriculum[:step_idx]
                args.pretrained_model_folder = prev_exp_dir
            prev_exp_dir = main(args)
    else:
        main(args)
