import torch
import numpy as np
import os
import wandb

class RNNTrainer():

    def __init__(self, args, data_dir, device, create_dir=True):
        self.args = args
        self.data_dir = data_dir
        self.device = device

        self.init_default_args()

        self.model_name = RNNTrainer.define_model_name(self.args)
        print(f"\n[*] Current model name:\n\t{self.model_name}")

        if create_dir:
            self.exp_dir = self.define_exp_dir()
            print(f"\n[+] Created model directory\n\t{self.exp_dir}")

    @staticmethod
    def define_model_name(args):
        # convert args to dict if it is a Namespace
        args = vars(args) if not isinstance(args, dict) else args

        # argsからモデル名を生成する(デフォルトはrnn)
        architecture = args.get('architecture', 'rnn')
        if architecture in {'gru', 'lstm'}:
            model_name = architecture.upper()
            if args['name_prefix'] : model_name += f'_{args["name_prefix"]}'
            model_name += f'_{args["behaviour"]}'
            return model_name

        # RSSM variants: RNN-style suffix plus stochastic-latent parameters
        if architecture in {'rssm', 'mtrssm', 'crssmv4'}:
            model_name = architecture.upper()
            if args['name_prefix'] : model_name += f'_{args["name_prefix"]}'
            if args['pretrained_model_folder'] : model_name += '_ft'
            if args['reset_hidden_at'] is not None : model_name += f'_reset{args["reset_hidden_at"]}'
            model_name += (
                f'_f{args["n_future_pred"]}_w{args["bptt_steps"]}_st{args["stride"]}'+
                f'_fss4_do{str(args["dropouts"]).replace(" ", "")}'+
                f'_lat{args["latent_dim"]}_stoch{args["stoch_dim"]}'+
                f'_kl{args["kl_scale"]}_fn{args["free_nats"]}'+
                f'_hreg{args["hidden_reg"]}_wreg{args["weights_reg"]}_s{args["seed"]:02d}'
            )
            if architecture == 'mtrssm':
                model_name += (
                    f'_hlat{args["higher_latent_dim"]}'
                    f'_hstoch{args["higher_stoch_dim"]}'
                    f'_ta{args["temporal_abstraction"]}'
                    f'_tau{args["lower_tau"]}-{args["higher_tau"]}'
                )
            elif architecture == 'crssmv4':
                model_name += (
                    f'_emb{args["embed_obs_dim"]}'
                    f'_coarse{args["coarse_dim"]}'
                    f'_cstoch{args["coarse_stoch_dim"]}'
                    f'_l0{args["w_l0_norm"]}'
                )
            if args.get('stoch_dist', 'normal') == 'categorical':
                model_name += f'_cat{args["stoch_n_class"]}'
            return model_name

        model_name = f'RNN'
        if args['name_prefix'] : model_name += f'_{args["name_prefix"]}'
        if args['pretrained_model_folder'] : model_name += '_ft'
        if args['moredata'] : model_name += f'_moredata{args["moredata"]}'
        if args['n_gridcells'] > 0:
            model_name += f'_gridcells{args["n_gridcells"]}' if not args['gridcells_softmax'] else f'_gridcellssm{args["n_gridcells"]}'
            model_name += f'_mod{str(args["gridcells_modules"]).replace(" ", "")}'
            model_name += f'_ori{str(args["gridcells_orientations"]).replace(" ", "")}'
        if args['reset_hidden_at'] is not None : model_name += f'_reset{args["reset_hidden_at"]}'

        model_name += (
            f'_f{args["n_future_pred"]}_w{args["bptt_steps"]}_st{args["stride"]}'+
            f'_fss4_do{str(args["dropouts"]).replace(" ", "")}'+
            f'_lat{args["latent_dim"]}_nl{args["nonlinearity"]}'+
            f'_hreg{args["hidden_reg"]}_wreg{args["weights_reg"]}_s{args["seed"]:02d}'
        )
        return model_name
    
    def init_default_args(self):
        """訓練と評価に必要な初期設定値を定義"""
        self.args.env_shape = self.args.env.split("_")[0]

        # Rat In A Box simulations information
        #  train/test/activityのseedを指定
        n_trials_train = 17
        n_trials_test = 4
        n_trials_act = 5
        if self.args.moredata is None:
            self.args.seeds_train = list(range(1, n_trials_train+1))
        else:
            start_seed = 1+n_trials_test+n_trials_act+(n_trials_train*self.args.moredata)
            self.args.seeds_train = list(range(
                start_seed, start_seed+n_trials_train
            ))

        self.args.seeds_test = list(range(
            n_trials_train+1, n_trials_train+n_trials_test+1
        ))
        self.args.seeds_act = list(range(
            n_trials_train+1, n_trials_train+n_trials_test+n_trials_act+1
        ))
        self.args.seeds_all = self.args.seeds_train + list(np.union1d(self.args.seeds_test, self.args.seeds_act))

        print(f'\tSeeds train: {self.args.seeds_train}\n\tSeeds test: {self.args.seeds_test}\n\tSeeds activity: {self.args.seeds_act}')
        # フレームレート、一回のシミュレート時間、フレームの次元、フレームの間引き設定
        self.args.fps = 10
        self.args.seconds = 720
        self.args.frame_dim = [128, 64]
        self.args.frame_subsampling = 4

        # 未来の予測に対する割引係数
        self.args.discount_factor = 0.7

        # 学習率スケジューラが始まるエポック数、最終的な減衰りつ、減衰率の計算
        self.args.lr_sched_start_epoch = self.args.epochs // 2
        self.args.lr_sched_final_gamma = 0.1
        self.args.lr_sched_gamma =\
            self.args.lr_sched_final_gamma**(1/self.args.lr_sched_start_epoch)

        # モデルの保存間隔、ログ出力間隔、勾配クリッピングの値を設定
        self.args.save_model_every = 500
        self.args.log_every = 50
        self.args.clip_value = 1.

    def define_exp_dir(self):
        """実験結果を保存するディレクトリを作成"""
        if self.args.pretrained_behav is not None:
            folder_name = '_'.join(self.args.pretrained_behav)
        else:
            folder_name = "vanilla"

        exp_dir = os.path.join(
            self.data_dir, self.args.env_shape, self.args.behaviour,
            "predictions", self.args.env, folder_name, self.model_name
        )
        os.makedirs(exp_dir, exist_ok=True)

        return exp_dir
    
    def get_model_name(self):
        return self.model_name
    def get_exp_dir(self):
        return self.exp_dir
    def get_args(self):
        return self.args


    def load_simulations(self):
        from utils.data_handler import read_video_files_lq

        videos, thetas, positions, velocities, rot_velocities =\
            [], [], [], [], []
        print("\n[*] Loading simulations from:")

        load_dirs = [
            os.path.join(
                self.data_dir, self.args.env_shape, self.args.behaviour,
                f"exp_dim{self.args.env_dim}_fps{self.args.fps}_s{self.args.seconds}_seed{s:02d}"
            ) for s in self.args.seeds_all
        ]
        for idx, ld in enumerate(load_dirs):
            # activity_onlyがTrueの場合、seeds_trainに含まれるシードのデータはスキップする
            if (
                self.args.activity_only and
                idx+1 in self.args.seeds_train
            ):
                videos.append([])
                thetas.append([])
                positions.append([])
                velocities.append([])
                rot_velocities.append([])
                continue
            video_lq = read_video_files_lq(
                os.path.join(ld, self.args.env),
                [self.args.frame_dim[0]//self.args.frame_subsampling, self.args.frame_dim[1]//self.args.frame_subsampling]
            )
            # 平滑化(n_frames,1,64,128)-> (n_frames, 64*128)
            videos.append(video_lq.reshape(video_lq.shape[0], -1))

            # load riab data
            thetas.append(np.expand_dims(
                np.load(os.path.join(ld, "riab_simulation/thetas.npy")).astype(np.float32),
                axis=-1
            ))
            positions.append(np.load(os.path.join(ld, "riab_simulation/positions.npy")).astype(np.float32))
            velocities.append(np.load(os.path.join(ld, "riab_simulation/velocities.npy")).astype(np.float32))
            rot_velocities.append(np.expand_dims(
                np.load(os.path.join(ld, "riab_simulation/rot_velocities.npy")).astype(np.float32),
                axis=-1
            ))
        print()
        return videos, thetas, positions, velocities, rot_velocities
    
    def check_shapes(self, data):
        """dataに含まれるデータが全て同じ形状かを確認する関数"""
        # check that all loaded data have the same shape
        shape = None
        for idx, d in enumerate(data):
            if len(d) == 0 : continue
            if shape is None:
                shape = d.shape
            else:
                if shape != d.shape:
                    raise ValueError(f"Shape {shape} != {d.shape} at index {idx}")
        if shape is None:
            raise ValueError("No data loaded, check the data directory and the environment name")
        return shape

    def convolve_videos(self, videos, positions, thetas):
        """CNNで動画を畳み込む関数だが使われていなさそう"""
        from utils import encode_video_cnn

        if "singleenv" in self.args.cnn:
            print("\n[+] Convolve videos with CNN pre-trained on single environment")
            cnn_ae_dir = os.path.join(
                self.data_dir, self.args.env_shape,
                "cnn_ae",
                self.args.env
            )
            model_name = 'CNN_AE_lq_k[(4,4),(5,5),(3,3),(3,3)]_s[2,2,1,1]_ch[8,8,16,16]_ld150_actrelu_dropout0.0'
        elif "multienv" in self.args.cnn:
            print("\n[+] Convolve videos with CNN pre-trained on multiple environments")
            cnn_ae_dir = os.path.join(
                self.data_dir, "multi_env", "cnn_ae",
                "box_landmarks+circle_messy+triangle_messy+box_messy"
            )
            model_name = 'CNN_AE_lq_k[(4,4),(5,5),(3,3),(3,3)]_s[2,2,1,1]_ch[8,8,16,16]_ld150_actrelu_dropout0.0'

        positions_list, thetas_list = [], []
        for idx, (pos, thet) in enumerate(zip(positions, thetas)):
            cnn_ae, _, v = encode_video_cnn(
                os.path.join(cnn_ae_dir, model_name),
                videos[idx],
                self.device
            )
            positions_list.append(pos.copy())
            thetas_list.append(thet.copy().squeeze())
            videos[idx] = v

        return cnn_ae, videos, positions_list, thetas_list


    def preprocess_data(self, videos, velocities, rot_velocities):
        """画像、速度、回転速度のデータをmin,maxで0,1に正規化する関数"""
        from utils.data_handler import minmax_normalization

        print("\tNormalizing data with min-max scaling")
        videos, velocities, rot_velocities = minmax_normalization(
            videos, velocities, rot_velocities
        )

        return videos, velocities, rot_velocities

    def combine_videos(self, videos, velocities, rot_velocities, positions, thetas):
        """複数のシミュレーション思考を訓練・テスト・活動用に分割する関数"""
        # train,test,activityのデータに分割
        videos_multisubs, videos_multisubs_test, videos_multisubs_act = [], [], []
        velocities_multisubs, velocities_multisubs_test, velocities_multisubs_act = [], [], []
        rot_velocities_multisubs, rot_velocities_multisubs_test, rot_velocities_multisubs_act = [], [], []
        positions_multisubs, positions_multisubs_test, positions_multisubs_act = [], [], []
        thetas_multisubs, thetas_multisubs_test, thetas_multisubs_act = [], [], []

        stride = self.args.stride

        if stride > 1:
            from utils.data_handler import create_multiple_subsampling
            
            print(f"\n[+] Creating multiple subsampling with stride {stride}")

            for s, video, velocity, rot_velocity, pos, thet in\
            zip(self.args.seeds_all, videos, velocities, rot_velocities, positions, thetas):
                if s in self.args.seeds_test:
                    videos_multisubs_test.append(create_multiple_subsampling(video, stride))
                    velocities_multisubs_test.append(create_multiple_subsampling(velocity, stride, is_velocity=True))
                    rot_velocities_multisubs_test.append(create_multiple_subsampling(rot_velocity, stride, is_velocity=True))
                    positions_multisubs_test.append(create_multiple_subsampling(pos, stride))
                    thetas_multisubs_test.append(create_multiple_subsampling(thet, stride))
                    
                    videos_multisubs_act.append(create_multiple_subsampling(video, stride))
                    velocities_multisubs_act.append(create_multiple_subsampling(velocity, stride, is_velocity=True))
                    rot_velocities_multisubs_act.append(create_multiple_subsampling(rot_velocity, stride, is_velocity=True))
                    positions_multisubs_act.append(create_multiple_subsampling(pos, stride))
                    thetas_multisubs_act.append(create_multiple_subsampling(thet, stride))
                elif s in self.args.seeds_act:
                    videos_multisubs_act.append(create_multiple_subsampling(video, stride))
                    velocities_multisubs_act.append(create_multiple_subsampling(velocity, stride, is_velocity=True))
                    rot_velocities_multisubs_act.append(create_multiple_subsampling(rot_velocity, stride, is_velocity=True))
                    positions_multisubs_act.append(create_multiple_subsampling(pos, stride))
                    thetas_multisubs_act.append(create_multiple_subsampling(thet, stride))
                elif not self.args.activity_only:
                    videos_multisubs.append(create_multiple_subsampling(video, stride))
                    velocities_multisubs.append(create_multiple_subsampling(velocity, stride, is_velocity=True))
                    rot_velocities_multisubs.append(create_multiple_subsampling(rot_velocity, stride, is_velocity=True))
                    positions_multisubs.append(create_multiple_subsampling(pos, stride))
                    thetas_multisubs.append(create_multiple_subsampling(thet, stride))
        else:
            print(f"\n[*] Combining videos as is because stride={stride}")
            for s, video, velocity, rot_velocity, pos, thet in\
            zip(self.args.seeds_all, videos, velocities, rot_velocities, positions, thetas):
                if s in self.args.seeds_test:
                    videos_multisubs_test.append(video[None, ...])
                    velocities_multisubs_test.append(velocity[None, ...])
                    rot_velocities_multisubs_test.append(rot_velocity[None, ...])
                    positions_multisubs_test.append(pos[None, ...])
                    thetas_multisubs_test.append(thet[None, ...])

                    videos_multisubs_act.append(video[None, ...])
                    velocities_multisubs_act.append(velocity[None, ...])
                    rot_velocities_multisubs_act.append(rot_velocity[None, ...])
                    positions_multisubs_act.append(pos[None, ...])
                    thetas_multisubs_act.append(thet[None, ...])
                elif s in self.args.seeds_act:
                    videos_multisubs_act.append(video[None, ...])
                    velocities_multisubs_act.append(velocity[None, ...])
                    rot_velocities_multisubs_act.append(rot_velocity[None, ...])
                    positions_multisubs_act.append(pos[None, ...])
                    thetas_multisubs_act.append(thet[None, ...])
                elif not self.args.activity_only:
                    videos_multisubs.append(video[None, ...])
                    velocities_multisubs.append(velocity[None, ...])
                    rot_velocities_multisubs.append(rot_velocity[None, ...])
                    positions_multisubs.append(pos[None, ...])
                    thetas_multisubs.append(thet[None, ...])

        if not self.args.activity_only:
            video_train, velocity_train, rot_velocity_train, positions_train, thetas_train =\
                np.concatenate(videos_multisubs, axis=0), np.concatenate(velocities_multisubs, axis=0),\
                np.concatenate(rot_velocities_multisubs, axis=0), np.concatenate(positions_multisubs, axis=0),\
                np.concatenate(thetas_multisubs, axis=0)
            print(
                f"\tvideo_train: {video_train.shape}\n"+
                f"\tvelocity_train: {velocity_train.shape}\n\trot_velocity_train: {rot_velocity_train.shape}\n"+
                f"\tpositions_train: {positions_train.shape}\n\tthetas_train: {thetas_train.shape}\n"
            )
        else:
            video_train, velocity_train, rot_velocity_train, positions_train, thetas_train =\
                None, None, None, None, None
            
        video_test, velocity_test, rot_velocity_test, positions_test, thetas_test =\
            np.concatenate(videos_multisubs_test, axis=0), np.concatenate(velocities_multisubs_test, axis=0),\
            np.concatenate(rot_velocities_multisubs_test, axis=0), np.concatenate(positions_multisubs_test, axis=0),\
            np.concatenate(thetas_multisubs_test, axis=0)
        video_act, velocity_act, rot_velocity_act, positions_act, thetas_act =\
            np.concatenate(videos_multisubs_act, axis=0), np.concatenate(velocities_multisubs_act, axis=0),\
            np.concatenate(rot_velocities_multisubs_act, axis=0), np.concatenate(positions_multisubs_act, axis=0),\
            np.concatenate(thetas_multisubs_act, axis=0)

        # shape is now (n_trajectories, n_frames, n_features)
        print(
            f"\tvideo_test: {video_test.shape}\n\tvelocity_test: {velocity_test.shape}\n"+
            f"\trot_velocity_test: {rot_velocity_test.shape}\n"+
            f"\tpositions_test: {positions_test.shape}\n\tthetas_test: {thetas_test.shape}\n\n"+
            f"\tvideo_act: {video_act.shape}\n\tvelocity_act: {velocity_act.shape}\n"+
            f"\trot_velocity_act: {rot_velocity_act.shape}\n"+
            f"\tpositions_act: {positions_act.shape}\n\tthetas_act: {thetas_act.shape}"
        )

        return (
            video_train, velocity_train, rot_velocity_train, positions_train, thetas_train,
            video_test, velocity_test, rot_velocity_test, positions_test, thetas_test,
            video_act, velocity_act, rot_velocity_act, positions_act, thetas_act
        )

    def generate_dataloader(
        self, video, velocity, rot_velocity, positions, thetas, verbose=False
    ):
        # we now introduce windows to the data, so that each batch contains
        # STRIDE*SEEDS examples of the same time window length and the number
        # of batches corresponds to the number of windows that fit into the experiment length

        # gridcellを注入する場合
        if self.args.n_gridcells > 0:
            from architectures.recurrent_gridcells.datasets import WindowedPredictionDataset
            dataloader = torch.utils.data.DataLoader(
                WindowedPredictionDataset(
                    video,
                    velocity, rot_velocity,
                    self.args.n_gridcells, self.args.gridcells_modules,
                    self.args.gridcells_orientations, self.args.gridcells_softmax,
                    positions, thetas,
                    self.args.bptt_steps, n_future_pred=self.args.n_future_pred
                ),
                shuffle=False,
                num_workers=self.args.num_workers,
            )
            if verbose:
                print("\tDataloader length:", len(dataloader))
                print()
                for i, batch in enumerate(dataloader):
                    if (i == 0) or (i == len(dataloader)-1):
                        scene, vel, rot_vel, gc, pos, thet, labels = batch

                        print(f"\tBATCH {i}")
                        print(f"\tvideo: {scene.shape}")
                        print(f"\tvelocity: {vel.shape}\n\trotational velocity: {rot_vel.shape}")
                        print(f"\tgrid cells: {gc.shape}")
                        print(f"\tlabels: {labels.shape}\n\tpositions: {pos.shape}\n\tthetas: {thet.shape}\n")
            return dataloader

        from architectures.recurrent.datasets import WindowedPredictionDataset
        dataloader = torch.utils.data.DataLoader(
            WindowedPredictionDataset(
                video,
                velocity, rot_velocity,
                positions, thetas,
                self.args.bptt_steps, n_future_pred=self.args.n_future_pred
            ),
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        if verbose:
            print("\tDataloader length:", len(dataloader))
            print()
            for i, batch in enumerate(dataloader):
                if (i == 0) or (i == len(dataloader)-1):
                    scene, vel, rot_vel, pos, thet, labels = batch

                    print(f"\tBATCH {i}")
                    print(f"\tvideo: {scene.shape}\n")
                    print(f"\tvelocity: {vel.shape}\n\trotational velocity: {rot_vel.shape}")
                    print(f"\tlabels: {labels.shape}\n\tpositions: {pos.shape}\n\tthetas: {thet.shape}\n")
        return dataloader

    def generate_lightning_world_dataloader(
        self, video, velocity, rot_velocity, positions, thetas
    ):
        """Create long sequence batches consumed directly by WorldModel."""
        from architectures.recurrent.datasets import WindowedPredictionDataset
        from architectures.recurrent.lightning_training import (
            WorldModelBatchCollator,
        )

        sequence_steps = self.args.bptt_steps * self.args.windows_per_lightning_batch
        if self.args.bptt_steps < 1 or self.args.windows_per_lightning_batch < 1:
            raise ValueError(
                "bptt_steps and windows_per_lightning_batch must be positive"
            )
        height = self.args.frame_dim[1] // self.args.frame_subsampling
        width = self.args.frame_dim[0] // self.args.frame_subsampling
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.args.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "collate_fn": WorldModelBatchCollator((height, width)),
        }
        if self.args.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )
        dataset = WindowedPredictionDataset(
            video,
            velocity,
            rot_velocity,
            positions,
            thetas,
            sequence_steps,
            n_future_pred=self.args.n_future_pred,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"Sequence length {sequence_steps} is too long for this dataset"
            )
        return torch.utils.data.DataLoader(
            dataset,
            **loader_kwargs,
        )


    def load_model_pretrained(self):
        import re

        load_model_dir = os.path.join(
            self.data_dir,
            self.args.env_shape,
            self.args.pretrained_behav[-1],
            "predictions", self.args.env,
            '_'.join(self.args.pretrained_behav[:-1]) if len(self.args.pretrained_behav)>1 else "vanilla",
            self.model_name
        ) if self.args.pretrained_model_folder is None else self.args.pretrained_model_folder

        rnn_files = [f for f in os.listdir(load_model_dir) if re.match(r"rnn_epoch\d+\.pth", f)]
        epoch_max = max([int(re.search(r'\d+', f).group()) for f in rnn_files])
        load_model_dir = os.path.join(load_model_dir, f"rnn_epoch{epoch_max}.pth")
        print(f"\n[+] Loading model from {load_model_dir}")
        rnn = torch.load(
            load_model_dir,
            weights_only=False,
            map_location=torch.device(self.device)
        ).to(self.device)

        return rnn

    def define_training_objects(self, scene_dim, vel_dim, output_dim):
        """RNNモデルと学習設定一式を生成する関数"""
        from architectures.losses_custom import DiscountLoss

        rnn_loaded = None
        # 事前学習済みモデルが指定されている場合、モデルをロードする
        if self.args.pretrained_model_folder or self.args.pretrained_behav:
            rnn_loaded = self.load_model_pretrained()

        # define RNN architecture
        if getattr(self.args, 'architecture', 'rnn') in {'gru', 'lstm'}:
            if self.args.n_gridcells > 0:
                raise NotImplementedError(
                    f"{self.args.architecture.upper()} architecture does not support grid cells input"
                )
            if self.args.architecture == 'gru':
                from architectures.recurrent.gru_bptt import GRU
                recurrent_model = GRU
            else:
                from architectures.recurrent.lstm_bptt import LSTM
                recurrent_model = LSTM
            # LSTMまたはGRUモデルをインスタンス化する
            # [scene_dim+vel_dim]->[output_dim]
            rnn = recurrent_model(
                self.device,
                scene_dim+vel_dim, output_dim,
                latent_dim = self.args.latent_dim,
                nonlinearity = self.args.nonlinearity,
                dropouts = self.args.dropouts,
                bias = self.args.bias,
            ).to(self.device)
        elif getattr(self.args, 'architecture', 'rnn') in {'rssm', 'mtrssm'}:
            if self.args.n_gridcells > 0:
                raise NotImplementedError(
                    f"{self.args.architecture.upper()} architecture does not "
                    "support grid cells input"
                )
            height = self.args.frame_dim[1] // self.args.frame_subsampling
            width = self.args.frame_dim[0] // self.args.frame_subsampling
            if output_dim != scene_dim or scene_dim != height * width:
                raise ValueError(
                    f"{self.args.architecture.upper()} expects flattened "
                    f"grayscale frames with {height * width} features, got "
                    f"scene_dim={scene_dim} and output_dim={output_dim}"
                )
            from architectures.recurrent.world_model_training import (
                build_world_model,
            )
            rnn = build_world_model(
                self.args,
                architecture=self.args.architecture,
                action_dim=vel_dim,
                frame_shape=(height, width),
            ).to(self.device)
        elif getattr(self.args, 'architecture', 'rnn') == 'crssmv4':
            if self.args.n_gridcells > 0:
                raise NotImplementedError(
                    "CRSSMV4 architecture does not support grid cells input"
                )
            height = self.args.frame_dim[1] // self.args.frame_subsampling
            width = self.args.frame_dim[0] // self.args.frame_subsampling
            if output_dim != scene_dim or scene_dim != height * width:
                raise ValueError(
                    "CRSSMV4 expects flattened grayscale frames with "
                    f"{height * width} features, got scene_dim={scene_dim} "
                    f"and output_dim={output_dim}"
                )
            from architectures.recurrent.crssmv4_training import (
                build_crssmv4_world_model,
            )
            rnn = build_crssmv4_world_model(
                self.args,
                action_dim=vel_dim,
                frame_shape=(height, width),
            ).to(self.device)
        elif self.args.n_gridcells > 0:
            from architectures.recurrent_gridcells.rnn_bptt import RNN
            rnn = RNN(
                self.device,
                scene_dim+vel_dim, self.args.n_gridcells, output_dim,
                latent_dim = self.args.latent_dim,
                nonlinearity = self.args.nonlinearity,
                dropouts = self.args.dropouts,
                bias = self.args.bias,
            ).to(self.device)
        else:
            from architectures.recurrent.rnn_bptt import RNN
            rnn = RNN(
                self.device,
                scene_dim+vel_dim, output_dim,
                latent_dim = self.args.latent_dim,
                nonlinearity = self.args.nonlinearity,
                dropouts = self.args.dropouts,
                bias = self.args.bias,
            ).to(self.device)

        print("\n[*] Model parameters:")
        for name, p in rnn.named_parameters():
            print(f"\t{name}, shape {p.shape}, requires grad {p.requires_grad}")
            # 事前学習済みのモデルが指定されている場合、重みをロードする
            if rnn_loaded is not None and name in rnn_loaded.state_dict():
                print(f"\t\t+++ Loading weights from previous model +++")
                rnn.state_dict()[name].copy_(rnn_loaded.state_dict()[name].detach())

        # 損失関数(L1Loss)の定義
        loss_fn = DiscountLoss(
            torch.nn.L1Loss(reduction='none'), discount_factor=self.args.discount_factor, n_future_pred=self.args.n_future_pred
        ).to(self.device)

        optimizer = torch.optim.RMSprop(
            rnn.parameters(),
            lr=self.args.lr
        )

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.args.lr_sched_gamma
        )

        return rnn, loss_fn, optimizer, lr_scheduler

    def define_bptt_trainer(self, optimizer, loss_fn):
        """BPTTの学習を行うためのTrainerクラスを定義する関数"""
        if getattr(self.args, 'architecture', 'rnn') in {'rssm', 'mtrssm'}:
            from architectures.recurrent.world_model_training import (
                TrainerWorldModel,
            )
            return TrainerWorldModel(
                self.args, optimizer, loss_fn, self.device
            )
        if getattr(self.args, 'architecture', 'rnn') == 'crssmv4':
            from architectures.recurrent.crssmv4_training import TrainerCRSSMV4
            return TrainerCRSSMV4(
                self.args, optimizer, loss_fn, self.device
            )

        if self.args.n_gridcells > 0:
            from architectures.recurrent_gridcells.training import TrainerBPTT
        else:
            from architectures.recurrent.training import TrainerBPTT

        return TrainerBPTT(
            self.args, optimizer, loss_fn, self.device
        )

    def train(self, rnn, bptt_trainer, dl_train, dl_test, lr_sched):
        import time

        loss_train_list = []
        loss_test_list = []
        epoch_time_sum = 0

        for epoch in range(self.args.epochs):
            start = time.time()
            # 学習実行
            rnn, dict_train = bptt_trainer.train_epoch(rnn, dl_train)
            loss_train = dict_train["loss/tot_loss_train"]
            # テスト実行
            dict_test = bptt_trainer.test_epoch(rnn, dl_test)
            loss_test = dict_test["loss/tot_loss_test"]

            if epoch >= self.args.lr_sched_start_epoch:
                lr_sched.step()

            end = time.time()
            epoch_time_sum += (end-start)

            if self.args.wandb:
                wandb.log({**dict_train, **dict_test})
            
            if (epoch+1)%self.args.log_every==0:
                print(f'[*] EPOCH {epoch+1}:')
                print(f'\tLOSS train {loss_train:.5f} test {loss_test:.5f}')
                print(f"\t{epoch_time_sum/(epoch+1):.3f} seconds per epoch")
                print(flush=True)

            if (epoch+1)%self.args.save_model_every==0:
                print(f"[+] Saving model at epoch {epoch+1}...\n")
                torch.save(rnn, os.path.join(self.exp_dir, f"rnn_epoch{epoch+1}.pth"))

            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)
        
        np.save(os.path.join(self.exp_dir, 'loss_train.npy'), np.array(loss_train_list))
        np.save(os.path.join(self.exp_dir, 'loss_test.npy'), np.array(loss_test_list))

        return rnn
