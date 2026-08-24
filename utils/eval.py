import os
from omegaconf import DictConfig
import einops
import numpy as np
import torch
import yaml
import time
from typing import Union
from einops import rearrange
from glob import glob
from natsort import natsorted
from hydra.utils import instantiate
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from src.modules.world import WorldModel, CoarseWorldModel
from utils.loss import CalcFreeEnergy
from utils.utils import mytorch, torch_fix_seed, actions_to_positions
from utils.states import stack_worlds, Worlds, ExpectedFreeEnergy, stack_efe, get_dist
from src.utils.world_visualize import (save_selected_frames_as_pdf, visualize_feature, visualize_joint_prediction,
                                 visualize_histogram, decomposite_feature, array2gif,
                                 visualise_continuous, visualize_comparing_video,
                                 decomposite_two_features, visualize_img_seq,
                                 decomposite_any_features, plot_3d, plot_2d)
from src.data.dataset import get_all_data
from src.utils.visualize import plot_posi_var, plot_efe_bar, decomposite_feature_2
import warnings
from utils.config import AgentConfig, ExperimentConfig, load_config, CRSSMV4Config, CRSSMConfig
from src.data.world_dataset import split_id, setup_data
from src.data.make_predata import (joint_detransform, joint_transform, scale_obs, unscale_obs)
from src.modules.diffusion_policy import DiffusionPolicy
import torch.nn.functional as F

from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.")
warnings.filterwarnings("ignore", ".*Input tensor has dimensions outside of.*")
warnings.filterwarnings(
    "ignore", ".*I found a path object that I don't think is part of a bar chart.*"
)
os.environ["MUJOCO_GL"] = "egl"


def check_data(
        data_dir: str,
        data_idx: int,
        test_images: bool = False
        ):
    if test_images:
        image_name = "predictions/test_images"
    else:
        image_name = "predictions/image_states"

    images = np.load(f"{data_dir}/{image_name}.npy")
    print("images", images.shape, images.min(), images.max())
    images = images[data_idx]
    visualize_img_seq(
        f"{data_dir}/data_check",
        images,
        show_num=5
    )

# @torch.no_grad()
# def efe_(
#     model_name: str,
#     alpha: float = 0.1,
#     seed: int = 0,
#     device: int = 0,
#     preferred_state: str = "lerobot-trim/bL_lB",
#     init_state: str = "lerobot-trim/bL_lB",
#     pref_std: float = 0.1,
#     sample_num: int= 256,
#     use_nce: bool = False,
#     stepwise: bool = False,
#     warmup: int = 5,
#     use_esseintial: bool = False
# ):

#     conf_path = f"models/cfg/{model_name}.yaml"

#     config = load_config(conf_path)
#     config.alpha = alpha
#     config.seed = seed
#     config.device = device

#     torch_fix_seed(config.seed)

#     data_cfg = load_config(f"data/{config.data_dir}/config.yaml")

#     env_cfg = dict(
#         obs_shape=data_cfg.obs_shape,
#         action_dim=data_cfg.action_dim,
#     )


#     print("envrionment:", env_cfg)
#     config.update(env_cfg)
#     OmegaConf.resolve(config)
#     config: ExperimentConfig = instantiate(
#         config,
#     )
#     action_states, image_states = setup_data(
#         config.data_dir,
#         config.datamodule.action_diff,
#         data_cfg.gripper_dim
#     )
#     # image_states = image_states[:, indices.validation]

#     log_path = f"reports/{config.data_dir}_{model_name}/alpha:{config.alpha}/seed:{config.seed}/efe_test"
#     os.makedirs(log_path, exist_ok=True)

#     action_vqvae_path = f"models/params/{config.action_vqvae.load_name}/seed:{config.seed}"
#     action_vqvae = ActionVQVAE(
#         config.action_vqvae
#     )
#     action_vqvae.load_state_dict(torch.load(f"{action_vqvae_path}/ActionVQVAE.ckpt")["state_dict"])
#     action_vqvae.eval()
#     action_vqvae.freeze()
    
#     world_path = f"models/params/{config.world.load_name}/alpha:{config.alpha}/seed:{config.seed}"
#     if isinstance(config.world.dynamics_cfg, CRSSMV4Config) or isinstance(config.world.dynamics_cfg, CRSSMConfig):
#         world = CoarseWorldModel(
#             config.world
#         )
#     else:
#         world = WorldModel(
#             config.world
#             )
#     world.load_state_dict(torch.load(f"{world_path}/{world.__class__.__name__}.ckpt", map_location="cpu")["state_dict"])
#     world.eval()
#     world.freeze()

#     agent_path = f"models/params/{model_name}/alpha:{config.alpha}/seed:{config.seed}"

#     config.agent.stoch_dim = world.dynamics.stoch_dim
#     if isinstance(config.world.dynamics_cfg, CRSSMV4Config):
#         config.agent.coarse_stoch_dim = world.dynamics.coarse_stoch_dim
        
#     agent = Agent(config.agent)

#     agent.load_state_dict(torch.load(f"{agent_path}/{agent.__class__.__name__}.ckpt")["state_dict"])
#     agent.eval()
#     agent.freeze()
    
#     data_cfg = load_config(f"data/{config.data_dir}/config.yaml")

#     np_preferred_obs = np.load(
#         f'preference/{preferred_state}.npy')
#     plt.imsave(f"{log_path}/preferred_obs.png", np_preferred_obs)
#     preferred_obs = scale_obs(
#         np_preferred_obs[None, ...]).to(world.device)
#     preferred_obs = preferred_obs.permute(0, 3, 1, 2)
#     print("preferred_obs", preferred_obs.shape, preferred_obs.max(), preferred_obs.min())
#     np_image_state = np.load(
#         f'preference/{init_state}.npy')
#     plt.imsave(f"{log_path}/init_obs.png", np_image_state)
#     image_state = scale_obs(
#         np_image_state[None, ...]).to(world.device)
#     image_state = image_state.permute(0, 3, 1, 2)
#     print("init_obs", image_state.shape, image_state.max(), image_state.min())
#     pref_obs_embed = world.obs_encoder(preferred_obs)
#     world.cuda()
#     embed_for_critic = []
#     world.cpu()



#     home_position = data_cfg.mean
#     home_position = torch.tensor([home_position]).to(device).float()
#     current_position = home_position

#     first_action = joint_transform(
#         home_position.unsqueeze(0).cpu().numpy(),
#         data_cfg,
#         action_diff=config.datamodule.action_diff,
#     )[0]

#     batch_size = 1

#     # move home position
#     joint_call = current_position.squeeze().clone().detach().cpu()


#     observations = []
#     efes = []
#     predicted_obss = []
#     argmin_actions = []

        
#     # check robot_vision
#     if use_esseintial:
#         essential_indices = [345, 410] 
#         essential_action = action_states[:config.chunk_size, essential_indices].transpose(1, 0)
#         print("essential_action", essential_action.shape, essential_action.max(), essential_action.min())
        
#         essential_action = rearrange(essential_action, 'b t c -> b (t c)')
#         essential_action = action_vqvae.encoder(essential_action)
#         ctx_action, *_ = action_vqvae.quantize(essential_action)
#     else:
#         ctx_action = action_vqvae.vq_layer.all_codes
#     print("ctx_action", ctx_action.shape)


#     embed_obs = world.obs_encoder(image_state.to(
#         world.device).reshape([1, -1]))

#     world.dynamics.init_latent(batch_size, embed_obs)

#     action = first_action.to(world.device).reshape([1, -1])


#     for _ in range(warmup):
#         embed_obs = world.obs_encoder(
#             image_state).reshape([1, -1])

#         states, *_ = world.dynamics.step(world.action_transform(action), embed_obs)

#     efe, predicted_obs, argmin_action = calc_efe(
#         states, 
#         ctx_action,
#         world, 
#         agent, 
#         action_vqvae,
#         preferred_obs, 
#         pref_obs_embed, 
#         pref_std, 
#         sample_num,
#         embed_obs,
#         use_nce,
#         stepwise
#     )

#     efes.append(efe)
#     predicted_obss.append(predicted_obs)
#     argmin_actions.append(argmin_action)
#     visualize_efe(f'{log_path}/efe_test', efes.efe.detach().clone().cpu().numpy(
#     ), observations.squeeze().permute(1,2,0)*255, np_preferred_obs*255, predicted_obss[argmin_actions], argmin=argmin_actions)
#     visualize_efe(f"{log_path}/efe_test", efes.epistemic.detach().clone().cpu().numpy(
#     ), observations.squeeze().permute(1,2,0)*255, np_preferred_obs*255, predicted_obss[argmin_actions], fig_name="intrinsic", argmin=argmin_actions)
#     visualize_efe(f"{log_path}/efe_test", efes.extrinsic.detach().clone().cpu().numpy(
#     ), observations.squeeze().permute(1,2,0)*255, np_preferred_obs*255, predicted_obss[argmin_actions], fig_name="extrinsic", argmin=argmin_actions)
#     visualize_efe(f"{log_path}/efe_test", efes.nce.detach().clone().cpu().numpy(
#     ), observations.squeeze().permute(1,2,0)*255, np_preferred_obs*255, predicted_obss[argmin_actions], fig_name="nce", argmin=argmin_actions)
#     visualize_predicted_obs(f"{log_path}/efe_test", predicted_obss)


#一つの方策についてefeを計算（サンプリングを並列処理）
def calc_efe(
        states: Worlds, 
        ctx_actions: torch.Tensor, #
        world: Union[WorldModel, CoarseWorldModel], 
        preferred_obs: torch.Tensor, 
        pref_std: float = 0.1,
        n_sample: int = 1, #Nc: 上階層のposteriorからのサンプル数
        inv_tmp: float = 1.,
        n_sample_d: int = 1, #Nd： 低階層のpriorからのサンプル数
        ):
    
    """
    ctx_actions: Length, dim
    preferred_obs: c, h, w
    """
    world.cuda()
    preferred_obs = preferred_obs.to(world.device)
    if preferred_obs.ndim == 3:
        preferred_obs = preferred_obs.unsqueeze(0).expand(ctx_actions.shape[0], -1, -1, -1)
    else:
        preferred_obs = preferred_obs.unsqueeze(0).expand(ctx_actions.shape[0], -1)
    
    ctx_actions = ctx_actions.to(world.device)
    

    pred_states = step_imagination(
        states,
        ctx_actions,
        world
        )
    
    pred_coarse = pred_states.coarse #l, 1, 32
    c_posterior = pred_states.c_posterior #1, l, 4, 4
    pred_determ = pred_states.determ

    c_posterior_dist = get_dist(c_posterior)
    c_posterior_stoch = c_posterior_dist.sample([n_sample]).reshape([n_sample, len(ctx_actions), -1]) #Nc, l, 16
    c_posterior_stoch = 2*c_posterior_stoch-1
    
    pred_coarse = torch.permute(pred_coarse.expand(-1, n_sample, -1), (1,0,2)) #Nc, l, 32
    pred_determ = torch.permute(pred_determ.expand(-1, n_sample, -1), (1,0,2))    

    coarse_latent_states = torch.cat([pred_coarse, c_posterior_stoch], dim=-1) #Nc, l, 48
    
    d_prior = world.dynamics.d_prior(pred_determ) #Nc, l, 8, 8
    d_prior_dist = get_dist(d_prior) #Nc, l, 8, 8
    d_prior_stoch = d_prior_dist.rsample([n_sample_d]).reshape( #Nc*Nd, l, 64
                            n_sample_d**2, 
                            -1,
                            d_prior.stoch.shape[-1]
                        )
    d_prior_stoch = d_prior_stoch*2-1
    
    pred_determ = pred_determ.unsqueeze(0).expand(n_sample_d, -1, -1, -1).flatten(0, 1) #Nc*Nd, l, 128
    coarse_latent_states = coarse_latent_states.unsqueeze(0).expand(n_sample_d, -1, -1, -1).flatten(0, 1) #Nc*Nd, l, 48
    
    predicted_obs, _ = world._decode_obs(latent_states=torch.cat([pred_determ, #Nc*Nd, l, c, h, w
                                                               d_prior_stoch,
                                                               coarse_latent_states], dim=-1))
    imaginations = predicted_obs[0]
    embed_pred_obs = world.obs_encoder(predicted_obs) #Nc*Nd, l, 128

    if world.dynamics.coarse_obs == "determ":
        d_posterior = world.dynamics.d_posterior.forward(
                pred_determ,
                embed_pred_obs,
                inv_tmp=inv_tmp
                )
        d_prior = d_prior.unsqueeze(0).expand(n_sample_d).flatten(0, 1)
        d_posterior_dist = get_dist(d_posterior)
        d_prior_dist = get_dist(d_prior)
    
    if predicted_obs.ndim != 3:
        predicted_obs = pred_coarse.unsqueeze(0).expand(n_sample_d, -1, -1, -1).flatten(0, 1)
        
    efe = CalcFreeEnergy.expected_alpha_sub(
            d_prior, #if world.dynamics.coarse_obs == "determ" else prior.unsqueeze(0).expand(n_sample*n_sample_d),
            d_posterior, #if world.dynamics.coarse_obs == "determ" else posterior,
            predicted_obs,
            preferred_obs,
            pref_std=pref_std,
            alpha=world.alpha,
            pixel=True
            )


    efe_value = -efe["extrinsic"].mean() - efe["epistemic"].mean()

    efes = ExpectedFreeEnergy(
            efe["epistemic"].mean(),
            efe["extrinsic"].mean(),
            efe["nce"],
            efe_value,
            )
    world.cpu()
    ctx_actions = ctx_actions.cpu()
    return efes, imaginations


@torch.no_grad()
def step_imagination(
    states: Worlds,
    ctx_action: torch.Tensor,
    world: Union[WorldModel, CoarseWorldModel],
    ):
    """
    ctx_action: Length, dim
    """
    if isinstance(world, CoarseWorldModel):
        if hasattr(world.dynamics.precise_dyn, "hidden"):
            p_hidden_states = world.dynamics.precise_dyn.hidden.detach().clone()
        if hasattr(world.dynamics.coarse_dyn, "hidden"):
            c_hidden_states = world.dynamics.coarse_dyn.hidden.detach().clone()
    else:
        if hasattr(world.dynamics.rnn, "hidden"):
            hidden_states = world.dynamics.rnn.hidden.detach().clone()
    # print(ctx_action.shape) #l, 32
    # expanded_states = states.detach().clone().expand(1)
    world.dynamics.set_prev_states(states.to(world.device),set_mtrnn_hidden=False)
    policies = ctx_action.to(world.device).unsqueeze(0) #1, l, d

    start_time = time.time()
    history = []
    for t in range(policies.shape[1]):
        pred_states, _ = world.dynamics.step(policies[:, t]) 
        # print(pred_states.determ.shape) #1, 128
        history.append(pred_states)
    end_time = time.time()
    print("time for imagination: stepwise", end_time-start_time)
    
    # #imagination前に状態を戻しておく
    world.dynamics.set_prev_states(states, set_mtrnn_hidden=False)
    if isinstance(world, CoarseWorldModel):
        if hasattr(world.dynamics.precise_dyn, "hidden"):
            world.dynamics.precise_dyn.hidden = p_hidden_states
        if hasattr(world.dynamics.coarse_dyn, "hidden"):
            world.dynamics.coarse_dyn.hidden = c_hidden_states
    else:
        if hasattr(world.dynamics.rnn, "hidden"):
            world.dynamics.rnn.hidden = hidden_states

    return stack_worlds(history)


def action_select(world: str,
             policy: str,
             current_states,
             goal,
             observations,
             num_a: int = 4,
             Nc: int = 10,
             Nd: int = 10):
    """
    goal: ゴール画像（C, H ,W）
    observations: 観測（length, C, H, W）
    num_a: 方策モデルからサンプリングする行動系列の数
    """

    #行動生成
    T_a = policy.action_len
    T_p = policy.obs_len
    obs = observations.unsqueeze(0).repeat(num_a, 1, 1, 1, 1)
    policy.reset()
    
    condition = []
    actions = []
    for i in range(T_a):
        if i == 0:
            sample, noise, cond = policy.select_action(obs[:, i])
            cond = einops.rearrange(cond,
                                    "b (s d) -> b s d",
                                    b=cond.shape[0],
                                    s=T_p,
                                    d=cond.shape[-1]//T_p
                                    )
            print("generate")
            condition.append(cond)
        else:
            sample = policy.select_action(obs[:, i])
        
        actions.append(sample)

    condition = torch.stack(condition, dim=0).squeeze(0)
    actions = torch.stack(actions, dim=1)
    #直線速度の範囲を[-1,1]に戻す
    actions[:, :, 1] =  ((actions[:, :, 1]+1) / 2)*1.3 - 0.3
    
    actions_samples = actions
    
    #latent imgination ＆ EFE計算
    actions = world.action_transform(actions) #b, length, 32   

    efe_samples=[]
    imaginations_samples=[]
    with torch.no_grad():
        for s in range(num_a):
            samples = actions[s]
            efes, imaginations = calc_efe(states=current_states,
                                       ctx_actions=samples,
                                       world=world,
                                       preferred_obs=goal,
                                       pref_std=0.1,
                                       n_sample=Nc,
                                       n_sample_d=Nd)
            efe_samples.append(efes)
            # print(imaginations.shape) #length, 3, 60, 80
            imaginations_samples.append(imaginations)
            
    imaginations = torch.stack(imaginations_samples, dim=0) #num_a 
    efes = stack_efe(efe_samples, dim=0)
    # print(imaginations.shape) # num_a, length, c, h, w
    # print(efes.efe.shape) # num_a, length
    idx = torch.argmin(efes.efe)
    # imaginations = torch.permute(imaginations, (1,0,2,3,4))
    # efes = AttrDict(
    #     efe = efe,
    #     extrinsic = ext,
    #     epistemic = epi
    #     )
    
    return actions_samples, condition, imaginations, efes, idx
    


@torch.no_grad()
def efe_test(world: Union [WorldModel, CoarseWorldModel],
             data,
             policy_name: str = "base_64_mini",
             data_idx: int = 0,
             num_a: int = 4,
             goal_path: str = None,
             obs_init:bool = True,
             Nc: int =1,
             Nd: int=1,
             start: int = 0
             ):
    
    """
    data: observations(length, b, C, H, W)
            true_actions(length, b, dim)
    num_a: 方策モデルからサンプリングする行動系列の数
    goal_path: npファイルのpath
        無ければobservationのT_aステップ目
    obs_init: 世界モデルの状態初期化に観測画像を20ts入れる
    Nc: 世界モデルでo_hat作るのに各時刻で上階層posteriorからサンプリングするstochの数
    Nd: 世界モデルでo_hat作るのに各時刻で低階層priorからサンプリングするstochの数
    
    
    """
    coarse_ext = True


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_fix_seed(0)
    observations, true_actions = data
    # print(observations.shape)
    # print(true_actions.shape)
    observations = torch.permute(observations, (1,0,2,3,4))
    true_actions = torch.permute(true_actions, (1,0,2))
    
    observations = observations[data_idx]
    true_actions = true_actions[data_idx]
    observations = observations[start:]
    true_actions = true_actions[start:]

    #記録
    log_path = f"reports_efe/{policy_name}/{data_idx}_start:{start}"
    if coarse_ext:
        log_path = f"{log_path}_coarse"
    os.makedirs(log_path, exist_ok=True)
    
    #拡散方策
    policy_conf_path = f"models_d/cfg/{policy_name}.yaml"
    policy_cfg: DictConfig = load_config(policy_conf_path)
    policy_cfg = policy_cfg.model
    policy_cfg.noise_scheduler_type = "DDIM" #DDIMに変更
    
    T_f = policy_cfg.horizon
    T_a = policy_cfg.n_action_steps
    T_p =  policy_cfg.n_obs_steps
    
    policy = DiffusionPolicy(policy_cfg)
    p_path =f"models_d/params/{policy_name}"
    p_param = torch.load(f"{p_path}/latest.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(p_param, strict=False)
    policy.to(device)
    policy.eval()
    
    #ゴール画像
    if goal_path is not None:
        goal_np = np.load(goal_path, allow_pickle=True )
        goal = torch.tensor(goal_np)
    else:
        goal = observations[T_a]
        array2gif(save_path=f"{log_path}",
                  video_name="real_obs",
                  array= observations[:T_a].unsqueeze(0).detach().clone().cpu().numpy())
    goal = goal.to(device)
    if coarse_ext:
        world.to(device)
        goal_embed = world.obs_encoder(goal.unsqueeze(0))
        world.dynamics.init_latent(1, goal_embed.float())
        goal = world.dynamics.coarse_state.squeeze(0)
    
    #世界モデルの状態初期化
    world.to(device)
    with torch.no_grad():
        if obs_init:
            # obs_0 = F.interpolate(observations[0].unsqueeze(0).repeat(20, 1, 1, 1), size = [60, 80])
            obs_0 = observations[0].unsqueeze(0).repeat(20, 1, 1, 1)
            obs_0 = obs_0.unsqueeze(0)
            act_0 = torch.zeros(1, 20, 2)
            obs_0 = torch.permute(obs_0.to(device), (1,0,2,3,4)) #lengh, b, c, w, h
            act_0 = torch.permute(act_0.to(device), (1,0,2)) #length, b, dim
            
            act_0 = world.action_transform(act_0)  #length, 1, 32 
            
            world.dynamics._init_with_ = "posterior"
            embed_obs_0 = world.obs_encoder(obs_0)
            world.dynamics.init_latent(obs_0.shape[1], embed_obs_0[0].float())
            # history = []
            for i in range(act_0.shape[0]):
                current_states, _ = world.dynamics.step(act_0[i], embed_obs_0[i])
            
        else:
            obs_0 = F.interpolate(observations[0].unsqueeze(1), size=[60, 80])
            obs_0 =obs_0.unsqueeze(0)
            obs_0 = torch.permute(obs_0.to(device), (1,0,2,3,4)) 
            world.dynamics._init_with_ = "posterior"
            embed_obs_0 = world.obs_encoder(obs_0)
            world.dynamics.init_latent(obs_0.shape[1], embed_obs_0[0].float())
            act_0 = torch.zeros(20, 1, 2).to(device)
            act_0 = world.action_transform(act_0)
            current_states, _ = world.dynamics.step(act_0[0], obs_0[0])
    
    actions, condition, imaginations, efes, idx = action_select(
        world=world,
        policy=policy,
        current_states=current_states,
        goal=goal,
        observations=observations[:T_a],
        num_a=num_a,
        Nc=Nc,
        Nd=Nd)
    
    #efe
    efe = efes.efe
    extrinsic = efes.extrinsic
    epistemic = efes.epistemic
    print(efe)
    print(extrinsic)
    print(epistemic)
        
    #可視化
    #condition （spatial keypoints）
    array2gif(save_path=f"{log_path}/policy_cond",
                video_name="spatial",
                array=observations[:T_p].unsqueeze(0).detach().clone().cpu().numpy(),
                spatial=condition.detach().clone().cpu().numpy())
    #action
    actions = torch.permute(actions, (1,0,2))
    positions = actions_to_positions(actions)
    plot_posi_var(positions=positions.detach().clone().cpu(),
                  save_path=f"{log_path}",
                  fig_name="action_samples")
    
    #imagination
    array2gif(save_path=f"{log_path}/imagination",
              video_name="imagine",
              array=imaginations.detach().clone().cpu().numpy())
    
    #efe
    plot_efe_bar(efe=efe.detach().clone().cpu().numpy(),
                 epistemic=epistemic.detach().clone().cpu().numpy(),
                 extrinsic=extrinsic.detach().clone().cpu().numpy(),
                 save_path=f"{log_path}")
    

@torch.no_grad()
def eval_world(
        model_name: str,
        alpha: float,
        seed: int,
        device: int,
        load_last: bool = False,
        indices: list = None, 
        imagine: int =0,
        interval: int = None
        ):

    conf_path = f"models/cfg/{model_name}.yaml"

    config: DictConfig = load_config(conf_path)
    config.alpha = alpha
    config.seed = seed
    config.device = device

    torch_fix_seed(config.seed)

    if indices is None:
        indices = split_id(
                config.idx_splitter.num, 
                config.idx_splitter.change_point, 
                config.idx_splitter.n_val_each
                )
        indices = split_id(
                config.idx_splitter.num, 
                config.idx_splitter.change_point, 
                config.idx_splitter.n_val_each
                )
        indices = indices.validation
    else:
        indices = indices 
    data_cfg = load_config(f"data/{config.data_dir}/config.yaml")

    env_cfg = dict(
        obs_shape=data_cfg.obs_shape,
        action_dim=data_cfg.action_dim,
    )


    # print("envrionment:", env_cfg)
    config.update(env_cfg)
    OmegaConf.resolve(config)
    config: ExperimentConfig = instantiate(
        config,
    )
    if imagine > 0:
        log_path = f"reports/{config.data_dir}_{model_name}/alpha:{config.alpha}/seed:{config.seed}/imagine{imagine}_test"
        if interval is not None:
            log_path += f"_interval{interval}"
    else:
        log_path = f"reports/{config.data_dir}_{model_name}/alpha:{config.alpha}/seed:{config.seed}/world_test"
    path = f"models/params/{model_name}/alpha:{config.alpha}/seed:{config.seed}/"
    os.makedirs(log_path, exist_ok=True)
    
    with open(f'data/{config.data_dir}/config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    data_config = OmegaConf.create(data_config)

    *data, is_split = setup_data(
        config.data_dir,
        config.datamodule.action_diff,
        data_config.gripper_dim
    )
    
    if is_split:
        print("is_split")
        action_train, image_train, action_states, image_states  = data
        image_states = image_states.float()
    else:
        action_states = action_states[:, indices]
        image_states = image_states[:, indices].float()
    image_states = scale_obs(image_states)

    
    print("image_states", image_states.shape, image_states.min(), image_states.max())
    
    # # 必要な範囲をスライス
    # image_states_subset = torch.zeros((220, 2, *image_states.shape[2:]), device=image_states.device)
    # image_states_subset[:, 0] = image_states[0:220, 0]  # バッチ 1
    # image_states_subset[:, 1] = image_states[0:220, 1]  # バッチ 2

    # image_states_subset2 = torch.zeros((220, 2, *image_states.shape[2:]), device=image_states.device)
    # image_states_subset2[:, 0] = image_states[280:500, 2]  # バッチ 3
    # image_states_subset2[:, 1] = image_states[200:420, 3]  # バッチ 4

    # # 連結して b=4 に戻す
    # image_states= torch.cat([image_states_subset, image_states_subset2], dim=1)

    # # action_states も同様に処理
    # action_states_subset = torch.zeros((220, 2, action_states.shape[2]), device=action_states.device)
    # action_states_subset[:, 0] = action_states[0:220, 0]
    # action_states_subset[:, 1] = action_states[0:220, 1]

    # action_states_subset2 = torch.zeros((220, 2, action_states.shape[2]), device=action_states.device)
    # action_states_subset2[:, 0] = action_states[280:500, 2]
    # action_states_subset2[:, 1] = action_states[200:420, 3]

    # action_states= torch.cat([action_states_subset, action_states_subset2], dim=1)
    
    if isinstance(config.world.dynamics_cfg, CRSSMV4Config) or isinstance(config.world.dynamics_cfg, CRSSMConfig):
        world = CoarseWorldModel(config.world)
    else:
        world = WorldModel(config.world)

    if load_last:
        param = torch.load(f"{path}/last.ckpt", map_location="cpu")["state_dict"]
    else:
        param = torch.load(f"{path}/{world.__class__.__name__}.ckpt", map_location="cpu")["state_dict"]
    
    print("model name:", f"{world.__class__.__name__}")
    
    for k in list(param.keys()):
        if "codebook.mask" in k:
            del param[k]
    
    if "action_transform.k" in param:
        param.pop("action_transform.k")
    world.load_state_dict(param)
    world.eval()
    
    
    # all_obs, all_act = get_all_data(data_dir="/home/yokozawa/work/Turtlebot/data/",
    #                                 obs_size=[60,80])

    
    # all_obs, all_act = get_all_data(data_dir="/home/yokozawa/work/l_turtlebot4_ws/turtlebot4_ws/dataset/260516",
    #                                 obs_size=[60,80])
    # # 
    # print(all_obs.shape, all_obs.min().item(), all_obs.max().item())
    # print(all_act.shape, all_act.min().item(), all_act.max().item())
    
    # embed_obs = world.obs_encoder(all_obs)
    # all_action = world.action_transform(all_act)
    
    # print(all_action.shape, all_action.min().item(), all_action.max().item())
    
    # image_states = all_obs[:200, 8:12].float()
    # action_states = all_act[:200, 8:12].float()
    
    # # np.save("/home/yokozawa/work/Turtlebot/data/turtle-mini600/embed_all.npy", embed_obs.detach().cpu().numpy())
    
    # world.dynamics.init_latent(embed_obs.shape[1], embed_obs[0])
    
    # states, _ = world.dynamics.forward(all_action.float(), embed_obs.float())
    # coarse_all = states.coarse
    # determ_all = states.determ
    # coarse_all = coarse_all[:1000]
    
    # chunk_size = 2000
    # total_len = embed_obs.shape[0]
    # coarses = []
    # actions = []

    # # チャンクごとに処理
    # for i in range(0, total_len, chunk_size):
    #     # チャンクを取得
    #     obs_chunk = embed_obs[i:i+chunk_size]
    #     act_chunk = all_action[i:i+chunk_size]

    #     # チャンクの長さが短い可能性に注意
    #     if obs_chunk.shape[0] == 0:
    #         continue

    #     # init_latent をこのチャンク用に呼ぶ
    #     world.dynamics.init_latent(obs_chunk.shape[1], obs_chunk[0])

    #     # forward に通す
    #     states, _ = world.dynamics.forward(act_chunk.float(), obs_chunk.float())
    #     coarse_chunk = states.coarse

    #     coarses.append(coarse_chunk)
    #     actions.append(all_act[i:i+chunk_size])
    # # 結合（チャンク×バッチ を1つの大きなバッチに）
    # coarses = torch.cat(coarses, dim=1)   # shape: [200, 160, D]
    # actions = torch.cat(actions, dim=1)   # shape: [200, 160, A]
    # print(coarses.shape)
    # print(actions.shape)
    # # 保存
    # save_dir = "/home/yokozawa/work/l_turtlebot4_ws/turtlebot4_ws/dataset/2000ts_coarses"
    # np.save(f"{save_dir}/coarses.npy", coarses.detach().cpu().numpy())
    # np.save(f"{save_dir}/actions.npy", actions.detach().cpu().numpy())

    # print("保存完了")
    # raise KeyboardInterrupt
    
    
    # if coarse_all.shape[-1] == 2:
    #     plot_2d(coarse_all[:, 2:3], coarse_all, f"{log_path}", f"coarses")
    # a_list = [1, 6, 9, 14, 17, 31]
    # b_list = [3, 5, 10, 13, 18, 20, 22, 24, 28, 29] 
    # for a in a_list:
    #     for b in b_list:
    #         c = torch.cat([coarse_all[:,:, a-1:a], coarse_all[:,:, b-1:b]], dim=-1)
    #         # print(c.shape)
    #         plot_2d(c[:, 2:3], c, f"{log_path}/coarse_dim_pick", f"dim_{a}_{b}")    
    # raise KeyboardInterrupt
    # determ_all = states.determ
    # print(determ_all.shape)
    # np.save("/home/yokozawa/work/Turtlebot/data/turtle-mini600/determ_all.npy", determ_all.detach().cpu().numpy())
    # print(coarse_all[0,0])
    # np.save("/home/yokozawa/work/Turtlebot_b4/data/260516_mini/coarse_all.npy", coarse_all.detach().cpu().numpy())
    # np.save("/home/yokozawa/work/Turtlebot_b4/data/turtle-mini600/coarse_all_sorted.npy", coarse_all.detach().cpu().numpy())
    
    
    
    # raise KeyboardInterrupt
    
    
    # print(coarse_all.shape)
    
    # decomposite_feature_2(coarse_all,
    #                       "pca",
    #                       "all",
    #                       f"{log_path}",
    #                       "pca_all_r")
    # decomposite_feature_2(determ_all,
    #                       "pca",
    #                       "all",
    #                       f"{log_path}",
    #                       "determ_umap_all")
    # decomposite_feature_2(embed_obs,
    #                       "pca",
    #                       "all",
    #                       f"{log_path}",
    #                       "embed_umap_all")
    # decomposite_feature_2(embed_obs,
    #                       "tsne",
    #                       "all",
    #                       f"{log_path}",
    #                       "embed_tsne_all")
    # 
    # decomposite_feature(f"{log_path}", "determ_all_3d_r", torch.permute(determ_all, (1,0,2)), cmap="cool", decompose_ndim=3, scatter_mark = "indices")
    # decomposite_feature(f"{log_path}", "_coarse_all_3d_r", torch.permute(coarse_all, (1,0,2)), cmap="cool", decompose_ndim=3, scatter_mark = "indices")
    # decomposite_feature(f"{log_path}", "embed_all_3d_r", torch.permute(embed_obs, (1,0,2))[:, :1000], cmap="cool", decompose_ndim=3, scatter_mark = "indices")
    # decomposite_feature(f"{log_path}", "embed_all_3d", torch.permute(embed_obs, (1,0,2)), cmap="turbo", how2decompose="t-SNE", decompose_ndim=3, scatter_mark = "indices")
    # raise KeyboardInterrupt
    # world.dynamics.init_latent(embed_obs.shape[0], embed_obs[:,2:3])
    # coarse_init = world.dynamics.coarse_state
    # print(coarse_init.shape)
    # pca = PCA(n_components=3)
    # coarse_all = coarse_all[:1000].reshape(-1, 32)
    # c_pca = pca.fit_transform(coarse_all)
    # c_pca = c_pca.reshape(1000, 16, 3).transpose(1, 0, 2)
    # print(c_pca.shape)
    # coarse_init = pca.transform(coarse_init[:1000]).reshape(1, 1000, 3)
    # print(coarse_init.shape)
    # plot_3d(coarse_init, c_pca, pca.explained_variance_ratio_, f"{log_path}", "coarse_init_3d")
    
    
    # print(coarse_all.shape)
    # coarse_all = coarse_all.reshape([-1, coarse_all.shape[-1]])
    # print(coarse_all.shape)
        
    # all_path = os.path.join(f"/home/yokozawa/work/Turtlebot_b4/models/params/{model_name}", 'all_coarse.npy')
    # np.save(all_path, coarse_all.detach().clone().cpu().numpy())
    
    
    # efe_test(world=world,
    #          data=[image_states, action_states],
    #          data_idx = 2,
    #          start = 300
    #          )
    
    # coarse_all = np.load(f"/home/murata-lab/Turtlebot-1/models/params/{world_name}/all_coarse.npy")
    # print(coarse_all.shape)
    # pca_base = PCA(n_components=3)
    # pca_base.fit(coarse_all)
    
    
    # inference
    if config.world.use_vqvae:
        image_codes, code_indices = vqvae.quantize(image_states, True, world.cfg.binarize_code)
        print("code_indices", code_indices.shape, code_indices.min(), code_indices.max())
        embed_obs = world.obs_encoder(image_codes)
    else:
        embed_obs = world.obs_encoder(image_states)
    
    world.dynamics.init_latent(embed_obs.shape[1], embed_obs[0])
    print("embed_obs", embed_obs.shape, embed_obs.min(), embed_obs.max())

    action_states = world.action_transform(action_states)
    interval = interval if interval is not None else len(action_states)
    if imagine > 0:
        world_states = []
        zeros = torch.zeros(action_states.shape[1], 2)
        zeros = world.action_transform(zeros)
        for t in range(len(action_states)+imagine):
            if t < imagine :#or t % interval < 5:
                print("get first observation", t)
                world_state, _ = world.dynamics.step(zeros, embed_obs[0])
            else:
                world_state, _ = world.dynamics.step(action_states[t - imagine])
                world_states.append(world_state)
        world_states = stack_worlds(world_states)

    else:
        world_states, _ = world.dynamics(action_states, embed_obs)

    latent_states = world_states.latent_states

    predicted_obs, predicted_embed_obs = world._decode_obs(latent_states, embed_obs)

    if isinstance(world, CoarseWorldModel) or isinstance(world, CoarseWorldModel):
        coarse_latent_states = world_states.coarse_states
        coarse_predicted_obs, _  = world.decode_coarse_obs(coarse_latent_states, world_states.posterior.stoch)
    
    if config.world.use_vqvae:
        print("accuracy", code_indices.shape, predicted_obs.shape)
        if world.cfg.binarize_code:
            code_indices = code_indices.float().reshape_as(predicted_obs)
            predicted_codes = torch.where(predicted_obs > 0.5, 1., 0.)
            accuracy = (code_indices == predicted_codes).float()
        else:
            accuracy = (code_indices == predicted_obs.argmax(dim=-1)).float().sum(dim=-1)/code_indices.shape[-1]
        print("accuracy", accuracy.mean())
        predicted_obs = vqvae.project(predicted_obs)
        predicted_obs = vqvae.obs_decoder(predicted_obs)
        print("coarse_accuracy", code_indices.shape, coarse_predicted_obs.shape)
        if world.cfg.binarize_code:
            code_indices = code_indices.float().reshape_as(coarse_predicted_obs)
            coarse_predicted_codes = torch.where(coarse_predicted_obs > 0.5, 1., 0.)
            coarse_accuracy = (code_indices == coarse_predicted_codes).float()
        else:
            coarse_accuracy = (code_indices == coarse_predicted_obs.argmax(dim=-1)).float().sum(dim=-1)/code_indices.shape[-1]
        print("coarse_accuracy", coarse_accuracy.mean())
        coarse_predicted_obs = vqvae.project(coarse_predicted_obs)
        coarse_predicted_obs = vqvae.obs_decoder(coarse_predicted_obs)

    
    # cl = torch.permute(world_states.coarse, (1,0,2)).reshape(-1, world_states.coarse.shape[-1])
    # all_path = os.path.join(f"/home/yokozawa/work/Turtlebot/models/params/{model_name}", '4_coarse.npy')
    # np.save(all_path, cl.detach().clone().cpu().numpy())
    
    # pca = PCA(n_components=3)
    # _ = pca.fit_transform(torch.permute(world_states.coarse, (1,0,2)).reshape(-1, world_states.coarse.shape[-1]))
    # explain_ratio = pca.explained_variance_ratio_
    # coarse_all_pca = pca.transform(torch.permute(coarse_all, (1,0,2)).reshape(-1, coarse_all.shape[-1]))
    # plot_3d(coarse_all_pca.reshape(16, 2000, 3), explain_ratio, f"{log_path}", "3d_pca_all")
    
    
    # decomposite_feature(f"{log_path}/coarse", "coarse_2", torch.permute(world_states.coarse, (1,0,2)), cmap="turbo", decompose_ndim=3)
    

    # visualize
    print("video")
    os.makedirs(f"{log_path}", exist_ok=True)
    image_states = image_states.transpose(1, 0).detach().cpu().numpy()
    image_states = unscale_obs(image_states) / 255
    embed_obs = embed_obs.transpose(1, 0).detach().cpu().numpy()
    predicted_obs = predicted_obs.transpose(1, 0).detach().cpu().numpy()
    predicted_obs = unscale_obs(predicted_obs) / 255
    print("predicted_obs", predicted_obs.shape, predicted_obs.min(), predicted_obs.max())
    if predicted_embed_obs is not None:
        predicted_embed_obs = predicted_embed_obs.transpose(1, 0).detach().cpu().numpy()
    if isinstance(world, CoarseWorldModel) or isinstance(world, CoarseWorldModel):
        if coarse_predicted_obs is not None:
            coarse_predicted_obs = coarse_predicted_obs.transpose(1, 0).detach().cpu().numpy()
            coarse_predicted_obs = unscale_obs(coarse_predicted_obs) / 255
            print("coarse_predicted_obs", coarse_predicted_obs.shape, coarse_predicted_obs.min(), coarse_predicted_obs.max())
    world_states = world_states.transpose(1, 0).numpy()

    if world.alpha > 0:
        # print(predicted_obs.shape)
        # print(image_states.shape)
        # save_selected_frames_as_pdf(f"{log_path}/imagine_long", predicted_obs, image_states)
        # raise KeyboardInterrupt

        visualize_comparing_video(f"{log_path}/video", "precise",
                                  predicted_obs, image_states)
        if isinstance(world, CoarseWorldModel) or isinstance(world, CoarseWorldModel):
            if coarse_predicted_obs is not None:
                visualize_comparing_video(f"{log_path}/video", "coarse",
                                      coarse_predicted_obs, image_states)
    if world.spatial:
        array2gif(f"{log_path}/video", "spatial",
                  image_states, embed_obs)
    else:
        print(embed_obs.shape) #b, l, d
        # decomposite_feature(
        #     save_path=f"{log_path}/embed", 
        #     fig_name="embed",
        #     features=embed_obs, 
        #     cmap="turbo"
        # )
        # visualise_continuous(embed_obs, f"{log_path}/embed",
        #                      fig_name="embed", cmap="coolwarm")
    # plt.close("all")
    print("states")
    try:
        print("deter", world_states.determ.shape)
        # decomposite_feature(f"{log_path}/deter", "deter", world_states.determ, cmap="turbo", decompose_ndim=3)
        # decomposite_feature(f"{log_path}/deter", "deter", world_states.determ, cmap="turbo", how2decompose="UMAP", decompose_ndim=3)
        # decomposite_feature(f"{log_path}/deter", "deter", world_states.determ, cmap="turbo", how2decompose="t-SNE", decompose_ndim=3)
        # visualise_continuous(world_states.determ, f"{log_path}/deter",
        #                      fig_name="deter", cmap="coolwarm")
        if isinstance(world, CoarseWorldModel) or isinstance(world, CoarseWorldModel):
            print("coarse", world_states.coarse.shape)
            # decomposite_feature(f"{log_path}/coarse", "coarse_2", world_states.coarse[:2], cmap="turbo", decompose_ndim=3)
            # decomposite_feature(f"{log_path}/coarse", "coarse", world_states.coarse, cmap="turbo", how2decompose="UMAP", decompose_ndim=3)
            # decomposite_feature(f"{log_path}/coarse", "coarse", world_states.coarse, cmap="turbo", how2decompose="t-SNE", decompose_ndim=3)
            # visualise_continuous(world_states.coarse, f"{log_path}/coarse",
            #                      fig_name="coarse", cmap="coolwarm")
    except KeyboardInterrupt:
        pass
    plt.close("all")

    print("prior", world_states.prior.shape)
    visualize_feature(f"{log_path}/prior", world_states.prior,
                      stoch_type=world.dynamics.d_prior.dist, fig_name="prior")
    print("posterior", world_states.posterior.shape)
    visualize_feature(f"{log_path}/posterior", world_states.posterior,
                      stoch_type=world.dynamics.d_posterior.dist, fig_name="posterior")
    if isinstance(world, CoarseWorldModel) or isinstance(world, CoarseWorldModel):

        print("c_prior", world_states.c_prior.shape)
        visualize_feature(f"{log_path}/c_prior", world_states.c_prior,
                          stoch_type=world.dynamics.c_prior.dist, fig_name="c_prior")
        print("c_posterior", world_states.c_posterior.shape)
        visualize_feature(f"{log_path}/c_posterior", world_states.c_posterior,
                          stoch_type=world.dynamics.c_posterior.dist, fig_name="c_posterior")
    plt.close("all")

