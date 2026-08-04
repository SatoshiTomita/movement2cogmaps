import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.loss import LossFunctions
from architectures.losses_custom import get_hidden_l2norm, get_weights_l2norm, rssm_kl_loss
from architectures.training import Trainer


def _clone_recurrent_state(state):
    """Clone a tensor state or every component of an LSTM state tuple."""
    if isinstance(state, tuple):
        return tuple(component.clone() for component in state)
    return state.clone()


def _detach_recurrent_state(state):
    """Detach a tensor state or every component of an LSTM state tuple."""
    if isinstance(state, tuple):
        return tuple(component.detach() for component in state)
    return state.detach()


class TrainerBPTT(Trainer):

    def __init__(self, args, optimizer, loss_fn, device):
        super().__init__(optimizer, loss_fn, device)
        self.args = args

    def train_epoch(self, model, dataloader):
        """
        学習ループ
        """
        model.train()
    
        return_dict = {}
        hidden_last = None

        for i, data in enumerate(dataloader):
            # zero your gradients for every batch
            self.optimizer.zero_grad()

            # scene:時間窓のシーン画像, vel:時間窓の線形速度, rot_vel:時間窓の角速度, pos:時間窓の位置座標, thet:時間窓の方角, labels:時間窓の次ステップの画像
            scene, vel, rot_vel, _, _, labels = data
            scene = scene.squeeze(dim=0).to(self.device)

            outputs_all = [] # here we collect all the outputs from the rnn
            for f in range(self.args.n_future_pred):
                inputs = torch.cat((
                    scene,
                    vel.squeeze(dim=0)[:, f, ...].to(self.device),
                    rot_vel.squeeze(dim=0)[:, f, ...].to(self.device)
                ), dim=-1)

                # 最初の予測の場合、前の隠れ状態を利用して予測する
                if f == 0:
                    outputs, hidden_all, hidden_last = model(inputs, hidden_last)
                    h = _clone_recurrent_state(hidden_last) if self.args.n_future_pred>1 else None
                # それ以降の予測では、前の隠れ状態を利用して予測する
                else:
                    outputs, _, h = model(inputs, h)
                outputs_all.append(outputs)
                scene = outputs

            # 未来予測を1つのテンソルにまとめる
            outputs_all = torch.stack(outputs_all, dim=1)
            labels = labels.squeeze(dim=0).to(self.device)

            # モデルの予測と正解のラベルを比較する
            loss = self.loss_fn(outputs_all, labels)
            
            # 隠れ状態の正則化
            hidden_l2norm = get_hidden_l2norm(hidden_all)
            hidden_reg_loss = self.args.hidden_reg * hidden_l2norm
            # 重みの正則化
            weights_l2norm = get_weights_l2norm(model)
            weights_reg_loss = self.args.weights_reg * weights_l2norm

            return_dict = self._update_losses(
                ['loss_train', 'hidden_reg_loss_train', 'weights_reg_loss_train', 'tot_loss_train'],
                [loss, hidden_reg_loss, weights_reg_loss, loss+hidden_reg_loss+weights_reg_loss],
                return_dict
            )
            # 正則化項を損失に加える
            if self.args.hidden_reg > 0:
                loss += hidden_reg_loss
            if self.args.weights_reg > 0:
                loss += weights_reg_loss

            # calculate gradients and clip values (if applicable)
            loss.backward()
            # 勾配クリッピング
            if self.args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_value)

            # 重みを更新
            self.optimizer.step()

            # update dict with additional model information
            return_dict = self._update_hidden_layer(
                ['hidden_l2norm'], [hidden_l2norm], return_dict
            )
            return_dict = self._update_norms(model, return_dict)

            hidden_last = _detach_recurrent_state(hidden_last)
            # reset_hidden_atが指定されている場合、指定されたステップごとに隠れ状態をリセットする
            if self.args.reset_hidden_at is not None and i % self.args.reset_hidden_at == 0:
                hidden_last = None
        
        # take the average over all batches and return the dictionary
        return_dict = {k: v/(i+1) for k, v in return_dict.items()}
        return model, return_dict
    

    def test_epoch(self, model, dataloader, for_trajectory=False):
        """Run one evaluation epoch.

        Args:
            model: The RNN model.
            dataloader: DataLoader yielding (scene, vel, rot_vel, pos, theta, labels).
            for_trajectory: If True, also collect hidden activations, positions, and thetas.

        Returns:
            If for_trajectory is False: dict of averaged test metrics.
            If True: (metrics_dict, hidden_activity, positions, thetas,
                       loss_list, loss_wrt_input_list, distance_input_list).
        """
        model.eval()
        loss_list = []
        loss_wrt_input_list = []
        distance_input_list = []

        with torch.no_grad():
            return_dict = {}
            hidden_last = None

            if for_trajectory:
                hidden_activity, positions, thetas = [], [], []
            
            for i, tdata in enumerate(dataloader):
                scene, vel, rot_vel, pos, thet, labels = tdata
                inputs = torch.cat(
                    (scene.squeeze(dim=0), vel.squeeze(dim=0)[:, 0, ...], rot_vel.squeeze(dim=0)[:, 0, ...]),
                    dim=-1
                ).to(self.device)
                labels = labels.squeeze(dim=0)[:, 0, ...].to(self.device)

                outputs, hidden_all, hidden_last = model(inputs, hidden_last)

                loss = self.loss_fn(outputs, labels)
                loss_wrt_input = self.loss_fn(outputs, scene.squeeze(dim=0).to(self.device))
                distance_input = self.loss_fn(labels, scene.squeeze(dim=0).to(self.device))

                loss_list.append(loss.detach().item())
                loss_wrt_input_list.append(loss_wrt_input.detach().item())
                distance_input_list.append(distance_input.detach().item())

                # compute the hidden state and weights l2 norm
                hidden_l2norm = get_hidden_l2norm(hidden_all)
                hidden_reg_loss = self.args.hidden_reg * hidden_l2norm

                weights_l2norm = get_weights_l2norm(model)
                weights_reg_loss = self.args.weights_reg * weights_l2norm

                return_dict = self._update_losses(
                    ['loss_test', 'loss_wrt_input', 'distance_input', 'hidden_reg_loss_test', 'weights_reg_loss_test', 'tot_loss_test'],
                    [loss, loss_wrt_input, distance_input, hidden_reg_loss, weights_reg_loss, loss+hidden_reg_loss+weights_reg_loss],
                    return_dict
                )
                if self.args.hidden_reg > 0:
                    loss += hidden_reg_loss
                if self.args.weights_reg > 0:
                    loss += weights_reg_loss
            
                # update dict with additional model information
                return_dict = self._update_hidden_layer(
                    ['hidden_l2norm'], [hidden_l2norm], return_dict
                )
                return_dict = self._update_norms(model, return_dict)

                hidden_last = _detach_recurrent_state(hidden_last)
                if self.args.reset_hidden_at is not None and i % self.args.reset_hidden_at == 0:
                    hidden_last = None

                if for_trajectory:
                    hidden_activity.append(hidden_all.detach().cpu().numpy())
                    positions.append(pos.squeeze(dim=0)[:, 0, ...].cpu().numpy())
                    thetas.append(thet.squeeze(dim=0)[:, 0, ...].cpu().numpy())
            
            return_dict = {k: v/(i+1) for k, v in return_dict.items()}
        
        if for_trajectory:
            return return_dict, hidden_activity, positions, thetas, loss_list, loss_wrt_input_list, distance_input_list
        return return_dict


    def plot_test_examples(self, model, dataloader, n_figures, n_examples, frame_dim,
                           truncate_scene=None, ae=None):
        """Generate side-by-side plots of input / output / label frames.

        Args:
            model: The RNN model.
            dataloader: Test DataLoader.
            n_figures: Number of figures to produce (evenly sampled across batches).
            n_examples: Total number of example rows across all figures.
            frame_dim: (width, height) of the scene frame for reshaping.
            truncate_scene: Optional channel truncation for the scene tensor.
            ae: Optional autoencoder for decoding latent scenes to images.

        Returns:
            List of matplotlib Figure objects.
        """
        with torch.no_grad():
            ex_per_figure = int(np.ceil(n_examples / n_figures))
            sampled_indices = np.linspace(0, len(dataloader)-1, n_figures, dtype=int)
            
            hidden_last = None
            figs = []

            for i, tdata in enumerate(dataloader):
                scene, vel, rot_vel, _, _, labels = tdata
                inputs = torch.cat(
                    (scene.squeeze(dim=0), vel.squeeze(dim=0)[:, 0, ...], rot_vel.squeeze(dim=0)[:, 0, ...]),
                    dim=-1
                ).to(self.device)
                labels = labels.squeeze(dim=0)[:, 0, ...]

                outputs, _, hidden_last = model(inputs, hidden_last)
                hidden_last = _detach_recurrent_state(hidden_last)
                if self.args.reset_hidden_at is not None and i % self.args.reset_hidden_at == 0:
                    hidden_last = None

                if i in sampled_indices:
                    for scene_idx in [0, -1]:
                        fig, axs = plt.subplots(ex_per_figure, 3, figsize=(7, ex_per_figure*2.5))
                        fig.suptitle(f"Example batch {i}, scene {scene_idx}")
                        for idx in range(outputs.shape[1]-1, outputs.shape[1]-ex_per_figure-1, -1):
                            ax_idx = outputs.shape[1]-idx-1

                            inputs_plot = inputs[..., :outputs.shape[-1]]
                            inputs_plot = inputs_plot[..., :truncate_scene] if truncate_scene is not None else inputs_plot
                            inputs_plot = (
                                inputs_plot[scene_idx, idx, :].cpu().reshape(*frame_dim[::-1]) if ae is None else
                                ae.decode(inputs_plot[scene_idx, idx, :].unsqueeze(0)).cpu().detach().numpy().squeeze()
                            )
                            axs[ax_idx, 0].set_title(f'Input {idx}', fontsize=14)
                            axs[ax_idx, 0].imshow(inputs_plot, cmap='gray')
                            axs[ax_idx, 0].set_axis_off()

                            outputs_plot = outputs[..., :truncate_scene] if truncate_scene is not None else outputs
                            outputs_plot = (
                                outputs_plot[scene_idx, idx, :].cpu().reshape(*frame_dim[::-1]) if ae is None else
                                ae.decode(outputs_plot[scene_idx, idx, :].unsqueeze(0)).cpu().detach().numpy().squeeze()
                            )
                            axs[ax_idx, 1].set_title(f'Output {idx}', fontsize=14)
                            axs[ax_idx, 1].imshow(outputs_plot, cmap='gray')
                            axs[ax_idx, 1].set_axis_off()

                            labels_plot = labels[..., :truncate_scene] if truncate_scene is not None else labels
                            labels_plot = (
                                labels_plot[scene_idx, idx, :].reshape(*frame_dim[::-1]) if ae is None else
                                ae.decode(labels_plot[scene_idx, idx, :].unsqueeze(0).to(self.device)).cpu().detach().numpy().squeeze()
                            )
                            axs[ax_idx, 2].set_title(f'Label {idx}', fontsize=14)
                            axs[ax_idx, 2].imshow(labels_plot, cmap='gray')
                            axs[ax_idx, 2].set_axis_off()
                        plt.axis('off')
                        plt.tight_layout()
                        figs.append(fig)
        return figs


class TrainerRSSM(TrainerBPTT):
    """Trainer for the RSSM model.

    Reuses the BPTT window/hidden-carryover machinery of ``TrainerBPTT`` but
    adds the KL divergence between the RSSM prior and posterior to the
    reconstruction loss. The recurrent state is a packed ``concat(determ, stoch)``
    tensor, so it is carried across windows exactly like the plain RNN hidden
    state. Expects ``args.kl_scale`` and ``args.free_nats`` to be set.

    Args:
        args: Namespace with training hyperparameters (adds kl_scale, free_nats).
        optimizer: Torch optimizer.
        loss_fn: Reconstruction loss function.
        device: Torch device.
    """

    def train_epoch(self, model, dataloader):
        """Run one training epoch, adding the RSSM KL term to the loss."""
        model.train()

        return_dict = {}
        hidden_last = None

        for i, data in enumerate(dataloader):
            self.optimizer.zero_grad()

            scene, vel, rot_vel, _, _, labels = data
            # [1,B,T,O]->[B,T,O]
            scene = scene.squeeze(dim=0).to(self.device)

            # [1,B,F,T,D]->[B,T,D]
            velocity=(
                vel.squeeze(dim=0)[:,0].to(self.device)
            )

            rotational_velocity=(
                rot_vel.squeeze(dim=0)[:,0].to(self.device)
            )

            # RSSMへ渡す行動(行動をconcat)
            # [B,T,velocity_dim+rotational_velocity_dim]
            action=torch.vat(
                [
                    velocity,
                    rotational_velocity,
                ],
                dim=1,
            )

            observation=(
                labels.squeeze(dim=0)[:,0].to(self.device)
            )

            initial_obs=(
                scene[:,0]
                if hidden_last is None
                else None
            )

            outputs,hidden,hidden_last=model.observe(
                action=action,
                observation=observation,
                state=hidden_last,
                initial_obs=initial_obs,
            )

            print("========")
            print("action:",action.shape)
            print("observation:",observation.shape)
            print("outputs:",outputs.shape)
            print("hidden_all:",hidden_all.shape)
            print("hidden_last:",hidden_last.shape)
            print("========")

            

            
