import numpy as np
import torch
import tqdm
from sklearn.cluster import KMeans

from .abstract_trainer import AbstractTrainer

# from .kmeans_utils import KMeans
from .snl_loop import SNLTrainer
from .trainer_utils import update_dic


class EMSNL(SNLTrainer):
    """
    SNL Trainer class, inherits from AbstractTrainer.

    Args:
    =====
    energy: Energy
        Energy model.

    proposal: Proposal
        Proposal model.

    dataloader: torch.utils.data.DataLoader
        Training dataloader.

    """

    def __init__(
        self,
        energy,
        proposal,
        dataloader,
        val_dataloader,
        n_sample_train=1000,
        n_sample_test=10_000,
        stochastic_em=False,
        lr=1e-4,
    ) -> None:
        super().__init__(
            energy=energy,
            proposal=proposal,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            n_sample_train=n_sample_train,
            n_sample_test=n_sample_test,
            lr=lr,
            name_wandb=None,
        )

        self.stochastic_em = stochastic_em

    def get_optimizer(self, optimizer, lr, weight_decay):
        self.energy.logit_pi.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.energy.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.optimizer_explicit_bias = torch.optim.Adam(
            [self.energy.explicit_bias], lr=lr, weight_decay=weight_decay
        )
        # TODO : Check that this is correct

    def get_approximate_normalisation_constant_per_cluster(
        self,
        n_sample: int,
        batch_sample: int = 1e4,
        energy=None,
    ) -> torch.Tensor:
        """
        Get the approximate normalisation constant per cluster.
        """
        if energy is None:
            energy = self.energy
        remaining_sample = n_sample
        current_batch = min(remaining_sample, batch_sample)

        x_sample = self.proposal.sample(current_batch)
        remaining_sample -= current_batch

        energy_proposal = energy.energy_per_cluster(
            x_sample,
        )

        proposal_log_prob = (
            torch.from_numpy(self.proposal.log_prob(x_sample).detach().cpu().numpy())
            .reshape((current_batch, 1))
            .to(self.device)
        )

        log_normalisation_constant = (
            -energy_proposal
            - energy.explicit_bias
            - proposal_log_prob
            - torch.tensor(n_sample).log()
        ).logsumexp(dim=0, keepdim=True)

        while remaining_sample > 0:
            print(remaining_sample)
            current_batch = min(remaining_sample, batch_sample)
            x_sample = self.proposal.sample(current_batch)
            remaining_sample -= batch_sample

            energy_proposal = energy.energy_per_cluster(x_sample).reshape(
                (current_batch, self.energy.num_cluster)
            )

            proposal_log_prob = torch.from_numpy(
                self.proposal.log_prob(x_sample).detach().cpu().numpy()
            ).reshape((current_batch, 1))

            log_normalisation_constant_aux = (
                -energy_proposal
                - energy.explicit_bias
                - proposal_log_prob
                - torch.tensor(n_sample).log()
            ).logsumexp(dim=0, keepdim=True)

            log_normalisation_constant = torch.logsumexp(
                torch.cat([log_normalisation_constant, log_normalisation_constant_aux]),
                dim=0,
                keepdim=True,
            )

        return log_normalisation_constant

    def snl_forward(
        self,
        x: torch.Tensor,
        attribution: torch.Tensor,
        n_sample: int,
    ) -> torch.Tensor:
        """
        Forward pass for the SNL, in which case the different normalisation constant are computed
        per cluster.
        """
        loss_target = None
        loss_proposal = None
        dic = {}

        energy_target = self.energy(x=x, add_explicit_bias=True, per_cluster=True)

        approx_norm_constant = self.get_approximate_normalisation_constant_per_cluster(
            n_sample=n_sample,
        )

        if self.stochastic_em:
            # print("Stochastic EM")
            energy_target = (
                energy_target * attribution
            )  # If int mask, directly put to 0 the energy target, as it does not appear
            avg_attrib = attribution.mean(
                dim=0, keepdim=True
            )  # in this case, the average attribution is the 1/n \sum_i p(z_k|x_i)
        else:
            # print("Deterministic EM")
            energy_target = energy_target + torch.log(
                attribution + 1e-8
            )  # If proba, we can put the energy in as E_i * p(z_k|x_i), the - comes later in the loss
            avg_attrib = attribution.mean(
                dim=0, keepdim=True
            )  # in this case, the average attribution is the 1/n \sum_i p(z_k|x_i)

        loss_target = energy_target.mean(dim=0, keepdim=False)  # +> 1/n \sum_i E_i

        dic.update(
            {
                f"energy_no_bias_{k}": loss_target[k]
                for k in range(self.energy.num_cluster)
            }
        )

        loss_proposal = ((approx_norm_constant.exp() - 1) * avg_attrib).reshape(
            (self.energy.num_cluster,)
        )

        dic.update(
            {
                f"normalisation_diff_{k}": loss_proposal[k]
                for k in range(self.energy.num_cluster)
            }
        )

        dic.update(
            {
                f"loss_total_{k}": loss_target[k] + loss_proposal[k]
                for k in range(self.energy.num_cluster)
            }
        )

        log_likelihood_lower_bound = -loss_target - loss_proposal
        log_likelihood_upper_bound = -loss_target - approx_norm_constant * avg_attrib

        loss_total = (loss_target + loss_proposal).sum()
        dic.update(
            {
                "loss": loss_total,
                "energy_no_bias": loss_target.sum(),
                "diff_norm": loss_proposal.sum(),
                "log_likelihood_lower_bound (SNL)": log_likelihood_lower_bound.sum(),
                "log_likelihood_upper_bound": log_likelihood_upper_bound.sum(),
            }
        )
        return dic

    def m_step(
        self,
        x: torch.Tensor,
        n_sample: int,
        attribution: torch.Tensor,
        step: int,
        log_every: int,
        plot_every: int,
        only_bias: bool = False,
    ) -> torch.Tensor:
        """
        In the m-step, we update the parameters of the energy model
        """

        self.optimizer.zero_grad()
        self.optimizer_explicit_bias.zero_grad()
        dic_loss = self.snl_forward(
            x,
            attribution=attribution,
            n_sample=n_sample,
        )
        dic_loss["loss"].backward()
        # Normalize gradient :
        # for param in self.energy.parameters():
        #     if param.grad is not None:
        #         param.grad /= torch.linalg.norm(param.grad) + 1e-8

        # for k, mixture_component in enumerate(self.energy.mixture_component):
        #     for param in mixture_component.parameters():
        #         if param.grad is not None:
        #             param.grad /= torch.linalg.norm(param.grad) + 1e-8

        if only_bias:
            self.optimizer_explicit_bias.step()
        else:
            self.optimizer.step()

        if step % log_every == 0:
            current_param = self.energy.get_parameters(step=step)
            self.log(current_param, split="param", step=step)
            self.log(dic_loss, split="train", step=step)

        if step % plot_every == 0:
            self.plot_sample_energy(
                n_sample=1000,
                step=step,
            )
            self.plot_data_energy(
                step=step,
                data=x,
                attribution=attribution,
            )

        return dic_loss

    def calc_posterior(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        In the e-step, we use the current estimate of the log-likehood to get the posteriori
        """

        energy_target = self.energy.energy_per_cluster(x=x) + self.energy.explicit_bias
        self.energy.logit_pi.data = torch.nn.functional.log_softmax(
            self.energy.logit_pi,
            dim=-1,
        )

        log_posterior = torch.nn.functional.log_softmax(
            -energy_target + self.energy.logit_pi, dim=-1
        )  # Wether I replace with the approximation, or I just consider that it is self normalized

        # TODO : Change this with an option ?
        # min_pi = torch.full_like(log_prop, 1e-2)
        # log_prop = torch.where(
        #     log_prop < torch.log(min_pi), torch.log(min_pi), log_prop
        # )

        return log_posterior

    @torch.no_grad()
    def e_step(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        In the e-step, we use the current estimate of the log-likehood to get the posteriori"
        """

        log_posterior = self.energy.calc_posterior(x)
        attribution = self.energy.calculate_attribution(
            x, log_posterior, self.stochastic_em
        )

        self.energy.logit_pi.data = (
            log_posterior
            - torch.log(
                torch.tensor(
                    x.shape[0],
                ).to(torch.float32)
            )
        ).logsumexp(
            dim=0
        )  # 1/n \sum_i p(z_k|x_i)

        return attribution

    def train(
        self,
        n_iter: int,
        n_iter_pretrain: int = 1000,
        tolerance: float = 1e-3,
        max_number_m_step: int = 10000,
        eval_every: int = 100,
        log_every: int = 10,
        plot_every: int = 10,
    ) -> None:

        step = 0
        # Set the model with kmeans :

        assert hasattr(
            self.energy, "set_kmeans_centers"
        ), "Initialize the centers with KMeans not implemented"
        attribution = self.energy.set_kmeans_centers(complete_data=self.complete_data)

        # attribution = self.e_step(
        # self.complete_data,
        # )
        # attribution = attribution.detach()

        for step in tqdm.tqdm(range(n_iter + n_iter_pretrain)):
            if tolerance < 1e-3:
                break
            # while step < n_iter + n_iter_pretrain or tolerance > 1e-3:

            complete_data = self.complete_data

            dic = self.m_step(
                complete_data,
                n_sample=1000,
                attribution=attribution,
                step=step,
                log_every=log_every,
                plot_every=plot_every,
                only_bias=step < n_iter_pretrain,
            )

            if step > 0 and step % log_every == 0:
                self.log(dic, split="train", step=step)

            if (
                step > n_iter_pretrain
                and (step - n_iter_pretrain) % max_number_m_step == 0
            ):
                attribution = self.e_step(
                    complete_data,
                )
                attribution = attribution.detach()

            # Evaluation :
            if step % eval_every == 0:
                loss_dic = self.evaluate()
                loss_dic.update({"iter": step})
                update_dic(loss_dic, self.global_dic_test)
                self.log(loss_dic, split="val", step=step)

            step += 1


def forward(self, x: torch.Tensor, n_sample: int) -> torch.Tensor:
    """
    Forward pass for the EMSNL.
    """
    dic = self.snl_forward(x=x, n_sample=n_sample)
    if hasattr(self.energy, "regularize"):
        loss_reg = self.energy.regularize()
        loss += loss_reg
        dic.update({"loss_reg": loss_reg})
    if torch.any(torch.isnan(loss)):
        raise ValueError("Loss is NaN")
    return dic
