import torch

from .abstract_trainer import AbstractTrainer


class SNLTrainer(AbstractTrainer):
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
        lr=1e-4,
        weight_decay=0,
    ) -> None:
        super().__init__(
            energy=energy,
            proposal=proposal,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            n_sample_train=n_sample_train,
            n_sample_test=n_sample_test,
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, x: torch.Tensor, n_sample: int) -> torch.Tensor:
        """
        Forward pass for the SNL.
        """
        x_sample = self.proposal.sample(n_sample)
        energy_target = self.energy(x=x)
        # energy_proposal = self.energy(x_sample).reshape((n_sample,))

        # proposal_log_prob = torch.from_numpy(
        #     self.proposal.log_prob(x_sample).detach().cpu().numpy()
        # ).reshape((n_sample,))

        loss_target = energy_target.mean(dim=0, keepdim=True)

        loss_proposal = (
            self.get_approximate_normalisation_constant(n_sample=n_sample).exp()
            + self.energy.explicit_bias
            - 1
        )
        likelihood_lower_bound = -loss_target - loss_proposal
        likelihood_upper_bound = (
            -energy_target.mean(dim=0, keepdim=True)
            - self.get_approximate_normalisation_constant(n_sample=n_sample)
            - self.energy.explicit_bias
        )

        loss = loss_target + loss_proposal

        dic = {
            "loss": loss,
            "loss_target": loss_target,
            "loss_proposal": loss_proposal,
            "log_likelihood_lower_bound (SNL)": likelihood_lower_bound,
            "log_likelihood_upper_bound": likelihood_upper_bound,
        }

        if hasattr(self.energy, "regularize"):
            loss_reg = self.energy.regularize()
            loss += loss_reg
            dic.update({"loss_reg": loss_reg})
        if torch.any(torch.isnan(loss)):
            raise ValueError("Loss is NaN")
        return dic

    def get_approximate_normalisation_constant(
        self, n_sample: int, batch_sample: int = 1e4
    ) -> torch.Tensor:
        """
        Get the approximate normalisation constant.
        """
        remaining_sample = n_sample
        current_batch = min(remaining_sample, batch_sample)

        x_sample = self.proposal.sample(current_batch)
        remaining_sample -= current_batch
        energy_proposal = self.energy(x_sample).reshape((current_batch,))
        proposal_log_prob = torch.from_numpy(
            self.proposal.log_prob(x_sample).detach().cpu().numpy()
        ).reshape((current_batch,))

        log_normalisation_constant = (
            -energy_proposal
            - self.energy.explicit_bias
            - proposal_log_prob
            - torch.tensor(n_sample).log()
        ).logsumexp(dim=0, keepdim=True)

        while remaining_sample > 0:
            current_batch = min(remaining_sample, batch_sample)
            x_sample = self.proposal.sample(current_batch)
            remaining_sample -= batch_sample

            energy_proposal = self.energy(x_sample).reshape((current_batch,))

            proposal_log_prob = torch.from_numpy(
                self.proposal.log_prob(x_sample).detach().cpu().numpy()
            ).reshape((current_batch,))

            log_normalisation_constant_aux = (
                -energy_proposal
                - self.energy.explicit_bias
                - proposal_log_prob
                - torch.tensor(n_sample).log()
            ).logsumexp(dim=0, keepdim=True)

            log_normalisation_constant = torch.logsumexp(
                torch.cat([log_normalisation_constant, log_normalisation_constant_aux]),
                dim=0,
                keepdim=True,
            )

        return log_normalisation_constant

    def evaluate(self, test_dataloader=None, n_sample: int = 1000) -> dict:
        """
        Evaluate the model on the test set.
        """
        if test_dataloader is None:
            test_dataloader = self.val_dataloader
        current_dataloader = iter(test_dataloader)

        x = next(current_dataloader)[0]
        n_sample = self.n_sample_test
        with torch.no_grad():
            forward = self.forward(x=x, n_sample=n_sample)
        return forward
