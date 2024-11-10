import torch

from .abstract_trainer import AbstractTrainer
from .snl_loop import SNLTrainer


class SNLTrainer(AbstractTrainer, SNLTrainer):
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
    ) -> None:
        super().__init__(
            energy=energy,
            proposal=proposal,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            n_sample_train=n_sample_train,
            n_sample_test=n_sample_test,
            lr=lr,
        )

    def forward(self, x: torch.Tensor, n_sample: int) -> torch.Tensor:
        super(SNLTrainer, self).forward(x, n_sample)

    def e_step(self, x: torch.Tensor, n_sample: int) -> torch.Tensor:
        energy_per_cluster()

    def train(
        self,
        n_iter: int,
        n_iter_pretrain: int,
        max_step_every: int = 1000,
        eval_every: int = 500,
        log_every: int = 100,
        plot_every: int = 1000,
    ) -> None:
        current_dataloader = iter(self.dataloader)

        for step in range(n_iter + n_iter_pretrain):

            # Forward pass
            try:
                x = next(current_dataloader)[0]
            except StopIteration:
                current_dataloader = iter(self.dataloader)
                x = next(current_dataloader)[0]
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss_dic = self.forward(x, self.n_sample_train)

            # Plotting and feedback
            if step % log_every == 0:
                param_dic = self.energy.get_parameters()
                self.log(loss_dic, split="train", step=step)
                self.log(param_dic, split="param", step=step)
            if step % plot_every == 0:
                self.plot_sample_energy(
                    n_sample=1000,
                    step=step,
                )

                # plot_curve(self.global_dic_train, self.global_param)

            # Update weights

            loss_dic["loss"].backward()

            self.optimizer_step(step=step, n_iter_pretrain=n_iter_pretrain)
            # self.scheduler_step()

            # Evaluation :
            if step % eval_every == 0:
                loss_dic = self.evaluate()
                loss_dic.update({"iter": step})
                update_dic(loss_dic, self.global_dic_test)
                self.log(loss_dic, split="val", step=step)
