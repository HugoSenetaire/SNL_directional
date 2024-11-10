import copy

import torch

import wandb

from .plot_utils import plot_curve
from .trainer_utils import update_dic


class AbstractTrainer:
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

        self.logger = wandb
        wandb.init(project="snl_directional", entity="hugosenetaire")
        wandb.config.update(
            {
                "n_sample_train": n_sample_train,
                "n_sample_test": n_sample_test,
                "lr": lr,
                "energy": energy.__class__.__name__,
                "proposal": proposal.__class__.__name__,
            }
        )

        self.energy = energy
        self.best_energy = copy.deepcopy(energy)

        self.proposal = proposal
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.n_sample_train = n_sample_train
        self.n_sample_test = n_sample_test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_dic_train = {}
        self.global_dic_test = {}
        self.global_param = {}

        self.get_optimizer(optimizer="adam", lr=lr, weight_decay=weight_decay)
        self.get_lr_scheduler(
            optimizer=self.optimizer, lr_scheduler="step", step_size=1000, gamma=0.9
        )
        self.total_step = 0
        self.dir = wandb.run.dir
        self.best_loss = float("inf")
        self.best_step = -1

    def save_model(self, step) -> None:
        torch.save(self.energy.state_dict(), f"{self.dir}/energy_{step}.pt")

    def save_best_model(
        self,
    ) -> None:
        torch.save(self.energy.state_dict(), f"{self.dir}/best_energy.pt")

    def get_best_model(
        self,
    ) -> None:

        self.best_energy.load_state_dict(torch.load(f"{self.dir}/best_energy.pt"))

    def get_optimizer(self, optimizer, lr, weight_decay) -> None:
        self.optimizer = torch.optim.Adam(
            self.energy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.optimizer_explicit_bias = torch.optim.Adam(
            [self.energy.explicit_bias],
            lr=1e-2,
        )

    def forward(self, x: torch.Tensor, n_sample: int) -> torch.Tensor:
        raise NotImplementedError

    def get_lr_scheduler(
        self, optimizer, lr_scheduler="step", step_size=1000, gamma=0.99
    ) -> None:

        if lr_scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.1
            )
        else:
            self.scheduler = None

    def scheduler_step(
        self,
    ) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def optimizer_step(self, step, n_iter_pretrain) -> None:
        if step < n_iter_pretrain:
            self.optimizer_explicit_bias.step()
        else:
            self.optimizer.step()

    def evaluate(self) -> dict:
        raise NotImplementedError

    def log(self, loss_dic: dict, split: str = "train", step: int = -1) -> None:
        loss_dic = {f"{split}/{key}": value for key, value in loss_dic.items()}
        self.logger.log(
            loss_dic,
            step=step,
        )

    def plot_sample_energy(
        self,
        n_sample: int,
        step: int,
    ) -> None:
        if hasattr(self.energy, "plot_distribution"):
            self.energy.plot_distribution(
                step=step,
                n_sample=n_sample,
            )

    def train(
        self,
        n_iter: int,
        n_iter_pretrain: int,
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
                self.log(loss_dic, split="train", step=self.total_step)
                self.log(param_dic, split="param", step=self.total_step)
            if step % plot_every == 0:
                self.plot_sample_energy(
                    n_sample=1000,
                    step=step,
                )

                # plot_curve(self.global_dic_train, self.global_param)

            # Update weights

            loss_dic["loss"].backward()

            self.optimizer_step(step=self.total_step, n_iter_pretrain=n_iter_pretrain)
            # self.scheduler_step()

            # Evaluation :
            if step % eval_every == 0:
                loss_dic = self.evaluate()
                if loss_dic["loss"].item() < self.best_loss:
                    self.best_loss = loss_dic["loss"].item()
                    self.save_best_model()
                    self.best_step = self.total_step
                loss_dic.update({"iter": self.total_step})
                update_dic(loss_dic, self.global_dic_test)
                self.log(loss_dic, split="val", step=self.total_step)

            self.total_step += 1
