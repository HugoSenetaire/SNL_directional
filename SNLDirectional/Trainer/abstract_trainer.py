import copy

import torch
import tqdm

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
        no_wandb=False,
        name_wandb=None,
    ) -> None:
        if not no_wandb:
            self.logger = wandb
            wandb.init(
                project="snl_directional",
                entity="hugosenetaire",
                name=name_wandb,
                # mode="offline",
            )
            #    force=False)
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
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # self.device = "mps"
            self.device = "cpu"
        else:
            self.device = "cpu"

        self.energy.to(self.device)
        self.proposal.to(self.device)

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
            lr=lr,
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
        samples: torch.Tensor = None,
    ) -> None:
        """
        Plot both the energy contour and samples from the model.
        """
        if hasattr(self.energy, "sample_distribution") or samples is not None:
            if samples is None:
                samples = self.energy.sample_distribution(n_sample=n_sample)
            if hasattr(self.energy, "plot_input_and_energy"):
                self.energy.plot_input_and_energy(
                    step=step,
                    data=samples,
                    title="EnergyProposalSamples",
                )
            if hasattr(self.energy, "plot_input_and_energy_single") and hasattr(
                self.energy, "calculate_attribution"
            ):
                attribution = self.energy.calculate_attribution(samples)
                self.energy.plot_input_and_energy_single(
                    step=step,
                    data=samples,
                    attribution=attribution,
                    title="EnergyProposalSamples",
                )

    def plot_data_energy(
        self,
        step,
        data: torch.Tensor = None,
        attribution: torch.Tensor = None,
    ) -> None:
        """
        Plot the energy contour. If data is given, the data is also scatter on the energy.
        In the case of mixture models, the attribution is used to color the data points, if not given
        """

        if hasattr(self.energy, "plot_input_and_energy"):
            self.energy.plot_input_and_energy(
                step=step,
                data=None,
                attribution=None,
                title="EnergyContour",
            )
        if hasattr(self.energy, "plot_input_and_energy_single") and hasattr(
            self.energy, "calculate_attribution"
        ):
            attribution = self.energy.calculate_attribution(data)
            self.energy.plot_input_and_energy_single(
                step=step,
                data=data,
                attribution=attribution,
                title="EnergyContour",
            )

        if data is not None:
            if attribution is None and hasattr(self.energy, "calculate_attribution"):
                attribution = self.energy.calculate_attribution(data)
            if hasattr(self.energy, "plot_input_and_energy"):
                self.energy.plot_input_and_energy(
                    step=step,
                    data=data,
                    attribution=attribution,
                    title="EnergyData",
                )
            if hasattr(self.energy, "plot_input_and_energy_single") and hasattr(
                self.energy, "calculate_attribution"
            ):
                attribution = self.energy.calculate_attribution(data)
                self.energy.plot_input_and_energy_single(
                    step=step,
                    data=data,
                    attribution=attribution,
                    title="EnergyData",
                )

    def train(
        self,
        n_iter: int,
        n_iter_pretrain: int,
        eval_every: int = 500,
        log_every: int = 100,
        plot_every: int = 1000,
    ) -> None:

        if hasattr(self.energy, "set_kmeans_centers"):
            _ = self.energy.set_kmeans_centers(complete_data=self.complete_data)

        current_dataloader = iter(self.dataloader)

        plot_data = next(current_dataloader)[0]
        for k in range(len(current_dataloader)):
            if len(plot_data) > 1000:
                break
            try:
                plot_data = torch.cat([plot_data, next(current_dataloader)[0]], dim=0)
            except StopIteration:
                break

        plot_data = plot_data[:1000]

        self.plot_sample_energy(
            step=0,
            n_sample=1000,
        )

        self.plot_data_energy(
            step=0,
            data=plot_data,
            attribution=None,
        )

        for step in tqdm.tqdm(range(n_iter + n_iter_pretrain)):

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
                param_dic = self.energy.get_parameters(
                    step=step,
                )
                self.log(loss_dic, split="train", step=self.total_step)
                self.log(param_dic, split="param", step=self.total_step)
            if step % plot_every == 0:
                self.plot_sample_energy(
                    n_sample=1000,
                    step=step,
                )
                self.plot_data_energy(
                    step=step,
                    data=x,
                )

                # plot_curve(self.global_dic_train, self.global_param)

            # Update weights
            loss_dic["loss"].backward()

            self.optimizer_step(step=self.total_step, n_iter_pretrain=n_iter_pretrain)

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
