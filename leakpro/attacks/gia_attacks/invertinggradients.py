"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
import torch
from torchvision.utils import save_image

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.attacks.utils.util_functions import total_variation
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


class InvertingGradients(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(self: Self, handler: AbstractInputHandler, configs: dict) -> None:
        super().__init__(handler)
        self.handler = handler
        self.t_v_scale = configs.get("total_variation")
        self.attack_lr = configs.get("attack_lr")
        self.iterations = configs.get("at_iterations")
        logger.info("Inverting gradient initialized")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Inverting gradients"
        reference_str = """Geiping, Jonas, et al. Inverting gradients-how easy is it to
            break privacy in federated learning? Neurips, 2020."""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """

        self.reconstruction_loader = self.handler.replace_data_with_noise(self.train_indices, req_grad=True)
        feature_name = self.handler.feature_name
        self.reconstruction_data = getattr(self.reconstruction_loader.dataset, feature_name)
        assert self.reconstruction_data.requires_grad, "The dataset must require gradients."

        path = self.handler.configs["audit"]["output_dir"] + "/true_images.png"
        save_image(self.client_loader.dataset.data[:]*2+0.5, path)

    def run_attack(self:Self) -> GIAResults:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        optimizer = torch.optim.Adam([self.reconstruction_data], lr=self.attack_lr)
        # Decay the learning rate similar to original paper
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[self.iterations // 2.667,
                                                                        self.iterations // 1.6,
                                                                        self.iterations // 1.142],
                                                            gamma=0.1)

        for i in range(self.iterations):
            # loss function which does training and compares distance from reconstruction training to the real training.

            closure = self._gradient_closure(optimizer)

            loss = optimizer.step(closure)
            scheduler.step()
            mu, sigma = self.handler.get_meanstd(self.reconstruction_data)
            mu = mu.view(1, 3, 1, 1)
            sigma = sigma.view(1, 3, 1, 1)
            with torch.no_grad():
                self.reconstruction_data = torch.max(torch.min(self.reconstruction_data, (1 - mu) / sigma), -mu / sigma)
            if i % 20 == 0:
                logger.info(f"{i}: {loss}")
                tmp_name = f"iteration_{i}.png"
                path = self.handler.configs["audit"]["output_dir"] + "/" + tmp_name
                save_image(self.reconstruction_data, path)

            # add PSNR calculation and pick best image..
        # Collect client data to one tensor
        return GIAResults(self.client_loader, self.reconstruction_loader, 0, 0, 1)


    def _gradient_closure(self: Self, optimizer: torch.optim.Optimizer) -> Callable:
        """Returns a closure function that performs a gradient descent step.

        The closure function computes the gradients, calculates the reconstruction loss,
        adds a total variation regularization term, and then performs backpropagation.
        """
        def closure() -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function zeroes out the gradients of the optimizer and the model,
            computes the gradient and reconstruction loss, logs the reconstruction loss,
            optionally adds a total variation term, performs backpropagation, and optionally
            modifies the gradient of the input image.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
            optimizer.zero_grad()
            self.handler.target_model.zero_grad()

            gradient = self._get_pseudo_gradient(self.handler.target_model, self.reconstruction_loader)
            rec_loss = self._reconstruction_costs(gradient, self.client_gradient)

            if self.handler.configs["audit"]["modality"] == "image":
                # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
                rec_loss += (self.t_v_scale * total_variation(self.reconstruction_data))
            rec_loss.backward()
            return rec_loss
        return closure

    def _reconstruction_costs(self: Self, client_gradient: torch.Tensor, reconstruction_gradient: torch.Tensor) -> torch.Tensor:
        """Computes the reconstruction costs between client gradients and the reconstruction gradient.

        This function calculates the pairwise costs between each client gradient and the reconstruction gradient
        using the cosine similarity measure. The costs are accumulated and averaged over all client gradients.

        Returns
        -------
            torch.Tensor: The average reconstruction cost.

        """
        indices = torch.arange(len(reconstruction_gradient))
        weights = reconstruction_gradient[0].new_ones(len(reconstruction_gradient))
        total_costs = 0
        pnorm = [0, 0]
        costs = 0
        for i in indices:
            costs -= (client_gradient[i] * reconstruction_gradient[i]).sum() * weights[i]
            pnorm[0] += client_gradient[i].pow(2).sum() * weights[i]
            pnorm[1] += reconstruction_gradient[i].pow(2).sum() * weights[i]
        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        total_costs += costs
        return total_costs / len(client_gradient)

    def _configure_attack(self: Self, configs: dict) -> None:
        pass
