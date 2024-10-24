import json
import os
from statistics import mean

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.constants import (
    CHOICES_SCORES,
    IDX,
    LABEL,
    LM_TARGET,
    NEG_INDEX,
    TEMPLATE_LOGITS,
)

from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler


class LitModule(LightningModule):
    def __init__(self, config, model, datamodule, evaluator):
        super().__init__()
        self.config = config
        self.model = model
        self.load_model()
        self.datamodule = datamodule
        self.evaluator = evaluator

        self.use_deepspeed = self.config.compute_strategy.startswith("deepspeed")
        self.use_ddp = self.config.compute_strategy.startswith("ddp")

        self._last_global_step_saved = -1

        self.validation_outputs = []
        self.test_outputs = []

    def _compute_lm_loss(self, output, label):
        logits = output[TEMPLATE_LOGITS]
        lm_target = output[LM_TARGET]
        num_choices = max(output[NEG_INDEX]) + 1  # Neg index is last element, so total number is +1

        bs = int(lm_target.size(0) / num_choices)

        lm_loss = F.cross_entropy(
            logits.view(bs, num_choices, *logits.size()[1:])[range(bs), label].flatten(0, 1),
            lm_target.view(bs, num_choices, -1)[range(bs), label].flatten(0, 1),
        )
        return lm_loss

    def _compute_mc_loss(self, output, label):
        choices_scores = output[CHOICES_SCORES]
        if self.config.mc_loss > 0:
            mc_loss = F.cross_entropy(choices_scores, label)
        else:
            mc_loss = 0.0
        return mc_loss

    def _compute_unlikely_loss(self, output, label):
        logits = output[TEMPLATE_LOGITS]
        lm_target = output[LM_TARGET]
        num_choices = max(output[NEG_INDEX]) + 1  # Neg index is last element, so total number is +1

        bs = int(lm_target.size(0) / num_choices)

        if self.config.unlikely_loss > 0:
            cand_loglikely = -F.cross_entropy(logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none").view(
                bs, num_choices, -1
            )
            cand_loglikely += (lm_target < 0).view(bs, num_choices, -1) * -100
            cand_loglikely[range(bs), label] = -100
            unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
        else:
            unlikely_loss = 0.0
        return unlikely_loss

    def _compute_loss(self, output, label):
        lm_loss = self._compute_lm_loss(output, label)
        tensorboard_logs = {"lm_loss": lm_loss.item()}

        mc_loss = self._compute_mc_loss(output, label)
        tensorboard_logs["mc_loss"] = mc_loss.item()

        unlikely_loss = self._compute_unlikely_loss(output, label)
        tensorboard_logs["unlikely_loss"] = unlikely_loss.item()

        loss = lm_loss + mc_loss * self.config.mc_loss + unlikely_loss * self.config.unlikely_loss

        return loss, tensorboard_logs

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        label = batch[LABEL]

        loss, tensorboard_logs = self._compute_loss(output=output, label=label)

        # if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
        #     self.log_dict(tensorboard_logs)
        
        self.log("train_loss", loss.cpu().item(), prog_bar=True, on_step=True)

        # if self.global_step % self.config.save_step_interval == 0:
        #     self.save_model()

        return loss

    def validation_step(self, batch, batch_idx):
        batch_output = self.model(batch)  # TODO: Remove Label from output for clean seperation.
        self.validation_outputs.append(batch_output)
        return batch_output

    def on_validation_epoch_end(self):
        # exchange outputs between processes
        if self.use_deepspeed or self.use_ddp:
            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, self.validation_outputs)
            if dist.get_rank() == 0:
                outputs = [batch_output for outputs in gathered_outputs for batch_output in outputs]

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            # let rank 0 collect all outputs
            accumulated = {key: [] for key in self.validation_outputs[0].keys()}
            for batch_output in self.validation_outputs:
                for key, value in batch_output.items():
                    accumulated[key].extend(value)

            # multi-process may yield dupliated examples in the last batch
            valid_mask = []
            idx_set = set()
            for idx in accumulated[IDX]:
                valid_mask.append(idx not in idx_set)
                idx_set.add(idx)
            for key, values in accumulated.items():
                accumulated[key] = [v for v, m in zip(values, valid_mask) if m]

            # compute and log results
            metrics = self.evaluator.compute_metric(accumulated)

            result_str = json.dumps(metrics) + "\n"
            with open(self.config.dev_score_file, "a+") as f:
                f.write(result_str)
        else:
            metrics = {}

        self.validation_outputs = []
        return metrics

    def test_step(self, batch, batch_idx):
        batch_output = self.model(batch)  # TODO: Remove Label from output for clean seperation.
        self.test_outputs.append(batch_output)
        return batch_output

    def on_test_epoch_end(self):
        # exchange outputs between processes
        if self.use_deepspeed or self.use_ddp:
            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, self.test_outputs)
            if dist.get_rank() == 0:
                outputs = [batch_output for outputs in gathered_outputs for batch_output in outputs]

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            # let rank 0 collect all outputs
            accumulated = {key: [] for key in self.test_outputs[0].keys()}
            for batch_output in self.test_outputs:
                for key, value in batch_output.items():
                    accumulated[key].extend(value)

            # multi-process may yield dupliated examples in the last batch
            valid_mask = []
            idx_set = set()
            for idx in accumulated[IDX]:
                valid_mask.append(idx not in idx_set)
                idx_set.add(idx)
            for key, values in accumulated.items():
                accumulated[key] = [v for v, m in zip(values, valid_mask) if m]

            # compute and log results
            metrics = self.evaluator.compute_metric(accumulated)

            result_str = json.dumps(metrics) + "\n"
            with open(self.config.test_score_file, "a+") as f:
                f.write(result_str)
        else:
            metrics = {}

        return metrics


    def configure_optimizers(self):
        optimizer, self.trainable_param_names = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_end(self):
        self.save_model(finish=True)

    def load_model(self):
        if self.config.load_weight != "":
            print("Loading weights for model...")
            trainable_states = torch.load(self.config.load_weight, map_location=torch.device("cpu"))
            load_result = self.model.load_state_dict(trainable_states, strict=False)
            assert (
                len(load_result.unexpected_keys) == 0
            ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def save_model(self, finish=False):
        if self.config.save_model and (finish or self._last_global_step_saved != self.global_step):
            if finish:
                model_fname = os.path.join(self.config.exp_dir, "finish.pt")
            else:
                model_fname = os.path.join(self.config.exp_dir, f"global_step{self.global_step}.pt")

            trainable_states = {
                param_name: param_weight.cpu()
                for param_name, param_weight in self.model.state_dict().items()
                if param_name in self.trainable_param_names
            }
            torch.save(trainable_states, model_fname)

            self._last_global_step_saved = self.global_step