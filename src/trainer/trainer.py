from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        gen_outputs = self.model(batch["mel_gt"])
        batch["prediction"] = gen_outputs["prediction"]

        if self.is_train:
            # ---- Обновление дискриминатора ----
            self.optimizer_d.zero_grad()

            disc_out = self.model.discriminator_forward(
                batch["prediction"].detach(),
                batch["wav_gt"]
            )

            batch["outs_predicted"] = disc_out["outs_predicted"]
            batch["fmaps_predicted"] = disc_out["fmaps_predicted"]
            batch["outs_gt"] = disc_out["outs_gt"]
            batch["fmaps_gt"] = disc_out["fmaps_gt"]

            d_loss_dict = self.criterion.discriminator_loss(
                outs_predicted=batch["outs_predicted"],
                fmaps_predicted=batch["fmaps_predicted"],
                outs_gt=batch["outs_gt"],
                fmaps_gt=batch["fmaps_gt"]
            )

            batch["disc_loss"] = d_loss_dict["disc_loss"]
            batch["disc_loss"].backward()

            self._clip_grad_norm(self.model.MSD)
            self._clip_grad_norm(self.model.MPD)
            self.optimizer_d.step()

            # ---- Обновление генератора ----
            self.optimizer_g.zero_grad()

            disc_out_2 = self.model.discriminator_forward(
                batch["prediction"],
                batch["wav_gt"]
            )

            batch["outs_predicted"] = disc_out_2["outs_predicted"]
            batch["fmaps_predicted"] = disc_out_2["fmaps_predicted"]
            batch["outs_gt"] = disc_out_2["outs_gt"]
            batch["fmaps_gt"] = disc_out_2["fmaps_gt"]

            g_loss_dict = self.criterion.generator_loss(
                outs_predicted=batch["outs_predicted"],
                fmaps_predicted=batch["fmaps_predicted"],
                outs_gt=batch["outs_gt"],
                fmaps_gt=batch["fmaps_gt"]
            )

            batch["gen_loss"] = g_loss_dict["gen_loss"]
            batch["gen_loss"].backward()

            self._clip_grad_norm(self.model.generator)
            self.optimizer_g.step()

            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
