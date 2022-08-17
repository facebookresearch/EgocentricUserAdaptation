from pytorch_lightning.trainer.connectors.logger_connector.logger_connector import LoggerConnector


class CustomLoggerConnector(LoggerConnector):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def should_update_logs(self) -> bool:
        """
        Log on
        - initial 0th step (we overwrite 'self.trainer.global_step+1' to 'self.trainer.global_step')
        - last training batch in epoch
        """
        should_log_every_n_steps = (self.trainer.global_step) % self.trainer.log_every_n_steps == 0
        return should_log_every_n_steps or self.trainer.is_last_batch

