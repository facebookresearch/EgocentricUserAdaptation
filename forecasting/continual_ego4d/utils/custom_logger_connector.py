from pytorch_lightning.trainer.connectors.logger_connector.logger_connector import LoggerConnector


class CustomLoggerConnector(LoggerConnector):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def should_update_logs(self) -> bool:
        should_log_every_n_steps = (self.trainer.global_step) % self.trainer.log_every_n_steps == 0
        return should_log_every_n_steps or self.trainer.should_stop


