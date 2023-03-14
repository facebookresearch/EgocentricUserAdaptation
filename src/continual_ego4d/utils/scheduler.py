import copy
import multiprocessing as mp

from collections import deque
import traceback
import wandb
from ego4d.utils import logging

logger = logging.get_logger(__name__)


class RunConfig:

    def __init__(self, run_id: str, target_fn, fn_args):
        self.run_id: str = run_id
        self.fn_args: tuple = fn_args  # (cfg, user_id, user_datasets[user_id], device_id, path_handler, queue)
        self.target_fn = target_fn  # e.g. online_adaptation_single_user
        assert callable(self.target_fn)

    def get_process_instance(self, prefix_args=tuple(), daemon=False) -> mp.Process:
        return mp.Process(
            target=self.target_fn,
            args=prefix_args + self.fn_args,
            daemon=daemon  # Needs to have children for num_workers (set dameon=False)
        )

    def run_in_main_process(self, prefix_args=tuple()):
        """ Run in main process"""
        return self.target_fn(
            *(prefix_args + self.fn_args)
        )


class SchedulerConfig:
    """ Config attributes used for scheduling the job either sequentially or parallel."""

    def __init__(self,
                 run_entries: list[RunConfig], processed_run_ids, available_device_ids, max_runs_per_device):
        """

        :param run_entries: All the runs to schedule.
        :param processed_run_ids: Which ones of the 'all' should we skip as already processed in previous run.
        :param available_device_ids: Which CUDA idxs are free?
        :param max_runs_per_device: How many concurrent runs to schedule max on 1 GPU?
        """
        self.all_run_ids: list[str] = [e.run_id for e in run_entries]
        self.processed_run_ids: list[str] = processed_run_ids  # To skip
        logger.info(f"Skipping {processed_run_ids} as already processed")

        # To process
        self.runs_to_process: list[RunConfig] = [e for e in run_entries if e.run_id not in self.processed_run_ids]
        self.run_ids_to_process: list[str] = [e.run_id for e in self.runs_to_process]
        self.run_id_to_cfg: dict[str, RunConfig] = {e.run_id: e for e in self.runs_to_process}

        # Static device slots
        self.max_runs_per_device = max_runs_per_device
        assert self.max_runs_per_device >= 1
        self.available_device_ids: list[int] = available_device_ids * self.max_runs_per_device

        if len(self.runs_to_process) < len(self.available_device_ids):
            self.available_device_ids = self.available_device_ids[:len(self.runs_to_process)]
            logger.info(f"Defined more devices than runs, only using devices: {self.available_device_ids}")

        # State
        self.is_multiprocessing = len(self.available_device_ids) > 1 and len(self.runs_to_process) > 1

    def is_all_runs_processed(self):
        return len(self.processed_run_ids) >= len(self.all_run_ids)

    def schedule(self):
        if not self.is_multiprocessing:
            self.process_runs_sequentially()

        else:
            self.process_runs_parallel()

    def process_runs_parallel(self, ):
        """ This process is master process that spawn user-specific processes on free GPU devices once they are free.
        There is a known bug in Pytorch: https://github.com/pytorch/pytorch/issues/44156
        GPU memory is not freed after job completion. BUT if using the same process on the same device, it will reuse this
        memory. (e.g. for finished jobs about 2G memory remains, but if a next user-job is launched it will reuse this memory).

        Solution:
        Python API multiprocessing: https://docs.python.org/3/library/multiprocessing.html
        Create new Process manually and listen for output in shared queue.
        - Non-daemon(ic) process: Can create child-processes. Is required for our num_workers in dataloaders.
        - 2 reasons not to use join explicitly:
            - 'non-daemonic processes will be joined automatically'
            - Using join before queue.get() can result in deadlock:
            https://docs.python.org/3/library/multiprocessing.html#all-start-methods
            - If want to join explicitly: do so for processes that returned a value in the queue

        """
        process_timeout_s = 60 * 60 * 8  # hours timeout for single run
        run_id_queue = deque(copy.deepcopy(self.run_ids_to_process))
        submitted_run_ids = []
        finished_run_ids = []

        if len(run_id_queue) == 0:
            return

        # Init setup
        wandb.setup()  # See: https://docs.wandb.ai/guides/track/advanced/distributed-training#wandb-service

        # Shared queue
        queue = mp.Queue()

        runid_to_process = {}
        for device_id in self.available_device_ids:
            run_id = run_id_queue.popleft()
            submitted_run_ids.append(run_id)
            runid_to_process[run_id] = self.run_id_to_cfg[run_id].get_process_instance(
                prefix_args=(queue, device_id, run_id))

        # Start processes
        for p in runid_to_process.values():
            p.start()
        logger.info(f"Started and joined processes for run ids: {submitted_run_ids}")

        # Receive results for ALL user-processes, also the last ones
        remaining_runs_to_process = len(self.runs_to_process)
        interrupted_runs = []
        while remaining_runs_to_process > 0:
            # Get first next ready result
            try:
                interrupted, device_id, finished_run_id = queue.get(block=True, timeout=process_timeout_s)
            except:
                logger.exception(traceback.format_exc())
                logger.info(f"Executing on Queue time-out (deadlock?)")
                interrupted = True
            finished_run_ids.append(finished_run_id)
            remaining_runs_to_process -= 1

            logger.info(f"Finished processing run {finished_run_id}"
                        f" -> run_id_queue= {run_id_queue}, "
                        f"available_devices={device_id}, "
                        f"UNFINISHED={set(self.run_ids_to_process) - set(finished_run_ids)}, "
                        f"finished runs={len(finished_run_ids)}/{len(self.run_ids_to_process)},"
                        f"#remaining runs: {remaining_runs_to_process}")

            if interrupted:
                interrupted_runs.append(finished_run_id)
                logger.exception(f"Process for RUN {finished_run_id} failed because of Trainer being Interrupted."
                                 f"Not releasing GPU as might give Out-of-Memory exception.")
                continue

            # Zombie is finished process never joined, join to avoid
            # user_to_process[finished_user_id].join()

            # Launch new user with new process
            if len(run_id_queue) > 0:
                new_run_id = run_id_queue.popleft()
                submitted_run_ids.append(new_run_id)
                logger.info(f"{'*' * 20} LAUNCHING RUN {new_run_id} (device_id={device_id}) {'*' * 20}")

                process_instance = self.run_id_to_cfg[new_run_id].get_process_instance(
                    prefix_args=(queue, device_id, new_run_id)
                )

                runid_to_process[new_run_id] = process_instance
                process_instance.start()
            else:
                logger.info(f"Not scheduling new user-process as all are finished or running.")

        logger.info(f"Processed all runs, {len(interrupted_runs)} interrupted: {interrupted_runs}")

    def process_runs_sequentially(self):
        """ Sequentially iterate over users and process on single device.
        All processing happens in master process. """

        for run_entry in self.runs_to_process:
            interrupted, *_ = run_entry.run_in_main_process(
                prefix_args=(None, self.available_device_ids[0], run_entry.run_id)
            )
            if interrupted:
                logger.exception(f"Shutting down on RUN {run_entry.run_id}, because of Trainer being Interrupted")
                raise Exception()
