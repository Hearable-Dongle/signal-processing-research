import shutil
import atexit
import dataclasses
import json
import logging
import random
import sys
from logging import Formatter, Logger, StreamHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.random import Generator
from numpy.typing import NDArray


@dataclasses.dataclass
class Audio_Sources:
    input: Path | str
    loc: list[float]
    classification: str

    def resolve_input(self, project_dir: Path) -> None:
        self.input = project_dir / "input" / self.input


class Config:
    __project_dir = Path(__file__).parent.parent.resolve()
    __log_str = f"{'Date/Time:':<19}  |  {'Level:':<8}  |  Message:\n"

    __log_fmt = Formatter(
        fmt="{asctime:<19}  |  {levelname:<8}  |  {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    def __init__(
        self,
        config_path: Path = Path(__project_dir, "config", "config.json"),
        output_path: Path | None = None,
    ) -> None:
        with config_path.open("r") as fp:
            self.__project_config = json.load(fp)

        self.__log = logging.getLogger(self.__project_dir.stem)

        if output_path:
            self.__output_path = output_path
        else:
            self.__output_path = (
                self.__project_dir / self.__project_config["output_dir"]
            )

        if not self.__output_path.exists():
            self.__output_path.mkdir(parents=True)

        shutil.copy(config_path, self.__output_path)

        self.__log.setLevel(logging.INFO)

        # Configure log handler
        log_hdlr = StreamHandler(sys.stdout)
        log_hdlr.setFormatter(self.__log_fmt)
        log_hdlr.setLevel(logging.INFO)
        self.__log.addHandler(log_hdlr)

        # Print to stdout
        sys.stdout.write(self.__log_str)

        atexit.register(self.close)

        self.set_all_seeds(self.__project_config["seed"])
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__log.info(f"Using device: {self.__device}")

    def set_all_seeds(self, seed: int) -> None:
        torch.manual_seed(seed)  # pyright: ignore [reportUnknownMemberType]
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        self.__rng = np.random.default_rng(seed=seed)

    @property
    def log(self) -> Logger:
        return self.__log

    @property
    def rng(self) -> Generator:
        return self.__rng

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def sound_speed(self) -> float:
        return self.__project_config["sound_speed"]

    @property
    def input_dir(self) -> Path:
        return Path(self.__project_dir, self.__project_config["input_dir"]).resolve()

    @property
    def output_dir(self) -> Path:
        return self.__output_path

    @property
    def checkpoint_file(self) -> Path | None:
        if self.__project_config["checkpoint_file"]:
            checkpoint_file_path = Path(
                self.__project_config["checkpoint_file"]
            ).resolve()
        else:
            checkpoint_file_path = None

        return checkpoint_file_path

    @property
    def train_max_sessions(self) -> int | None:
        max_sessions = self.__project_config["training"]["max_sessions"]
        if max_sessions == 0:
            max_sessions = None

        return max_sessions

    @property
    def val_max_sessions(self) -> int | None:
        max_sessions = self.__project_config["validation"]["max_sessions"]
        if max_sessions == 0:
            max_sessions = None

        return max_sessions

    @property
    def train_batch_size(self) -> int:
        return self.__project_config["training"]["batch_size"]

    @property
    def val_batch_size(self) -> int:
        return self.__project_config["validation"]["batch_size"]

    @property
    def optimizer_params(self) -> dict[str, Any]:
        return self.__project_config["optimizer"]

    @property
    def scheduler_params(self) -> dict[str, Any]:
        return self.__project_config["scheduler"]

    @property
    def epoch_count(self) -> int:
        return self.__project_config["epoch_count"]

    @property
    def fs(self) -> int:
        return self.__project_config["sampling_frequency"]

    @property
    def room_dim(self) -> NDArray[np.float64]:
        return np.array(
            list(self.__project_config["room_settings"]["dim"].values()), dtype=np.float64
        )

    @property
    def reflection_count(self) -> int:
        return self.__project_config["room_settings"]["reflection_count"]

    @property
    def mic_count(self) -> int:
        return self.__project_config["mic_settings"]["count"]

    @property
    def mic_spacing(self) -> float:
        return self.__project_config["mic_settings"]["spacing"]

    @property
    def mic_loc(self) -> NDArray[np.float64]:
        return np.array(self.__project_config["mic_settings"]["loc"], dtype=np.float64)

    @property
    def mic_type(self) -> str:
        return self.__project_config["mic_settings"]["type"]

    @property
    def sources(self) -> list[Audio_Sources]:
        sources = [Audio_Sources(**source) for source in self.__project_config["sources"]]
        for source in sources:
            source.resolve_input(self.__project_dir)

        return sources

    @property
    def noise_estimation_method(self) -> str:
        return self.__project_config.get("noise_estimation_method", "predict")

    @property
    def noise_pc_count(self) -> int:
        return self.__project_config["noise_pc_count"]

    @property
    def noise_reg_factor(self) -> float:
        return self.__project_config["noise_reg_factor"]

    @property
    def frame_duration(self) -> float:
        return self.__project_config["frame_duration"]

    def close(self) -> None:
        for log_hdlr in self.__log.handlers:
            self.__log.removeHandler(log_hdlr)
            log_hdlr.close()

        del self.__log
