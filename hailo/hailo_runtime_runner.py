from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


class HailoRuntimeUnavailable(RuntimeError):
    pass


def _import_hailo_platform():
    try:
        import hailo_platform as hp  # type: ignore
    except Exception as e:
        raise HailoRuntimeUnavailable(
            "Hailo runtime is unavailable. Install pyhailort/hailo_platform on target machine."
        ) from e
    return hp


def _to_nhwc(x: np.ndarray) -> np.ndarray:
    # NCHW -> NHWC
    if x.ndim != 4:
        raise ValueError(f"Expected 4D NCHW input, got shape={x.shape}")
    return np.transpose(x, (0, 2, 3, 1)).astype(np.float32, copy=False)


def _to_nchw(y: np.ndarray) -> np.ndarray:
    # Prefer NHWC -> NCHW
    if y.ndim == 4:
        return np.transpose(y, (0, 3, 1, 2)).astype(np.float32, copy=False)
    # NWC -> NCHW with H=1
    if y.ndim == 3:
        return np.transpose(y[:, np.newaxis, :, :], (0, 3, 1, 2)).astype(np.float32, copy=False)
    # NCW -> NCHW with H=1
    if y.ndim == 3 and y.shape[1] < y.shape[2]:
        return y[:, :, np.newaxis, :].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported runtime output shape={y.shape}")


class HailoHEFRunner:
    """Thin HEF inference helper around hailo_platform with minimal API assumptions."""

    def __init__(self, hef_path: str | Path):
        self.hp = _import_hailo_platform()
        self.hef_path = str(hef_path)
        self.hef = self.hp.HEF(self.hef_path)
        self.vdevice = self.hp.VDevice()

        interface = getattr(getattr(self.hp, "HailoStreamInterface", object), "PCIe", None)
        if interface is None:
            interface = getattr(getattr(self.hp, "HailoStreamInterface", object), "INTEGRATED", None)

        if interface is not None:
            cfg = self.hp.ConfigureParams.create_from_hef(self.hef, interface=interface)
        else:
            cfg = self.hp.ConfigureParams.create_from_hef(self.hef)

        n_groups = self.vdevice.configure(self.hef, cfg)
        self.network_group = n_groups[0] if isinstance(n_groups, (list, tuple)) else n_groups
        self.network_group_params = (
            self.network_group.create_params() if hasattr(self.network_group, "create_params") else None
        )

        fmt = getattr(getattr(self.hp, "FormatType", object), "FLOAT32", None)

        make_in = getattr(self.hp.InputVStreamParams, "make_from_network_group", None)
        if make_in is None:
            make_in = getattr(self.hp.InputVStreamParams, "make")
        self.in_params = make_in(self.network_group, format_type=fmt) if fmt is not None else make_in(self.network_group)

        make_out = getattr(self.hp.OutputVStreamParams, "make_from_network_group", None)
        if make_out is None:
            make_out = getattr(self.hp.OutputVStreamParams, "make")
        self.out_params = (
            make_out(self.network_group, format_type=fmt) if fmt is not None else make_out(self.network_group)
        )

        self.input_name = self._discover_single_name(self.hef, "get_input_vstream_infos")
        self.output_name = self._discover_single_name(self.hef, "get_output_vstream_infos")

    @staticmethod
    def _discover_single_name(hef: Any, method_name: str) -> str:
        infos = getattr(hef, method_name)()
        if not infos:
            raise RuntimeError(f"No vstream infos from HEF method {method_name}")
        if len(infos) != 1:
            names = [getattr(i, "name", "<unnamed>") for i in infos]
            raise RuntimeError(f"Expected single vstream for {method_name}, got {names}")
        return infos[0].name

    def infer_nchw(self, x_nchw: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x_nchw, torch.Tensor):
            arr = x_nchw.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = x_nchw.astype(np.float32, copy=False)
        inp = _to_nhwc(arr)

        activate_ctx = (
            self.network_group.activate(self.network_group_params)
            if self.network_group_params is not None and hasattr(self.network_group, "activate")
            else nullcontext()
        )
        with self.hp.InferVStreams(self.network_group, self.in_params, self.out_params) as pipe:
            with activate_ctx:
                out_dict = pipe.infer({self.input_name: inp})

        if isinstance(out_dict, dict):
            if self.output_name in out_dict:
                out = out_dict[self.output_name]
            elif len(out_dict) == 1:
                out = next(iter(out_dict.values()))
            else:
                raise RuntimeError(f"Unexpected runtime outputs: {list(out_dict.keys())}")
        else:
            out = out_dict
        return _to_nchw(np.asarray(out, dtype=np.float32))

