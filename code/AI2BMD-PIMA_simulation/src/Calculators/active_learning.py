import os
import os.path as osp
import pickle
from collections import deque
from pathlib import Path

from AIMD import arguments
from AIMD.fragment import FragmentData, FragmentInfo
from Calculators.monitor import FragmentMonitor


def frag2dict(x: FragmentData):
    return {
        'z': x.z,
        'pos': x.pos,
        'sym': x.sym
    }

def dict2frag(x: dict):
    return FragmentData(x['z'], x['pos'], x['sym'])


class ActiveLearningData:
    r"""A data packet for active learning.
    Each packet is of the following format:
        data: list[frame]
            frame: list[frag]
                frag: {
                    'type': 'dipeptides' | 'ACE_NMEs'
                    'index': int
                    'name': str like GLU, ASP, THR...
                    'z': ndarray[n]
                    'pos': ndarray[n, 3]
                }
    """
    def __init__(self, history_ringbuf: deque[FragmentData], monitor_errors: list[FragmentMonitor]):
        self.data = []
        for fragments in history_ringbuf:
            split = FragmentInfo.split(len(fragments))
            frame = []
            for monitor in monitor_errors:
                offset = 0
                if monitor.type == "ACE_NMEs":
                    offset = split[0]
                f_idx = monitor.idx + offset
                fragment = fragments[f_idx]
                frame.append({
                    'type': monitor.type,
                    'index': monitor.idx,
                    'name': fragment.sym,
                    'z': fragment.z,
                    'pos': fragment.pos,
                })
            self.data.append(frame)

    def dump(self, step: int):
        r"""Dumps the ringbuf to an active learning data packet."""
        dump_name = f"rollback_step_{step}_count_{len(self.data)}"
        dump_index = 0
        dump_dir = osp.join(arguments.get().log_dir, "ActiveLearningData")
        while True:
            dump_suffix = "" if dump_index == 0 else f".{dump_index}"
            dump_path = osp.join(dump_dir, f"{dump_name}{dump_suffix}.pkl")
            dump_xz = f"{dump_path}.tar.xz"
            if not Path(dump_xz).exists():
                break
            dump_index += 1
        with open(dump_path, "wb") as dump_file:
            pickle.dump(self.data, dump_file)
        assert 0 == os.system(f"cd {dump_dir} && tar cJf {dump_xz} {osp.basename(dump_path)} && rm {dump_path}")

    @classmethod
    def load(cls, dump_path: str):
        """Loads a ActiveLearningData packet."""
        if dump_path.endswith(".tar.xz"):
            assert 0 == os.system(f"tar xJf {dump_path}")
            dump_pkl = osp.basename(dump_path[:-7])
            with open(dump_pkl, "rb") as dump_file:
                data = pickle.load(dump_file)
            assert 0 == os.system(f"rm {dump_pkl}")
        else:
            with open(dump_path, "rb") as dump_file:
                data = pickle.load(dump_file)
        obj = cls(deque(), [])
        obj.data = data
        return obj

