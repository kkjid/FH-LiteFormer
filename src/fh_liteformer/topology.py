"""Six directed relation types on the measured six-role system."""
import numpy as np

ROLES = ("C1", "C2", "Q1", "Q2", "H1", "H2")


RELATIONS = ("bank_parallel", "cap_mount", "mount_cap", "switch_mount", "mount_switch", "thermal_bridge")


def edges(mounts):
    src, dst, rel = [0,1,4,5], [1,0,5,4], [0,0,5,5]
    for j in range(2):
        h = 4+int(mounts[j])
        src.extend([j,h]); dst.extend([h,j]); rel.extend([1,2])
        h = 4+int(mounts[2+j])
        src.extend([2+j,h]); dst.extend([h,2+j]); rel.extend([3,4])
    return np.array([src,dst,rel], dtype=np.int64)
