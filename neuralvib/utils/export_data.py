"""Exporting Data"""

# %%
from neuralvib.utils.ckeckpoint import ckpt_filename
from neuralvib.utils.ckeckpoint import load_data

import numpy as np
import pandas as pd

if __name__ == "__main__":
    ckpt_epoch = 73700
    ckpt_path = (
        "/data/ruisiwang/NeuralVibConfigFlow/"
        "CH5/NNPES/InvariantHermite/"
        "CH5+_n_6_orb_1_rnvp_256_mlp_20_2_adam_lr_0.0001_bth_6000"
        "_acc_1_mcthr_20_stp_100_std_1.0_clp_5.0/"
        "2025-02-18/18:05:43/"
    )
    ckpt_file = ckpt_filename(ckpt_epoch, ckpt_path)
    ckpt = load_data(ckpt_file)
    x = ckpt["x"]
    x = np.array(x)
    carbons = x[:, 0, :]
    x -= carbons[:, np.newaxis, :]
    x = np.array(x).reshape(x.shape[0], -1)
    column_names = [
        "Cx",
        "Cy",
        "Cz",  # First point (C)
        "H1x",
        "H1y",
        "H1z",  # Second point (H1)
        "H2x",
        "H2y",
        "H2z",  # Third point (H2)
        "H3x",
        "H3y",
        "H3z",  # Fourth point (H3)
        "H4x",
        "H4y",
        "H4z",  # Fifth point (H4)
        "H5x",
        "H5y",
        "H5z",  # Sixth point (H5)
    ]
    df = pd.DataFrame(x, columns=column_names)
    df.to_csv("configs.csv", index=True)
    # %%
    import sys

    sys.path.append("../../")
