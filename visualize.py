import argparse

import matplotlib.pyplot as plt

import dataset.loader


def run_mode_visualize(grid_height: int, grid_width: int):
    subset = dataset.loader.NuclearCataractDataset(
        dataset.loader.NuclearCataractDataset.TrainValMode(0.8, 0.2),
        return_paths=True,
    )
    t = subset.train_set()

    fig, axes = plt.subplots(grid_height, grid_width, figsize=(15, 15))
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        img_to_show = t[i][0].permute(1, 2, 0)
        ax.imshow(img_to_show)
        ax.set_title(f"Image {i + 1} ({t.idx_to_label[t[i][1]]})")
        axes_flat[i].text(
            0.5,
            -0.05,
            "..." + t[i][2][-16:],
            fontsize=8,
            ha="center",
            va="top",
            transform=axes_flat[i].transAxes,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("grid_height", type=int)
    parser.add_argument("grid_width", type=int)

    args = parser.parse_args()

    run_mode_visualize(args.grid_height, args.grid_width)
