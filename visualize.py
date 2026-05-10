import argparse

import matplotlib.pyplot as plt

import dataset.loader
import dataset.main


def visualize_grid(
    grid_height: int,
    grid_width: int,
    images: list,
    labels: list[str],
    paths: list[str],
):
    _, axes = plt.subplots(grid_height, grid_width, figsize=(15, 15))
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        img_to_show = images[i].permute(1, 2, 0)
        ax.imshow(img_to_show)
        ax.set_title(f"Image {i + 1} ({labels[i]})")
        axes_flat[i].text(
            0.5,
            -0.05,
            "..." + paths[i][-16:],
            fontsize=8,
            ha="center",
            va="top",
            transform=axes_flat[i].transAxes,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def run_mode_visualize(grid_height: int, grid_width: int, only_hard: bool = False):
    subset = dataset.loader.NuclearCataractDataset(
        dataset.loader.NuclearCataractDataset.TrainValMode(0.8, 0.2),
        return_paths=True,
        only_hard=only_hard,
    )
    t = subset.train_set()

    images = []
    labels = []
    paths = []

    for i in range(grid_height * grid_width):
        sample = t[i]
        images.append(sample[0])
        labels.append(t.idx_to_label[sample[1]])
        paths.append(sample[2])

    visualize_grid(
        grid_height,
        grid_width,
        images,
        labels,
        paths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--grid_height", type=int, default=3)
    parser.add_argument("--grid_width", type=int, default=3)
    parser.add_argument("--only_hard", action="store_true")

    args = parser.parse_args()

    print(
        f"Running grid {args.grid_height}x{args.grid_width}. Only hard: {args.only_hard}"
    )

    run_mode_visualize(args.grid_height, args.grid_width, only_hard=args.only_hard)
