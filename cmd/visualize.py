import matplotlib.pyplot as plt

import dataset.loader

if __name__ == "__main__":
    subset = dataset.loader.NuclearCataractDataset(
        dataset.loader.NuclearCataractDataset.TrainValMode(0.8, 0.2),
        return_paths=True,
    )
    t = subset.train_set()

    fig, axes = plt.subplots(5, 5, figsize=(15, 5))
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        img_to_show = t[i][0].permute(1, 2, 0)
        ax.imshow(img_to_show)
        ax.set_title(f"Image {i + 1} ({t.idx_to_label[t[i][1]]})")
        axes_flat[i].text(
            0.5,
            -0.12,
            "..." + t[i][2][-16:],
            fontsize=8,
            ha="center",
            va="top",
            transform=axes_flat[i].transAxes,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()
