import matplotlib.pyplot as plt

from config import Config


class PlotUtils:

    @staticmethod
    def get_doi_extent():
        doi_length = Config.config["plot"]["doi_length"]
        doi_width = doi_length
        extent = [-doi_length/2, doi_length/2, -doi_width/2, doi_width/2]
        return extent

    @staticmethod
    def get_cmap():
        cmap = Config.config["plot"]["cmap"]
        if cmap == "jet":
            return plt.cm.jet

    @staticmethod
    def plot_output(xr, xg, guess, psnr_start, psnr_current):
        plot_extent = PlotUtils.get_doi_extent()
        plot_cmap = PlotUtils.get_cmap()

        fig, (ax3, ax1, ax2) = plt.subplots(ncols=3)

        ground_truth = ax3.imshow(xr, cmap=plot_cmap, extent=plot_extent)
        cb2 = fig.colorbar(ground_truth, ax=ax3, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax3.title.set_text("Ground Truth")
        ax3.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_real = ax1.imshow(xg, cmap=plot_cmap, extent=plot_extent)
        cb2 = fig.colorbar(guess_real, ax=ax1, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax1.title.set_text(f"Initial Reconstruction {psnr_start}:.2f")
        ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_obtained = ax2.imshow(guess, cmap=plot_cmap, extent=plot_extent)
        cb2 = fig.colorbar(guess_obtained, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax2.title.set_text(f"Final Reconstruction {psnr_current}:.2f")
        ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax2.get_yticklabels(), fontsize=12)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

        plt.show()
