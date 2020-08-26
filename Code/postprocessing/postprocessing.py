from random import randint
import numpy as np
import pandas as pd
import PIL.Image as Image
import os
from os.path import join as osjoin
import matplotlib.pyplot as plt

class PostProc(object):

    def __init__(self):
        self.cur_dir = os.path.dirname(__file__)
        self.exp_dir = osjoin(self.cur_dir, os.path.dirname('../../FinalReport/experiments/'))
        self.pred_dir = osjoin(self.cur_dir, os.path.dirname('../../predictions/'))
        self.data_dir = osjoin(self.cur_dir, os.path.dirname('../../data/shock-datasets/'))
        self.nodal_dir = osjoin(self.data_dir, "nodal")
        self.shocknet4_dir = osjoin(self.cur_dir, os.path.dirname('../../predictions/'), "ShockNet4")
        self.pred_pred_dir = osjoin(self.shocknet4_dir, "nodal_ShockNet_adadelta_200-250", "Prediction")
        self.analysis_dir = osjoin(self.cur_dir, os.path.dirname('../../FinalReport/analysis/'))
        self.mats = ["Granite", "Basalt", "Limestone", "Dolomite", "Sandstones", "Chalks"]
        plt.style.use("seaborn")

    def draw_epoch_series(self, fname, txts, names, fsize, dpi=100, title=None, a=0.5, to_ana=False, cal_min=True):
        """Generate epoch series from different txt files

        Args:
            fname (string): .png file name.
            txts (list): Text file names in a list.
            names (list): Label names for charts.
            fsize (tuple): Matplotlib figure size.
            dpi (int): DPI.
            title (string, optional): Main title. Defaults to None.
            a (float, optional): Alpha channel. Defaults to 0.5.
            to_ana (bool, optional): If save to analysis folder. Defaults to False.
            cal_min (bool, optional): If slice depending on the minimum length. Defaults to True.
        """
        src_dir = self.exp_dir
        txt_paths = [osjoin(src_dir, txt) for txt in txts]
        fig, ax = plt.subplots(2, 2, figsize=fsize, dpi=dpi)
        min_cnt = 10000

        if cal_min:
            # check the min length of the txt file
            for i, txt_path in enumerate(txt_paths):
                cur_cnt = sum(1 for line in open(txt_path))
                min_cnt = min(cur_cnt, min_cnt)

        print("Epoch bound: ", min_cnt)
        for i, txt_path in enumerate(txt_paths):
            txt = open(txt_path, 'r')
            txtdata = txt.read()
            txt.close()

            if txtdata.find("tensor") != -1:
                txtdata = txtdata.replace("tensor(", "")\
                                .replace(" device='cuda:0', grad_fn=<DivBackward0>),", "")\
                                .replace(", device='cuda:0')", "")
                txt = open(txt_path, 'w')
                txt.write(txtdata)
                txt.close()
                print("File processed")
            else:
                print("File already processed.")

            txt_df = pd.read_csv(txt_path, header=None)
            tname = names[i]
            train, val = tname + " Training", tname + " Validation"
            ctrain, cval = np.random.rand(3,), np.random.rand(3,)
            indices = txt_df[5][:min_cnt].values
            if indices[0] != 1:
                indices = [i - 1 for i in indices] 
            # sizes = (txt_df[1][0], txt_df[2][0])
            # lr = txt_df[7][0]
            train_mse = [float(i) for i in txt_df[10]][:min_cnt]
            train_mae = [float(i) for i in txt_df[11]][:min_cnt]
            val_mse = [float(i) for i in txt_df[12]][:min_cnt]
            val_mae = [float(i) for i in txt_df[13]][:min_cnt]
            
            ax[0][0].set_title("a)", loc="left")
            ax[0][0].plot(indices, train_mse, color=ctrain, alpha=a, label=train)
            ax[0][0].plot(indices, val_mse, color=cval, alpha=a, label=val)
            ax[0][0].set_xlabel('Log Epochs')
            ax[0][0].set_ylabel('Log MSE')
            ax[0][0].set_xscale('log')
            ax[0][0].set_yscale('log')
            ax[0][0].get_xaxis().get_major_formatter().labelOnlyBase = False
            ax[0][0].get_yaxis().get_major_formatter().labelOnlyBase = False
            ax[0][0].legend()

            ax[0][1].set_title("b)", loc="left")
            ax[0][1].plot(indices, train_mae, color=ctrain, alpha=a, label=train)
            ax[0][1].plot(indices, val_mae, color=cval, alpha=a, label=val)
            ax[0][1].set_xlabel('Log Epochs')
            ax[0][1].set_ylabel('Log MAE')
            ax[0][1].set_xscale('log')
            ax[0][1].set_yscale('log')
            ax[0][1].get_xaxis().get_major_formatter().labelOnlyBase = False
            ax[0][1].get_yaxis().get_major_formatter().labelOnlyBase = False
            ax[0][1].legend()

            ax[1][0].set_title("c)", loc="left")
            ax[1][0].plot(indices, train_mse, color=ctrain, alpha=a, label=train)
            ax[1][0].plot(indices, val_mse, color=cval, alpha=a, label=val)
            ax[1][0].set_xlabel('Epochs')
            ax[1][0].set_ylabel('MAE')
            ax[1][0].get_xaxis().get_major_formatter().labelOnlyBase = False
            ax[1][0].get_yaxis().get_major_formatter().labelOnlyBase = False
            ax[1][0].legend()

            ax[1][1].set_title("d)", loc="left")
            ax[1][1].plot(indices, train_mae, color=ctrain, alpha=a, label=train)
            ax[1][1].plot(indices, val_mae, color=cval, alpha=a, label=val)
            ax[1][1].set_xlabel('Epochs')
            ax[1][1].set_ylabel('MAE')
            ax[1][1].get_xaxis().get_major_formatter().labelOnlyBase = False
            ax[1][1].get_yaxis().get_major_formatter().labelOnlyBase = False
            ax[1][1].legend()

        fig.tight_layout()
        if title:
            fig.suptitle(title, fontsize=20)
            fig.subplots_adjust(top=0.9)
        plt.savefig(osjoin(src_dir, fname))
        if to_ana:
            plt.savefig(osjoin(self.analysis_dir, fname))

    def get_random_img_names(self):
        grnd_path = self.pred_pred_dir
        names = []
        for mat in self.mats:
            namel = [i for i in os.listdir(grnd_path) if mat in i]
            name = namel[randint(0, len(namel)-1)]
            name = '_'.join(name.split('_')[2:])
            names.append(name)
        return names

    def prog_pred_comp(self, fname, uni_names, dirn, dpi=300):
        """Compare predictions to ground truths progressively.

        Args:
            fname (str): file name to be saved
            uni_names (str): list of names for searching
            dirn (str): directory name to get predictions
            dpi (int, optional): DPI. Defaults to 300.
        """
        def match_name(namel, key):
            for i, n in enumerate(namel):
                if key in n:
                    return i
            return None

        src_dir = osjoin(self.pred_dir, dirn)
        pth_paths = [osjoin(src_dir, p) for p in os.listdir(src_dir)]

        fig, ax = plt.subplots(6, 6, figsize=(12, 12), dpi=dpi)

        grnd_path = osjoin(pth_paths[0], "ResultsPNG")
        for i, uni_name in enumerate(uni_names):
            mat = uni_name.split('_')[0]
            img = np.asarray(Image.open(osjoin(grnd_path, uni_name)))
            ax[i][0].imshow(img)
            if i == 0:
                ax[i][0].set_title("Ground Truth")
            ax[i][0].set_ylabel(mat)
            ax[i][0].grid(False)
            dic = {'50':1, '100':2, '150':3, '200':4, '250':5}
            for p in pth_paths:
                num = p.split('_')[-1].split('-')[-1]
                title = dirn + " " + num
                pth_pred_dir = osjoin(src_dir, p, "Prediction")
                all_names = os.listdir(pth_pred_dir)
                ind = match_name(all_names, uni_name)
                act_name = all_names[ind]
                pred_img_path = osjoin(pth_pred_dir, act_name)
                img = np.asarray(Image.open(pred_img_path))
                ax[i][dic[num]].imshow(img)
                if i == 0:
                    ax[i][dic[num]].set_title(title)
                ax[i][dic[num]].grid(False)
        fig.tight_layout()

        plt.savefig(osjoin(self.analysis_dir, fname))
    
    def shock_img_comp(self, fname, num, figsize, dpi=300):
        """Plot a figure containing initial conditions, \
           ground truths and predicitons with MSE and MAE.

        Args:
            fname (str): Figure name to be saved.
            num (int): Number of samples.
            figsize (tuple): Figure size.
            dpi (int): DPI.
        """
        pred_dir = self.pred_pred_dir
        mat = self.mats[randint(0, len(self.mats) - 1)]
        mat_l = [i for i in os.listdir(pred_dir) if mat in i]
        names = []
        for i in range(num):
            name = mat_l[randint(0, len(mat_l)-1)]
            names.append(name)
        init_dir = osjoin(self.nodal_dir, "nodal_48x96", "InitialCondition")
        grnd_dir = osjoin(self.nodal_dir, "nodal_48x96", "ResultsPNG")
        fig, ax = plt.subplots(num, 3, figsize=figsize, dpi=dpi)
        for i in range(num):
            pred_name = names[i]
            uni_name = '_'.join(pred_name.split('_')[2:])
            mse = float(pred_name.split('_')[0])
            mae = float(pred_name.split('_')[1])
            init_img = np.asarray(Image.open(osjoin(init_dir, uni_name)))
            grnd_img = np.asarray(Image.open(osjoin(grnd_dir, uni_name)))
            pred_img = np.asarray(Image.open(osjoin(pred_dir, pred_name)))
            ax[i][0].imshow(init_img)
            ax[i][0].set_ylabel("%s %d" % (mat, i+1))
            ax[i][0].grid(False)
            ax[i][1].imshow(grnd_img)
            ax[i][1].grid(False)
            ax[i][2].imshow(pred_img)
            ax[i][2].grid(False)
            ax[i][2].set_title("MSE: %.4f\nMAE: %.4f" % (mse, mae), fontsize=6, loc='left')
        fig.tight_layout()
        plt.savefig(osjoin(self.analysis_dir, fname))

def main_postprocessing(func, fname, a, dpi, n):
    pp = PostProc()
    if func == "epoch_series":
        tar_txts = ["nodal_experiment_adadelta.txt", "nodal_uni_experiment_adadelta.txt",\
                    "nodal_3_experiment_adadelta.txt"]
        names = ["ShockNet4", "ShockNet4 Kaiming Uniform", "ShockNet3"]
        pp.draw_epoch_series(fname, tar_txts, names, (10, 10), dpi=dpi, a=a, to_ana=True)
    elif func == "ND_vs_BD":
        tar_txts = ["nodal_experiment_adadelta.txt", "boundary_experiment_adadelta.txt"]
        names = ["NDs", "BDs"]
        pp.draw_epoch_series(fname, tar_txts, names, (10, 10), dpi=dpi, a=a, to_ana=True)
    elif func == "transf_learning":
        tar_txts = ["nodal_experiment_adadelta.txt", "64x128_nodal_experiment_adadelta.txt"]
        names = ["48x96", "64x128 Transferred"]
        pp.draw_epoch_series(fname, tar_txts, names, (10, 10), dpi=dpi, a=a, to_ana=True, cal_min=False)
    elif func == "prog_pred_comp":
        uni_names = pp.get_random_img_names()
        pp.prog_pred_comp("ShockNet4_prog_pred_comp.png", uni_names, "ShockNet4", dpi=dpi)
        pp.prog_pred_comp("ShockNet3_prog_pred_comp.png", uni_names, "ShockNet3", dpi=dpi)
    elif func == "shock_img_comp":
        pp.shock_img_comp(fname, n, (6,8), dpi=dpi)
    else:
        print("%s not found from available functions.\nAvailable options including: epoch_series, ND_vs_BD, transf_learning, prog_pred_comp and shock_img_comp" % func)


if __name__ == "__main__":
    import argparse, getopt
    description = """\
                    Postprocessing to draw:\n\
                    1. Epoch series: 
                        1.1 Epoch series of ShockNet3-norm, ShockNet4-norm and ShockNet4-uni; \n\
                        1.1 NDs vs BDs epoch series; \n\
                        1.3 Transferred learning epoch series; \n\
                    2. Inputs corresponding to the output images will then be generated. \n\n\
                    Material dictionary includes: \n\
                        Limestone, Dolomite, Chalks, Sandstones, Granite and Basalt.\n\
                    Examples: \n\
                               For epoch series between different ShockNets: python3 postprocessing.py -f epoch_series \n\
                                                  For ND vs BD epoch series: python3 postprocessing.py -f ND_vs_BD \n\
                                      For transferred learning epoch series: python3 postprocessing.py -f transf_learning \n\
                                            For prediction change over time: python3 postprocessing.py -f shock_img_pred \n\
                        For initial condition, ground truth and predictions: python3 postprocessing.py -f prog_pred \n\
                    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--func", "-f", 
                        help="call the exact postprocessing function by name")
    parser.add_argument("--name", "-n", 
                        narg="*", default="figure.png", 
                        help="figure name to be saved")
    parser.add_argument("--alpha", "-a", type=int,
                        narg=1, default=0.6,
                        help="alpha channel of the plot")
    parser.add_argument("--dpi", "-d", type=int,
                        narg=1, default=250,
                        help="dpi")
    parser.add_argument("--sample", "-s", type=int,
                        narg=1, default=6, 
                        help="number of samples for general comparison")
    args = parser.parse_args()
    func, fname, a, dpi, nsample = args.func, args.name, args.alpha, args.dpi, args.sample

    main_postprocessing(func, fname, a, dpi, nsample)



    