from __future__ import absolute_import, division, print_function
import argparse
import os


class PuriLightXOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PuriLightX options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(os.path.expanduser("~"), "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of total epochs to train",
                                 default=20)
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay",
                                 default=0.01)

        # ARCHITECTURE
        self.parser.add_argument("--model",
                                 type=str,
                                 help="model architecture",
                                 choices=["purilightx-lp", "purilightx-lf"],
                                 default="purilightx-lp")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50])
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--use120",
                                 help="if set, uses 120 degree FOV for Cityscapes",
                                 action="store_true")

        # DATASET
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed"])
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark",
                                          "cityscapes_preprocessed", "cityscapes"],
                                 default="eigen_zhou")
        self.parser.add_argument("--png",
                                 help="if set, trains on raw pngs instead of jpgs",
                                 action="store_true")

        # EVALUATION
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 help="which evaluation split to use",
                                 choices=["eigen", "eigen_benchmark", "cityscapes", "make3d"],
                                 default="eigen")
        self.parser.add_argument("--eval_doj",
                                 help="if set, evaluates on dynamic object regions",
                                 action="store_true")
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 type=list,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])
        self.parser.add_argument("--eval_teacher",
                                 help="if set, evaluates the teacher model",
                                 action="store_true")

        # OPTIMIZATION
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 type=int,
                                 nargs="+",
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")

        # LOGGING
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # SYSTEM
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--num_threads",
                                 help="number of dataloader threads",
                                 default=1,
                                 type=int)
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set, multiplies predictions by this number",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor_before_clip",
                                 help="if set, multiplies predictions by this number before clipping",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set, saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set, disables evaluation",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        if self.options.dataset == "cityscapes_preprocessed":
            self.options.eval_split = "cityscapes"
            self.options.split = "cityscapes_preprocessed"
        return self.options