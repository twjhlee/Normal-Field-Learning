"""
run training, export mesh(fine), render depth for a single config file
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--overwrite_datadir", default="None", type=str)
parser.add_argument("--overwrite_logdir", default="None", type=str)
args = parser.parse_args()

cmd = "python run.py --config {} --overwrite_datadir {} --overwrite_logdir {}".format(args.config, args.overwrite_datadir, args.overwrite_logdir) +\
    " && python run.py --config {} --overwrite_datadir {} --overwrite_logdir {} --export_fine_only".format(args.config, args.overwrite_datadir, args.overwrite_logdir) +\
    " && python run.py --config {} --overwrite_datadir {} --overwrite_logdir {} --render_test --render_only --dump_images".format(args.config, args.overwrite_datadir, args.overwrite_logdir)

os.system(cmd)
