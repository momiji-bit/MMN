import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import zipfile

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
# from tensorboardX import SummaryWriter # Removed: Not needed for validation only
from tqdm import tqdm
import json

from torchlight import DictAction

contrast = True
head_only = False
print(f"contrast:{contrast}, head_only:{head_only}")


def action2body(x):
    """Maps action class index to body part index."""
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6


def save_predictions_to_zip(scores, work_dir, k=5):
    """
    Processes prediction scores and saves them to prediction.csv and prediction.zip.

    :param scores: A 2D tensor of prediction scores (num_samples, num_classes)
    :param work_dir: The directory to save the output files.
    :param k: The number for top-k predictions.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)

    # Get top-k action predictions
    _, top_k_action_indices = torch.topk(scores, k=k, dim=1)

    num_samples, num_action_classes = scores.shape

    # Map top-k actions to corresponding body parts
    action_to_body_map = torch.tensor(
        [action2body(j) for j in range(num_action_classes)],
        dtype=torch.long,
        device=scores.device
    )
    top_k_body_indices = action_to_body_map[top_k_action_indices]

    # Convert to lists for CSV writing
    top_k_action_list = top_k_action_indices.cpu().tolist()
    top_k_body_list = top_k_body_indices.cpu().tolist()

    headers = ['vid']
    headers.extend([f'action_pred_{i + 1}' for i in range(k)])
    headers.extend([f'body_pred_{i + 1}' for i in range(k)])

    # Ensure output directory exists
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    csv_filepath = os.path.join(work_dir, "prediction.csv")

    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(num_samples):
            vid = f"test{i:04d}.mp4"
            action_preds = top_k_action_list[i]
            body_preds = top_k_body_list[i]
            row = [vid] + action_preds + body_preds
            writer.writerow(row)

    # Zip the CSV file
    zip_filepath = os.path.join(work_dir, "prediction.zip")
    with zipfile.ZipFile(zip_filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_filepath, arcname="prediction.csv")

    print(f"Predictions successfully saved to {zip_filepath}")


def merge_predictions(j_dir, b_dir, output_dir):
    """
    Loads J (joint) and B (bone) prediction.json files, merges them by
    addition, and saves the result in the same format as the eval function.
    """
    print(f"Starting prediction merge...")
    print(f"  Joint (J) Dir: {j_dir}")
    print(f"  Bone (B) Dir:  {b_dir}")
    print(f"  Output Dir:    {output_dir}")

    j_json_path = os.path.join(j_dir, 'prediction.json')
    b_json_path = os.path.join(b_dir, 'prediction.json')

    # Check if files exist
    if not os.path.exists(j_json_path):
        print(f"Error: Cannot find prediction file: {j_json_path}")
        return
    if not os.path.exists(b_json_path):
        print(f"Error: Cannot find prediction file: {b_json_path}")
        return

    if output_dir is None:
        print("Error: --work-dir must be specified as the output directory for merged results.")
        return

    try:
        # Load scores from JSON files
        with open(j_json_path, 'r') as f:
            j_scores_list = json.load(f)
        with open(b_json_path, 'r') as f:
            b_scores_list = json.load(f)

        # Convert to tensors
        j_scores = torch.tensor(j_scores_list)
        b_scores = torch.tensor(b_scores_list)

        # Check shapes
        if j_scores.shape != b_scores.shape:
            print(f"Error: Score shapes mismatch! J: {j_scores.shape}, B: {b_scores.shape}")
            return

        # Merge by adding scores
        merged_scores = j_scores * 0.5 + b_scores * 0.5

        print(f"Scores loaded and merged. Total samples: {merged_scores.shape[0]}")

        # Save the merged scores using the same logic
        save_predictions_to_zip(merged_scores, output_dir)

        print(f"Successfully merged predictions and saved to {os.path.join(output_dir, 'prediction.zip')}")

    except Exception as e:
        print(f"An error occurred during merging: {e}")
        traceback.print_exc()


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_parser():
    parser = argparse.ArgumentParser(
        description='Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-Action Recognition')
    parser.add_argument('--work-dir', default=None, help='the work folder for storing results or merged output')
    parser.add_argument('--model_saved_name', default='')

    # ===config==========================================================================================================================================
    parser.add_argument('--weights', default=None, help='the weights for network initialization')

    parser.add_argument('--config', default='./config/test/MA52_J.yaml', help='path to the configuration f_Lineare')
    # parser.add_argument('--config', default='./config/test/MA52_B.yaml', help='path to the configuration f_Lineare')
    # parser.add_argument('--config', default='./config/test/iMiGUE_J.yaml', help='path to the configuration f_Lineare')
    # parser.add_argument('--config', default='./config/test/iMiGUE_B.yaml', help='path to the configuration f_Lineare')
    # ===config==========================================================================================================================================

    # ===merge arg=======================================================================================================================================
    parser.add_argument('--merge', nargs=2, metavar=('J_DIR', 'B_DIR'), default=None,
                        help='Paths to J and B work directories to merge predictions from. '
                             'If set, runs merge-only. --work-dir is used as output dir.')
    # ===merge arg=======================================================================================================================================

    # processor
    parser.add_argument('--phase', default='test', help='must be test')  # Default changed to 'test'
    parser.add_argument('--save-score', type=str2bool, default=True,  # Default changed to 'True'
                        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default=None, help='data loader will be used')
    parser.add_argument('--num-worker', type=int, help='the number of worker for data loader')
    # Removed: --train-feeder-args
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    # parser.add_argument('--weights', default=None, help='the weights for network initialization') # Duplicates --weights above
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # optim
    # Removed all training-related optimizer and scheduler arguments
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    # Removed: --batch-size
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    # Removed: --start-epoch, --num-epoch
    parser.add_argument('--loss-type', type=str, default='CE')
    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        self.load_model()
        self.load_data()

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, weights_only=True)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights, strict=False)
                if head_only:
                    for param in self.model.parameters():
                        param.requires_grad = False
                    for param in self.model.head.parameters():
                        param.requires_grad = True
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state, strict=False)

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def eval(self, epoch, save_score=False, loader_name=['train']):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            score_frag = []

            process = tqdm(self.data_loader[ln], ncols=100)
            for batch_idx, (data, index_t, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    index_t = index_t.float().cuda(self.output_device)
                    output = self.model(data, index_t)
                    score_frag.append(output.data)

            score = torch.cat(score_frag, dim=0)

            if save_score:
                # Save the raw scores (logits) to prediction.json
                json_path = os.path.join(self.arg.work_dir, "prediction.json")
                with open(json_path, 'w') as f:
                    json.dump(score.cpu().tolist(), f)

                # Process scores and save to CSV and ZIP
                save_predictions_to_zip(score, self.arg.work_dir)
                self.print_log(f"Predictions saved to {os.path.join(self.arg.work_dir, 'prediction.zip')}")

    def start(self):
        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.arg.print_log = False
        self.print_log('Model:   {}.'.format(self.arg.model))
        self.print_log('Weights: {}.'.format(self.arg.weights))
        self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
        self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()

    # Only load config if we are NOT in merge mode
    if p.config is not None and p.merge is None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    # Check if we are in merge mode
    if arg.merge:
        # We are in merge-only mode.
        # arg.merge will be a list: [J_DIR, B_DIR]
        # arg.work_dir will be used as the output directory
        merge_predictions(j_dir=arg.merge[0], b_dir=arg.merge[1], output_dir=arg.work_dir)
    else:
        # Standard evaluation mode
        init_seed(arg.seed)
        processor = Processor(arg)
        processor.start()