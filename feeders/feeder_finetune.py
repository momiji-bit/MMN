import numpy as np
import random
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, split, data_type='j', repeat=1, p=0.5,
                 window_size=-1, debug=False, partition=False,
                 data_path=None, label_path=None):
        """
        Initialize the Feeder class for loading and preprocessing 3D skeleton data (17 joints).
        Modified for finetune datasets with custom paths.

        Args:
            split (str): Split type ('train' or 'test').
            data_type (str): Data type, default is 'j'.
            repeat (int): Number of times to repeat the dataset, default is 1.
            p (float): Probability of random dropout, default is 0.5.
            window_size (int): Window size, default is -1.
            debug (bool): Whether to enable debug mode, default is False.
            partition (bool): Whether to perform body part partition, default is False (not used for 17 joints).
            data_path (str): Path to data pkl file (e.g., 'finetune_train.pkl')
            label_path (str): Path to label pkl file (e.g., 'finetune_train_label.pkl')
        """

        self.split = split
        self.data_path = data_path
        self.label_path = label_path
        
        # Set time steps
        self.time_steps = 64  # Temporal length
        self.label = []

        # Save initialization parameters
        self.data_type = data_type
        self.partition = partition
        self.repeat = repeat
        self.p = p

        # 17-joint skeleton bone connections (H36M style)
        self.bone = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Right arm
            (8, 14), (14, 15), (15, 16),  # Left arm
        ]

        # Center joint for 17-joint skeleton (joint 0 is typically pelvis/root)
        self.center_joint = 0

        # Load data
        self.load_data()

    def load_data(self):
        """
        Load skeleton information for all data samples.
        Data format: N C V T M
        N: number of samples
        C: coordinate dimension (3 for x,y,z)
        V: number of joints (17)
        T: temporal length
        M: number of persons
        """
        # Use custom paths if provided, otherwise use default paths
        if self.data_path and self.label_path:
            label_file = self.label_path
            data_file = self.data_path
        else:
            # Default paths based on split
            split_map = {
                'train': ('finetune_train_label.pkl', 'finetune_train.pkl'),
                'test': ('finetune_test_label.pkl', 'finetune_test.pkl'),
                'val': ('finetune_test_label.pkl', 'finetune_test.pkl'),  # Use test as val
            }
            
            label_file, data_file = split_map.get(self.split, split_map['train'])

        print(f"Loading data from: {data_file}")
        print(f"Loading labels from: {label_file}")

        # Load label file
        with open(label_file, 'rb') as f:
            self.data_dict = pickle.load(f)
        
        # Load data file
        with open(data_file, 'rb') as f:
            self.all = pickle.load(f)

        # Create keypoint dictionary
        self.keypoint = {i['frame_dir']: i['keypoint'][0] for i in self.all['annotations']}

        # Build data and label lists
        self.data, self.label = [], []
        for info in tqdm(self.data_dict, ncols=100, desc=f"Loading {self.split} data"):
            self.data.append(self.keypoint[info['file_name']])
            self.label.append(int(info['label']))

        print(f"Loaded {len(self.data)} samples with {len(set(self.label))} unique classes")

    def __len__(self):
        """
        Return dataset length, considering repeat count.
        """
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        """
        Return the iterator itself.
        """
        return self

    def __getitem__(self, index):
        """
        Get a single data sample.

        Args:
            index (int): Data index.

        Returns:
            tuple: (data, index_t, label, index)
                - data: Processed skeleton data (C, T, V, 1)
                - index_t: Temporal position encoding
                - label: Sample label
                - index: Sample index
        """
        # Handle repeat: get actual index in data list
        index = index % len(self.data_dict)
        
        # Get data and label
        data = self.data[index].copy()  # Shape: (T, V, C) where T=frames, V=17, C=3
        label = self.label[index]

        # Data augmentation: random dropout (only during training)
        if 'train' in self.split and self.p > 0:
            # Randomly set some frames to zero
            T, V, C = data.shape
            mask = np.random.rand(T, V, 1) > self.p
            data = data * mask

        # Temporal sampling to fixed length
        length = data.shape[0]
        
        if length <= self.time_steps:
            # If sequence is shorter, pad with zeros
            data_pad = np.zeros((self.time_steps, data.shape[1], data.shape[2]))
            data_pad[:length, :, :] = data
            data = data_pad
            # Create temporal index (for positional encoding)
            idx = np.arange(length)
            idx_pad = np.full(self.time_steps, length - 1)
            idx_pad[:length] = idx
            index_t = 2 * idx_pad.astype(np.float32) / max(length - 1, 1) - 1
        else:
            # If sequence is longer, sample uniformly
            idx = np.round(np.linspace(0, length - 1, self.time_steps)).astype(int)
            data = data[idx, :, :]
            index_t = 2 * idx.astype(np.float32) / (length - 1) - 1

        # Data type conversion based on data_type
        if 'b' in self.data_type:
            # Compute relative positions between bones
            data_bone = np.zeros_like(data)
            for bone_idx in range(len(self.bone)):
                # Compute each bone vector (start joint - end joint)
                data_bone[:, self.bone[bone_idx][0], :] = (
                        data[:, self.bone[bone_idx][0], :] - data[:, self.bone[bone_idx][1], :])
            data = data_bone

        if 'm' in self.data_type:
            # Compute motion change (difference between current and next frame)
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        # Transpose data dimensions to (C, T, V)
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        # Add an extra dimension, shape becomes (C, T, V, 1)
        data = np.reshape(data, (C, T, V, 1))

        return data, index_t, label, index

    def top_k(self, score, top_k):
        """
        Compute top-k accuracy.

        Args:
            score (ndarray): Model prediction scores.
            top_k (int): The k value.

        Returns:
            float: top-k accuracy.
        """
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    """
    Dynamically import a class by name.

    Args:
        name (str): Full class path, e.g. 'module.submodule.ClassName'.

    Returns:
        class: The imported class object.
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

