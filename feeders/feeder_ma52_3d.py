import numpy as np
import random
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, split, data_type='j', repeat=1, p=0.5,
                 window_size=-1, debug=False, partition=False):
        """
        Initialize the Feeder class for loading and preprocessing 3D skeleton data (17 joints).

        Args:
            split (str): Path to the label file.
            data_type (str): Data type, default is 'j'.
            repeat (int): Number of times to repeat the dataset, default is 1.
            p (float): Probability of random dropout, default is 0.5.
            window_size (int): Window size, default is -1.
            debug (bool): Whether to enable debug mode, default is False.
            partition (bool): Whether to perform body part partition, default is False.
        """

        self.split = split
        # Set time steps
        self.time_steps = 64  # Temporal length
        self.label = []

        # Save initialization parameters
        self.data_type = data_type
        self.partition = partition
        self.repeat = repeat
        self.p = p

        # 17-joint skeleton bone connections (typical COCO/H36M style)
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
        split_map = {
            'train': ('data/MA52/train_label.pkl', 'data/MA52/train_data.pkl'),
            'val': ('data/MA52/val_label.pkl', 'data/MA52/val_data.pkl'),
            'test': ('data/MA52/test_label.pkl', 'data/MA52/test_data.pkl'),
        }

        for key, (label_file, data_file) in split_map.items():
            if key in self.split:
                with open(label_file, 'rb') as f:
                    self.data_dict = pickle.load(f)
                with open(data_file, 'rb') as f:
                    self.all = pickle.load(f)
                break

        self.keypoint = {i['frame_dir']: i['keypoint'][0] for i in self.all['annotations']}

        self.data, self.label = [], []
        for info in tqdm(self.data_dict, ncols=100):
            self.data.append(self.keypoint[info['file_name']])
            self.label.append(int(info['label']))

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
        Retrieve a sample by index.

        Args:
            index (int): Data index.

        Returns:
            tuple: (data, temporal index, label, original index)
        """

        channel = 3  # 3D coordinates (x, y, z)
        label = self.label[index % len(self.data_dict)]  # Get label
        value = self.data[index % len(self.data_dict)]  # Get data  T,17,3

        if self.split == 'train':

            def affine_transform(value):
                T, V, C = value.shape
                assert C == 3
                # 3D rotation around y-axis (vertical)
                angle = np.random.uniform(-15, 15) * np.pi / 180
                cos_val, sin_val = np.cos(angle), np.sin(angle)
                rotation_y = np.array([
                    [cos_val, 0, sin_val],
                    [0, 1, 0],
                    [-sin_val, 0, cos_val]
                ])
                scale = np.random.uniform(0.9, 1.1)
                scale_matrix = np.eye(3) * scale
                translation = np.random.uniform(-0.1, 0.1, size=(1, 1, 3))
                value = np.matmul(value, rotation_y.T)
                value = np.matmul(value, scale_matrix)
                value += translation
                return value

            def temporal_jitter(value, max_jitter=3):
                T = value.shape[0]
                jittered = np.zeros_like(value)
                for t in range(T):
                    offset = np.random.randint(-max_jitter, max_jitter + 1)
                    new_t = min(max(t + offset, 0), T - 1)
                    jittered[t] = value[new_t]
                return jittered

            random.random()  # Initialize random seed

            # ==================== Centering ====================
            center = value[0, self.center_joint, :]
            value = value - center

            # ==================== Normalization [-1, 1] ====================
            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                    np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, value.shape[1], channel))
            value = scalerValue

            # ==================== Temporal Sampling ====================
            data = np.zeros((self.time_steps, value.shape[1], channel))
            length = value.shape[0]
            random_idx = random.sample(list(np.arange(length)) * self.time_steps, self.time_steps)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]
            index_t = 2 * np.array(random_idx).astype(np.float32) / length - 1

            # ==================== Affine Transformation ====================
            if random.random() < self.p:
                data = affine_transform(data)

            # ==================== Temporal Jitter ====================
            if random.random() < self.p:
                data = temporal_jitter(data)

            # ==================== Temporal Reversal ====================
            if random.random() < self.p:
                data = data[::-1].copy()  # .copy() to avoid negative stride

            # ==================== Axis Masking ====================
            if random.random() < self.p:
                axis_next = random.randint(0, 2)  # 0, 1, or 2 for x, y, z
                data[:, :, axis_next] = 0

            # ==================== Joint Masking ====================
            if random.random() < self.p:
                T, V, C = data.shape
                joint_count = 0
                joints_to_drop = random.sample(range(V), joint_count)
                frame_count = random.randint(1, 16)
                frames_to_drop = random.sample(range(T), frame_count)
                data[np.ix_(frames_to_drop, joints_to_drop)] = 0

            # ==================== Temporal Block Dropout ====================
            if random.random() < self.p:
                block_size = random.randint(4, 16)
                start = random.randint(0, self.time_steps - block_size)
                data[start:start + block_size, :, :] = 0


        else:
            # Test or validation set, no data augmentation
            random.random()

            center = value[0, self.center_joint, :]
            value = value - center

            scalerValue = np.reshape(value, (-1, channel))
            epsilon = 1e-6
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                    np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0) + epsilon)
            scalerValue = scalerValue * 2 - 1

            scalerValue = np.reshape(scalerValue, (-1, value.shape[1], channel))

            data = np.zeros((self.time_steps, value.shape[1], channel))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            # Use linear sampling to get fixed-length time steps
            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :] = value[idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

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

    def top_k(self, score, top_k, f=False):
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

