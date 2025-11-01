import os
import pickle
import numpy as np
from pathlib import Path

def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                video_name = parts[0].replace('.mp4', '')
                label = int(parts[1])
                labels[video_name] = label
    return labels

def create_train_3d_pkl(train_dir, label_file, output_file):
    print("Loading labels...")
    labels = load_labels(label_file)
    
    print("Scanning train folders...")
    folders = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    print(f"Found {len(folders)} folders")
    
    annotations = []
    split_train = []
    
    print("Processing folders...")
    for idx, folder in enumerate(folders, 1):
        folder_path = os.path.join(train_dir, folder)
        pose_3d_path = os.path.join(folder_path, 'pose_3d.npy')
        
        if idx % 100 == 0 or idx == len(folders):
            print(f"  Processing {idx}/{len(folders)}...")
        
        if not os.path.exists(pose_3d_path):
            print(f"Warning: {pose_3d_path} not found, skipping...")
            continue
        
        if folder not in labels:
            print(f"Warning: No label found for {folder}, skipping...")
            continue
        
        pose_3d = np.load(pose_3d_path)
        
        num_frames = pose_3d.shape[0]
        num_keypoints = pose_3d.shape[1]
        
        keypoint = pose_3d.reshape(1, num_frames, num_keypoints, 3)
        keypoint_score = np.ones((1, num_frames, num_keypoints), dtype=np.float64)
        
        annotation = {
            'keypoint': keypoint,
            'keypoint_score': keypoint_score,
            'frame_dir': folder,
            'img_shape': (1080, 900),
            'original_shape': (1080, 900),
            'total_frames': num_frames,
            'label': labels[folder]
        }
        
        annotations.append(annotation)
        split_train.append(folder)
    
    data = {
        'split': {'train': split_train},
        'annotations': annotations
    }
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Successfully created {output_file}")
    print(f"Total samples: {len(annotations)}")
    print(f"Sample annotation keys: {list(annotations[0].keys())}")
    print(f"Keypoint shape: {annotations[0]['keypoint'].shape}")
    print(f"Keypoint score shape: {annotations[0]['keypoint_score'].shape}")

if __name__ == '__main__':
    train_dir = 'data/train'
    label_file = 'data/train_labels.txt'
    output_file = 'train.pkl'
    
    create_train_3d_pkl(train_dir, label_file, output_file)

