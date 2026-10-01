import pickle

def create_label_pkl(label_txt_file, data_pkl_file, output_pkl_file):
    """
    Create label pkl file from text file and data pkl file.
    
    Args:
        label_txt_file: Path to label text file (e.g., 'data/train_labels.txt')
        data_pkl_file: Path to data pkl file (e.g., 'train.pkl')
        output_pkl_file: Path to output label pkl file (e.g., 'train_label.pkl')
    """
    print(f"\nCreating {output_pkl_file}...")
    
    # Load labels from text file
    labels_dict = {}
    with open(label_txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                video_name = parts[0].replace('.mp4', '')
                label = int(parts[1])
                labels_dict[video_name] = label
    
    print(f"  Loaded {len(labels_dict)} labels from {label_txt_file}")
    
    # Load data pkl to get frame lengths
    with open(data_pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Loaded {len(data['annotations'])} annotations from {data_pkl_file}")
    
    # Create frame_dir to total_frames mapping
    frame_lengths = {}
    for annotation in data['annotations']:
        frame_dir = annotation['frame_dir']
        total_frames = annotation['total_frames']
        frame_lengths[frame_dir] = total_frames
    
    # Create label list in the required format
    label_list = []
    for frame_dir in sorted(labels_dict.keys()):
        if frame_dir in frame_lengths:
            entry = {
                'file_name': frame_dir,
                'length': frame_lengths[frame_dir],
                'label': labels_dict[frame_dir]
            }
            label_list.append(entry)
        else:
            print(f"  Warning: {frame_dir} not found in data pkl")
    
    # Save to pkl file
    with open(output_pkl_file, 'wb') as f:
        pickle.dump(label_list, f)
    
    print(f"  Successfully created {output_pkl_file}")
    print(f"  Total entries: {len(label_list)}")
    print(f"  Sample entries:")
    for i in range(min(3, len(label_list))):
        print(f"    {i}: {label_list[i]}")
    
    return label_list

if __name__ == '__main__':
    print("=" * 70)
    print("CREATING LABEL PKL FILES")
    print("=" * 70)
    
    # Create train_label.pkl
    train_labels = create_label_pkl(
        label_txt_file='data/train_labels.txt',
        data_pkl_file='train.pkl',
        output_pkl_file='train_label.pkl'
    )
    
    # Create test_label.pkl
    test_labels = create_label_pkl(
        label_txt_file='data/test_labels.txt',
        data_pkl_file='test.pkl',
        output_pkl_file='test_label.pkl'
    )
    
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Verify structure matches
    print("\nChecking if structure matches original format...")
    print(f"\nTrain labels:")
    print(f"  Type: {type(train_labels)}")
    print(f"  Length: {len(train_labels)}")
    print(f"  Keys in first entry: {list(train_labels[0].keys())}")
    print(f"  Expected keys: ['file_name', 'length', 'label']")
    print(f"  ✓ Match: {list(train_labels[0].keys()) == ['file_name', 'length', 'label']}")
    
    print(f"\nTest labels:")
    print(f"  Type: {type(test_labels)}")
    print(f"  Length: {len(test_labels)}")
    print(f"  Keys in first entry: {list(test_labels[0].keys())}")
    print(f"  ✓ Match: {list(test_labels[0].keys()) == ['file_name', 'length', 'label']}")
    
    print("\n" + "=" * 70)
    print("✓ LABEL PKL FILES CREATED SUCCESSFULLY")
    print("=" * 70)

