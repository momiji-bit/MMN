import os
import pickle
import numpy as np
from pathlib import Path
import sys

class TestDataValidator:
    def __init__(self, pkl_file, test_dir, label_file):
        self.pkl_file = pkl_file
        self.test_dir = test_dir
        self.label_file = label_file
        self.data = None
        self.labels_from_file = {}
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
    def log_error(self, message):
        self.errors.append(f"❌ ERROR: {message}")
        
    def log_warning(self, message):
        self.warnings.append(f"⚠️  WARNING: {message}")
        
    def log_pass(self, message):
        self.passed_checks.append(f"✓ {message}")
    
    def print_section(self, title):
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    
    def validate_file_existence(self):
        self.print_section("1. FILE EXISTENCE VALIDATION")
        
        if not os.path.exists(self.pkl_file):
            self.log_error(f"{self.pkl_file} does not exist")
            return False
        self.log_pass(f"{self.pkl_file} exists")
        
        if not os.path.exists(self.test_dir):
            self.log_error(f"{self.test_dir} does not exist")
            return False
        self.log_pass(f"{self.test_dir} exists")
        
        if not os.path.exists(self.label_file):
            self.log_error(f"{self.label_file} does not exist")
            return False
        self.log_pass(f"{self.label_file} exists")
        
        return True
    
    def load_data(self):
        self.print_section("2. DATA LOADING")
        
        try:
            with open(self.pkl_file, 'rb') as f:
                self.data = pickle.load(f)
            self.log_pass(f"Successfully loaded {self.pkl_file}")
        except Exception as e:
            self.log_error(f"Failed to load {self.pkl_file}: {str(e)}")
            return False
        
        try:
            with open(self.label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        video_name = parts[0].replace('.mp4', '')
                        label = int(parts[1])
                        self.labels_from_file[video_name] = label
            self.log_pass(f"Successfully loaded {len(self.labels_from_file)} labels from {self.label_file}")
        except Exception as e:
            self.log_error(f"Failed to load {self.label_file}: {str(e)}")
            return False
        
        return True
    
    def validate_structure(self):
        self.print_section("3. STRUCTURE VALIDATION")
        
        if not isinstance(self.data, dict):
            self.log_error(f"Data should be dict, got {type(self.data)}")
            return False
        self.log_pass("Data is a dictionary")
        
        required_keys = ['split', 'annotations']
        for key in required_keys:
            if key not in self.data:
                self.log_error(f"Missing required key: '{key}'")
            else:
                self.log_pass(f"Key '{key}' exists")
        
        if 'split' in self.data:
            if not isinstance(self.data['split'], dict):
                self.log_error(f"'split' should be dict, got {type(self.data['split'])}")
            elif 'test' not in self.data['split']:
                self.log_error("'split' missing 'test' key")
            else:
                self.log_pass(f"Split has 'test' key with {len(self.data['split']['test'])} samples")
        
        if 'annotations' in self.data:
            if not isinstance(self.data['annotations'], list):
                self.log_error(f"'annotations' should be list, got {type(self.data['annotations'])}")
            else:
                self.log_pass(f"Annotations is a list with {len(self.data['annotations'])} items")
        
        return len(self.errors) == 0
    
    def validate_annotations(self):
        self.print_section("4. ANNOTATIONS VALIDATION")
        
        required_fields = ['keypoint', 'keypoint_score', 'frame_dir', 'img_shape', 
                          'original_shape', 'total_frames', 'label']
        
        for idx, annotation in enumerate(self.data['annotations']):
            if not isinstance(annotation, dict):
                self.log_error(f"Annotation {idx} is not a dict")
                continue
            
            for field in required_fields:
                if field not in annotation:
                    self.log_error(f"Annotation {idx} missing field '{field}'")
        
        if len(self.errors) == 0:
            self.log_pass(f"All {len(self.data['annotations'])} annotations have required fields")
        
        return True
    
    def validate_keypoint_data(self):
        self.print_section("5. KEYPOINT DATA VALIDATION")
        
        for idx, annotation in enumerate(self.data['annotations']):
            keypoint = annotation.get('keypoint')
            keypoint_score = annotation.get('keypoint_score')
            
            if keypoint is None or keypoint_score is None:
                continue
            
            if not isinstance(keypoint, np.ndarray):
                self.log_error(f"Annotation {idx}: keypoint is not numpy array")
                continue
            
            if not isinstance(keypoint_score, np.ndarray):
                self.log_error(f"Annotation {idx}: keypoint_score is not numpy array")
                continue
            
            if len(keypoint.shape) != 4:
                self.log_error(f"Annotation {idx}: keypoint should have 4 dimensions, got {len(keypoint.shape)}")
            
            if len(keypoint_score.shape) != 3:
                self.log_error(f"Annotation {idx}: keypoint_score should have 3 dimensions, got {len(keypoint_score.shape)}")
            
            if keypoint.shape[0] != 1:
                self.log_warning(f"Annotation {idx}: keypoint first dimension is {keypoint.shape[0]}, expected 1")
            
            if keypoint.shape[3] != 3:
                self.log_error(f"Annotation {idx}: keypoint last dimension should be 3 (x,y,z), got {keypoint.shape[3]}")
            
            if keypoint.shape[:3] != keypoint_score.shape:
                self.log_error(f"Annotation {idx}: shape mismatch between keypoint {keypoint.shape[:3]} and keypoint_score {keypoint_score.shape}")
            
            total_frames = annotation.get('total_frames', 0)
            if keypoint.shape[1] != total_frames:
                self.log_error(f"Annotation {idx}: keypoint frames {keypoint.shape[1]} != total_frames {total_frames}")
            
            if np.any(np.isnan(keypoint)):
                self.log_error(f"Annotation {idx}: keypoint contains NaN values")
            
            if np.any(np.isinf(keypoint)):
                self.log_error(f"Annotation {idx}: keypoint contains Inf values")
            
            if np.any(np.isnan(keypoint_score)):
                self.log_error(f"Annotation {idx}: keypoint_score contains NaN values")
            
            if np.any(np.isinf(keypoint_score)):
                self.log_error(f"Annotation {idx}: keypoint_score contains Inf values")
            
            if np.any((keypoint_score < 0) | (keypoint_score > 1)):
                self.log_warning(f"Annotation {idx}: keypoint_score has values outside [0, 1]")
        
        if len(self.errors) == 0:
            self.log_pass("All keypoint data is valid")
        
        return True
    
    def validate_labels(self):
        self.print_section("6. LABEL VALIDATION")
        
        mismatches = []
        missing = []
        
        for idx, annotation in enumerate(self.data['annotations']):
            frame_dir = annotation.get('frame_dir', '')
            pkl_label = annotation.get('label')
            
            if frame_dir in self.labels_from_file:
                file_label = self.labels_from_file[frame_dir]
                if pkl_label != file_label:
                    mismatches.append({
                        'index': idx,
                        'frame_dir': frame_dir,
                        'pkl_label': pkl_label,
                        'file_label': file_label
                    })
            else:
                missing.append({'index': idx, 'frame_dir': frame_dir})
        
        if mismatches:
            self.log_error(f"Found {len(mismatches)} label mismatches")
            for mismatch in mismatches[:5]:
                self.log_error(f"  {mismatch['frame_dir']}: pkl={mismatch['pkl_label']}, file={mismatch['file_label']}")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches)-5} more")
        else:
            self.log_pass("All labels match test_labels.txt")
        
        if missing:
            self.log_error(f"Found {len(missing)} frame_dirs without labels in test_labels.txt")
            for m in missing[:5]:
                self.log_error(f"  Index {m['index']}: {m['frame_dir']}")
        else:
            self.log_pass("All frame_dirs have corresponding labels")
        
        return len(mismatches) == 0 and len(missing) == 0
    
    def validate_split_consistency(self):
        self.print_section("7. SPLIT CONSISTENCY VALIDATION")
        
        split_dirs = set(self.data['split']['test'])
        annotation_dirs = set([ann['frame_dir'] for ann in self.data['annotations']])
        
        if split_dirs == annotation_dirs:
            self.log_pass(f"Split and annotations match perfectly ({len(split_dirs)} samples)")
        else:
            missing_in_split = annotation_dirs - split_dirs
            missing_in_annotations = split_dirs - annotation_dirs
            
            if missing_in_split:
                self.log_error(f"{len(missing_in_split)} folders in annotations but not in split")
                for folder in list(missing_in_split)[:5]:
                    print(f"    {folder}")
            
            if missing_in_annotations:
                self.log_error(f"{len(missing_in_annotations)} folders in split but not in annotations")
                for folder in list(missing_in_annotations)[:5]:
                    print(f"    {folder}")
        
        if len(split_dirs) != len(self.data['split']['test']):
            self.log_warning("Split contains duplicate entries")
        else:
            self.log_pass("No duplicate entries in split")
        
        return split_dirs == annotation_dirs
    
    def validate_filesystem_consistency(self):
        self.print_section("8. FILESYSTEM CONSISTENCY VALIDATION")
        
        missing_folders = []
        missing_pose_files = []
        
        for annotation in self.data['annotations']:
            frame_dir = annotation['frame_dir']
            folder_path = os.path.join(self.test_dir, frame_dir)
            pose_file = os.path.join(folder_path, 'pose_3d.npy')
            
            if not os.path.exists(folder_path):
                missing_folders.append(frame_dir)
            elif not os.path.exists(pose_file):
                missing_pose_files.append(frame_dir)
        
        if missing_folders:
            self.log_error(f"{len(missing_folders)} folders referenced in pkl but missing from filesystem")
            for folder in missing_folders[:5]:
                print(f"    {folder}")
        else:
            self.log_pass("All referenced folders exist in filesystem")
        
        if missing_pose_files:
            self.log_error(f"{len(missing_pose_files)} pose_3d.npy files missing")
            for folder in missing_pose_files[:5]:
                print(f"    {folder}")
        else:
            self.log_pass("All referenced pose_3d.npy files exist")
        
        return len(missing_folders) == 0 and len(missing_pose_files) == 0
    
    def validate_data_integrity(self):
        self.print_section("9. DATA INTEGRITY VALIDATION")
        
        for idx, annotation in enumerate(self.data['annotations']):
            frame_dir = annotation['frame_dir']
            pose_file = os.path.join(self.test_dir, frame_dir, 'pose_3d.npy')
            
            if not os.path.exists(pose_file):
                continue
            
            try:
                pose_3d = np.load(pose_file)
                keypoint = annotation['keypoint']
                
                expected_shape = (pose_3d.shape[0], pose_3d.shape[1], pose_3d.shape[2])
                actual_shape = (keypoint.shape[1], keypoint.shape[2], keypoint.shape[3])
                
                if expected_shape != actual_shape:
                    self.log_error(f"Annotation {idx} ({frame_dir}): shape mismatch")
                    self.log_error(f"  File shape: {pose_3d.shape}, PKL shape: {keypoint.shape}")
                
                if not np.allclose(pose_3d, keypoint[0], rtol=1e-5, atol=1e-8):
                    self.log_error(f"Annotation {idx} ({frame_dir}): data values don't match source file")
                
            except Exception as e:
                self.log_error(f"Annotation {idx} ({frame_dir}): error loading pose file - {str(e)}")
        
        if len(self.errors) == 0:
            self.log_pass("All data matches source pose_3d.npy files")
        
        return True
    
    def validate_statistics(self):
        self.print_section("10. STATISTICAL VALIDATION")
        
        all_keypoints = []
        all_scores = []
        frame_counts = []
        label_distribution = {}
        
        for annotation in self.data['annotations']:
            keypoint = annotation['keypoint']
            keypoint_score = annotation['keypoint_score']
            
            all_keypoints.append(keypoint)
            all_scores.append(keypoint_score)
            frame_counts.append(annotation['total_frames'])
            
            label = annotation['label']
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        all_keypoints = np.concatenate([kp.flatten() for kp in all_keypoints])
        all_scores = np.concatenate([sc.flatten() for sc in all_scores])
        
        print(f"\nKeypoint Statistics:")
        print(f"  Min: {all_keypoints.min():.4f}")
        print(f"  Max: {all_keypoints.max():.4f}")
        print(f"  Mean: {all_keypoints.mean():.4f}")
        print(f"  Std: {all_keypoints.std():.4f}")
        
        print(f"\nKeypoint Score Statistics:")
        print(f"  Min: {all_scores.min():.4f}")
        print(f"  Max: {all_scores.max():.4f}")
        print(f"  Mean: {all_scores.mean():.4f}")
        print(f"  All scores = 1.0: {np.all(all_scores == 1.0)}")
        
        print(f"\nFrame Count Statistics:")
        print(f"  Min frames: {min(frame_counts)}")
        print(f"  Max frames: {max(frame_counts)}")
        print(f"  Mean frames: {np.mean(frame_counts):.2f}")
        
        print(f"\nLabel Distribution:")
        print(f"  Unique labels: {len(label_distribution)}")
        print(f"  Label range: [{min(label_distribution.keys())}, {max(label_distribution.keys())}]")
        print(f"  Most common: Label {max(label_distribution, key=label_distribution.get)} with {max(label_distribution.values())} samples")
        print(f"  Least common: Label {min(label_distribution, key=label_distribution.get)} with {min(label_distribution.values())} samples")
        
        if all_keypoints.min() < -100 or all_keypoints.max() > 100:
            self.log_warning("Keypoint coordinates seem unusual (expected normalized values)")
        else:
            self.log_pass("Keypoint coordinate ranges look reasonable")
        
        if min(frame_counts) < 10:
            self.log_warning(f"Some sequences are very short (min {min(frame_counts)} frames)")
        else:
            self.log_pass(f"All sequences have reasonable frame counts (min {min(frame_counts)})")
        
        return True
    
    def print_summary(self):
        self.print_section("VALIDATION SUMMARY")
        
        print(f"\n✓ PASSED: {len(self.passed_checks)}")
        for check in self.passed_checks:
            print(f"  {check}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORS: {len(self.errors)}")
            for error in self.errors:
                print(f"  {error}")
        
        print(f"\n{'='*60}")
        if self.errors:
            print("❌ VALIDATION FAILED")
            print(f"{'='*60}")
            return False
        elif self.warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            print(f"{'='*60}")
            return True
        else:
            print("✓ VALIDATION PASSED - ALL CHECKS SUCCESSFUL")
            print(f"{'='*60}")
            return True
    
    def run_all_validations(self):
        print("\n" + "="*60)
        print("TEST.PKL COMPREHENSIVE VALIDATION")
        print("="*60)
        
        if not self.validate_file_existence():
            self.print_summary()
            return False
        
        if not self.load_data():
            self.print_summary()
            return False
        
        self.validate_structure()
        self.validate_annotations()
        self.validate_keypoint_data()
        self.validate_labels()
        self.validate_split_consistency()
        self.validate_filesystem_consistency()
        self.validate_data_integrity()
        self.validate_statistics()
        
        return self.print_summary()

if __name__ == '__main__':
    validator = TestDataValidator(
        pkl_file='test.pkl',
        test_dir='data/test',
        label_file='data/test_labels.txt'
    )
    
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)

