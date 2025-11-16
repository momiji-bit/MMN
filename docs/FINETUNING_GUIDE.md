# Finetuning Guide - MMN Model on Finetune Dataset

## Overview

This guide explains how to finetune a pretrained MMN checkpoint (trained on MA52 dataset) on your new finetune dataset.

**Pretrained Model:**
- Checkpoint: `runs-120-21000.pt`
- Trained with: `MMN/config/train/MA52_J_3D.yaml`
- Architecture: MMN with 17 joints, 3D coordinates, 52 classes

**Finetune Dataset:**
- Train: `finetune_train.pkl` + `finetune_train_label.pkl` (253 samples)
- Test: `finetune_test.pkl` + `finetune_test_label.pkl` (28 samples)
- Format: H36M 17-keypoint (1, T, 17, 3)
- Classes: 42 unique labels (1-52 range)

---

## Quick Start

### 1. Setup Files

**Files Created:**
```
MMN/
├── feeders/
│   └── feeder_finetune.py          ← Custom feeder for finetune dataset
└── config/
    └── train/
        └── MA52_J_3D_finetune.yaml ← Finetuning configuration

Root directory:
├── finetune_train.pkl              ← Training data
├── finetune_train_label.pkl        ← Training labels
├── finetune_test.pkl               ← Test data
└── finetune_test_label.pkl         ← Test labels
```

### 2. Update Configuration

**Edit `MMN/config/train/MA52_J_3D_finetune.yaml`:**

```yaml
# Line 7: Update checkpoint path
weights: ./work_dir/train/MA52_J_3D/runs-120-21000.pt

# If your checkpoint is elsewhere, use full path:
# weights: /home/shanaka/Documents/skeleton/path/to/runs-120-21000.pt
```

### 3. Run Finetuning

```bash
cd /home/shanaka/Documents/skeleton/MMN

# Start finetuning
python main.py --config config/train/MA52_J_3D_finetune.yaml
```

---

## Configuration Details

### Key Parameters Explained

#### 1. **Checkpoint Loading**

```yaml
weights: ./work_dir/train/MA52_J_3D/runs-120-21000.pt
ignore_weights: []
```

**Options:**
- `ignore_weights: []` - Load ALL weights (recommended)
  - Use this when finetuning on similar task
  - Pretrained features are valuable
  
- `ignore_weights: ['head']` - Reinitialize only classifier head
  - Use if num_classes changed significantly
  - Keeps backbone frozen initially

#### 2. **Learning Rate (Critical!)**

```yaml
base_lr: 5e-5      # 20x lower than original (1e-3)
min_lr: 1e-6       # Lower minimum
warm_up_epoch: 5   # Shorter warmup
```

**Why lower LR?**
- Prevents catastrophic forgetting
- Preserves pretrained features
- Allows fine adjustments without destroying learned representations

**Rule of thumb:**
- Finetuning LR = 1/10 to 1/100 of original training LR
- Start conservative (lower) and increase if needed

#### 3. **Training Duration**

```yaml
num_epoch: 60      # Half of original (120)
warm_up_epoch: 5   # Shorter warmup (was 20)
```

**Why fewer epochs?**
- Finetuning converges faster
- Model already has good representations
- Risk of overfitting on small dataset

#### 4. **Dataset Configuration**

```yaml
train_feeder_args:
  data_path: 'finetune_train.pkl'
  label_path: 'finetune_train_label.pkl'
  
test_feeder_args:
  data_path: 'finetune_test.pkl'
  label_path: 'finetune_test_label.pkl'
```

**Paths can be:**
- Relative: `'finetune_train.pkl'` (if in MMN directory)
- Absolute: `'/home/shanaka/Documents/skeleton/finetune_train.pkl'`

#### 5. **Model Architecture**

```yaml
model_args:
  in_channels: 3     # Must match checkpoint
  num_classes: 42    # Your finetune dataset classes
  num_people: 1      # Must match checkpoint
  num_points: 17     # Must match checkpoint
```

**IMPORTANT:** All model_args except `num_classes` must match the pretrained checkpoint!

---

## Advanced Options

### Option 1: Freeze Backbone, Train Only Head

**Use Case:** Very small dataset or domain shift

**Modification:** In `main.py` after loading weights:

```python
# Freeze all parameters
for param in self.model.parameters():
    param.requires_grad = False

# Unfreeze only the classification head
for param in self.model.head.parameters():
    param.requires_grad = True
```

**Or in config:**
```yaml
ignore_weights: ['head']  # Reinitialize head
# Then manually freeze backbone in code
```

### Option 2: Gradual Unfreezing

**Strategy:** Start with frozen backbone, gradually unfreeze layers

```python
# Epoch 1-10: Only head
# Epoch 11-20: Unfreeze last transformer block
# Epoch 21+: Unfreeze all
```

### Option 3: Different Learning Rates per Layer

**Use Case:** Fine-grained control

```python
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-4}
])
```

### Option 4: Early Stopping

**Prevent overfitting on small dataset**

Monitor validation loss and stop when it stops improving for N epochs.

---

## Running Finetuning

### Basic Command

```bash
cd /home/shanaka/Documents/skeleton/MMN
python main.py --config config/train/MA52_J_3D_finetune.yaml
```

### With GPU Selection

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py --config config/train/MA52_J_3D_finetune.yaml

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python main.py --config config/train/MA52_J_3D_finetune.yaml
```

### Monitor Training

```bash
# In another terminal
cd /home/shanaka/Documents/skeleton/MMN
tensorboard --logdir work_dir/finetune/MA52_J_3D/runs
```

Open browser: http://localhost:6006

---

## Monitoring & Evaluation

### Checkpoints Saved

```
work_dir/finetune/MA52_J_3D/
├── runs-1-XXX.pt          # Epoch 1 checkpoints
├── runs-2-XXX.pt          # Epoch 2 checkpoints
├── ...
└── runs/                  # TensorBoard logs
    ├── train/
    └── val/
```

### Key Metrics to Watch

1. **Training Loss** - Should decrease steadily
2. **Validation Accuracy** - Should improve then plateau
3. **Learning Rate** - Should decay smoothly
4. **Overfitting Signs:**
   - Train acc increases, val acc decreases
   - Large gap between train and val accuracy

### Expected Performance

**Small Dataset (281 samples):**
- Training may converge in 20-40 epochs
- Validation accuracy might plateau early
- Risk of overfitting due to limited data

**Recommendations:**
- Use data augmentation (already enabled with `p: 0.2`)
- Monitor validation closely
- Consider early stopping
- Try ensemble of checkpoints

---

## Troubleshooting

### Issue 1: "Checkpoint file not found"

**Error:**
```
FileNotFoundError: ./work_dir/train/MA52_J_3D/runs-120-21000.pt
```

**Solution:**
```yaml
# Use absolute path in config
weights: /home/shanaka/Documents/skeleton/path/to/checkpoint.pt
```

### Issue 2: "Size mismatch for head.weight"

**Error:**
```
RuntimeError: size mismatch for head.weight: copying from (52, X), but (42, X) expected
```

**Solution:**
The pretrained model has 52 classes, but finetune has 42. Two options:

**Option A:** Ignore head weights (reinitialize classifier)
```yaml
ignore_weights: ['head']
```

**Option B:** Pad/truncate the head weights (advanced)
Manually adjust in code.

### Issue 3: "Model not learning / Loss not decreasing"

**Causes:**
1. Learning rate too low
2. Batch size too small
3. All layers frozen by mistake

**Solutions:**
```yaml
# Try higher learning rate
base_lr: 1e-4  # Instead of 5e-5

# Increase batch size
batch_size: 64  # Instead of 32

# Check that weights are loaded correctly
# Look for "Load weights from..." message in logs
```

### Issue 4: "Out of memory (OOM)"

**Solutions:**
```yaml
# Reduce batch size
batch_size: 16  # or 8

# Reduce sequence length in feeder
# Edit feeder_finetune.py line 26
self.time_steps = 32  # Instead of 64
```

### Issue 5: "Overfitting quickly"

**Symptoms:**
- Train acc: 95%+, Val acc: 60%

**Solutions:**
```yaml
# Increase dropout
model_args:
  drop: 0.2       # Was 0.0
  head_drop: 0.3  # Was 0.1
  drop_path: 0.5  # Was 0.3

# Increase data augmentation
train_feeder_args:
  p: 0.5  # Was 0.2

# Use weight decay
weight_decay: 0.1  # Was 0.05

# Early stopping after N epochs without improvement
```

### Issue 6: "Different number of classes"

If your checkpoint has different num_classes than config:

**Check checkpoint:**
```python
import torch
ckpt = torch.load('runs-120-21000.pt')
print(ckpt['head.weight'].shape)  # Should show (52, X) or similar
```

**Options:**
1. Use `ignore_weights: ['head']` to reinitialize
2. Modify finetune dataset to match (not recommended)
3. Use transfer learning approach (freeze backbone)

---

## Class Imbalance Handling

Your finetune dataset has **high class imbalance** (36:1 ratio).

### Solution 1: Class Weights

**Modify `main.py` to add class weights:**

```python
# After loading data
from sklearn.utils.class_weight import compute_class_weight

labels = [label for _, _, label, _ in train_loader.dataset]
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = torch.FloatTensor(class_weights).cuda()

# Use in loss function
self.loss = nn.CrossEntropyLoss(weight=class_weights)
```

### Solution 2: Weighted Sampling

**Use weighted random sampler:**

```python
from torch.utils.data import WeightedRandomSampler

# Compute sample weights
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Use in DataLoader
train_loader = DataLoader(..., sampler=sampler)
```

### Solution 3: Focal Loss

**Replace CrossEntropy with Focal Loss** (handles imbalance better):

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

---

## Best Practices

### 1. Start Conservative

```yaml
base_lr: 1e-5      # Very low
num_epoch: 30      # Short
batch_size: 32     # Moderate
```

Run for 30 epochs and evaluate. If underfitting, increase LR.

### 2. Use Validation Split

With only 281 samples, consider k-fold cross-validation:

```python
# Instead of fixed train/test, do 5-fold CV
from sklearn.model_selection import StratifiedKFold
```

### 3. Data Augmentation

Already enabled, but can increase:

```yaml
train_feeder_args:
  p: 0.5  # More aggressive dropout
```

### 4. Ensemble Models

Train multiple models and ensemble:
- Different random seeds
- Different checkpoints (epoch 40, 50, 60)
- Average predictions

### 5. Monitor Per-Class Metrics

Not just overall accuracy - check performance on each class:

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

---

## Example Training Session

```bash
# 1. Navigate to MMN directory
cd /home/shanaka/Documents/skeleton/MMN

# 2. Verify files exist
ls -lh ../finetune_*.pkl
ls -lh feeders/feeder_finetune.py
ls -lh config/train/MA52_J_3D_finetune.yaml

# 3. Update checkpoint path in config
nano config/train/MA52_J_3D_finetune.yaml
# Edit line 7: weights: /path/to/runs-120-21000.pt

# 4. Start training
python main.py --config config/train/MA52_J_3D_finetune.yaml

# 5. In another terminal, monitor with tensorboard
tensorboard --logdir work_dir/finetune/MA52_J_3D/runs

# 6. Open browser to http://localhost:6006
```

**Expected Output:**
```
Load weights from ./work_dir/train/MA52_J_3D/runs-120-21000.pt
Loading data from: finetune_train.pkl
Loading labels from: finetune_train_label.pkl
Loading train data: 100%|██████████| 253/253
Loaded 253 samples with 41 unique classes

Loading data from: finetune_test.pkl
Loading labels from: finetune_test_label.pkl
Loading test data: 100%|██████████| 28/28
Loaded 28 samples with 18 unique classes

Epoch: 1/60
Train: 100%|██████████| 8/8 [00:10<00:00,  1.25s/it]
Loss: 3.2145, Acc: 0.2341

Val: 100%|██████████| 1/1 [00:01<00:00,  1.05s/it]
Val Loss: 3.1234, Val Acc: 0.2857

...
```

---

## Resuming Training

If training stops, resume from checkpoint:

```yaml
# Edit config
weights: ./work_dir/finetune/MA52_J_3D/runs-30-XXX.pt  # Latest checkpoint
start_epoch: 30  # Start from epoch 31
```

---

## Testing Finetuned Model

After training, test the best checkpoint:

**Create test config:** `config/test/MA52_J_3D_finetune_test.yaml`

```yaml
phase: test
weights: ./work_dir/finetune/MA52_J_3D/runs-best.pt

feeder: feeders.feeder_finetune.Feeder
test_feeder_args:
  split: 'test'
  data_type: 'j'
  data_path: 'finetune_test.pkl'
  label_path: 'finetune_test_label.pkl'
```

**Run test:**
```bash
python main.py --config config/test/MA52_J_3D_finetune_test.yaml
```

---

## Summary

### Files Created
1. `MMN/feeders/feeder_finetune.py` - Custom feeder
2. `MMN/config/train/MA52_J_3D_finetune.yaml` - Finetuning config
3. `FINETUNING_GUIDE.md` - This guide

### Steps to Finetune
1. ✓ Update checkpoint path in config
2. ✓ Ensure dataset files are accessible
3. ✓ Run: `python main.py --config config/train/MA52_J_3D_finetune.yaml`
4. ✓ Monitor with tensorboard
5. ✓ Evaluate best checkpoint

### Key Considerations
- **Lower learning rate** (5e-5 vs 1e-3)
- **Fewer epochs** (60 vs 120)
- **Class imbalance** (use weights/sampling)
- **Small dataset** (careful of overfitting)
- **Monitor closely** (validation metrics)

---

## Questions?

**Common Q&A:**

**Q: How long will training take?**
A: With 253 samples, ~10-20 minutes per epoch on GPU (batch_size=32). Total: 10-20 hours for 60 epochs.

**Q: What accuracy should I expect?**
A: With 42 classes and 281 samples, 50-70% accuracy is realistic. Class imbalance may lower performance.

**Q: Should I train from scratch or finetune?**
A: Finetune! With only 281 samples, pretrained features are invaluable.

**Q: Can I finetune on multiple checkpoints?**
A: Yes! Try different checkpoints (runs-100-XXX.pt, runs-110-XXX.pt) and compare.

---

*Finetuning Guide - November 16, 2025*
*For MMN Model on H36M 17-keypoint Dataset*

