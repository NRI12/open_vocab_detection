import sys
sys.path.append('src')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from data.datamodule import Flickr30kDataModule

if __name__ == "__main__":
    dm = Flickr30kDataModule(data_dir='./data', batch_size=4, img_size=(800, 600))
    dm.setup('fit')

    # Get 1 batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print("=== BATCH INFO ===")
    print(f"Batch size: {len(batch['images'])}")
    print(f"Image shape: {batch['images'][0].shape}")
    print(f"Sample text: {batch['texts'][0]}")
    print(f"Sample boxes: {batch['boxes'][0].shape}")
    print(f"Sample labels: {batch['labels'][0]}")

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    img = batch['images'][0]  # (3, H, W)
    boxes = batch['boxes'][0]  # (N, 4)
    labels = batch['labels'][0]  # List of strings
    text = batch['texts'][0]

    # Denormalize và convert to numpy
    img_denorm = denormalize(img)
    img_np = img_denorm.permute(1, 2, 0).clamp(0, 1).numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_np)

    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        
        # Draw bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Draw label
        ax.text(x1, y1-10, label, color=color, fontsize=10, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    ax.set_title(f"DataLoader Test - {batch['img_ids'][0]}")
    ax.axis('off')

    plt.figtext(0.02, 0.02, f"Text: {text[:100]}...", fontsize=8, 
            bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('dataloader_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"=== SAVED: dataloader_test.png ===")
    print(f"Image ID: {batch['img_ids'][0]}")
    print(f"Boxes count: {len(boxes)}")
    print(f"Labels: {labels}")