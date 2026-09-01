import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os
import argparse

# Canonical FER-2013 emotion mapping (verified against official ICML 2013 dataset specifications)
FER2013_EMOTION_MAP = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'sad',
    5: 'surprised',
    6: 'neutral'
}

def prepare_dataset(csv_path='./fer2013.csv', output_dir='data'):
    """
    Parses FER2013 dataset CSV and saves images into structured folders.
    - Uses modern numpy array conversion from split string (avoiding np.fromstring deprecation).
    - Writes correctly to output_dir/train and output_dir/test.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found at: {csv_path}")

    # Create destination directories under output_dir/train and output_dir/test
    outer_names = ['train', 'test']
    for outer in outer_names:
        for inner in FER2013_EMOTION_MAP.values():
            os.makedirs(os.path.join(output_dir, outer, inner), exist_ok=True)

    # Initialize counters for each emotion category
    counts = {
        'train': {emotion: 0 for emotion in FER2013_EMOTION_MAP.values()},
        'test': {emotion: 0 for emotion in FER2013_EMOTION_MAP.values()}
    }

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    train_cutoff = 28709  # First 28709 rows are training, remaining are test

    print(f"Saving {total_samples} images to '{output_dir}' directory...")

    # Fast iteration using modern split + numpy array conversion
    for i in tqdm(range(total_samples)):
        split = 'train' if i < train_cutoff else 'test'
        emotion_id = int(df['emotion'].iloc[i])
        
        # Verify emotion ID is in valid FER2013 range 0..6
        assert 0 <= emotion_id <= 6, f"Invalid emotion ID {emotion_id} at row {i}"
        emotion_name = FER2013_EMOTION_MAP[emotion_id]

        # Fast parsing via split + uint8 array reshaping (modern, deprecation-free)
        pixels_str = df['pixels'].iloc[i]
        pixels_arr = np.array(pixels_str.split(), dtype=np.uint8).reshape((48, 48))

        img = Image.fromarray(pixels_arr)
        count = counts[split][emotion_name]
        save_path = os.path.join(output_dir, split, emotion_name, f'im{count}.png')
        img.save(save_path)
        counts[split][emotion_name] += 1

    print("\nDataset preparation completed successfully!")
    print(f"Training samples: {sum(counts['train'].values())}")
    print(f"Testing samples:  {sum(counts['test'].values())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract FER-2013 CSV to PNG image dataset")
    parser.add_argument('--csv', default='./fer2013.csv', help='Path to fer2013.csv file')
    parser.add_argument('--output', default='data', help='Output directory for train and test folders')
    args = parser.parse_args()

    prepare_dataset(csv_path=args.csv, output_dir=args.output)