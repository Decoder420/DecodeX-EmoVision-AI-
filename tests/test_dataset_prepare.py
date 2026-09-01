import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from dataset_prepare import prepare_dataset, FER2013_EMOTION_MAP

def test_prepare_dataset_structure_and_dimensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_fer.csv")
        output_dir = os.path.join(tmpdir, "output_data")

        # Create dummy FER2013 CSV with 10 rows
        samples = []
        for i in range(10):
            # 2304 pixels for 48x48
            pixels = " ".join([str(p) for p in np.random.randint(0, 256, 2304)])
            emotion = i % 7
            samples.append({"emotion": emotion, "pixels": pixels, "Usage": "Training" if i < 7 else "PrivateTest"})

        df = pd.DataFrame(samples)
        df.to_csv(csv_path, index=False)

        # Execute preparation
        prepare_dataset(csv_path=csv_path, output_dir=output_dir)

        # Assert directory structure
        assert os.path.isdir(os.path.join(output_dir, "train"))
        assert os.path.isdir(os.path.join(output_dir, "test"))

        for emo_name in FER2013_EMOTION_MAP.values():
            assert os.path.isdir(os.path.join(output_dir, "train", emo_name))
            assert os.path.isdir(os.path.join(output_dir, "test", emo_name))

        # Check saved images
        found_images = 0
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    assert img.size == (48, 48), f"Expected image size (48, 48), got {img.size}"
                    found_images += 1

        assert found_images == 10, f"Expected 10 saved images, got {found_images}"

def test_invalid_emotion_id_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "invalid_fer.csv")
        output_dir = os.path.join(tmpdir, "output_data")

        # Create row with invalid emotion id (7)
        pixels = " ".join(["100"] * 2304)
        df = pd.DataFrame([{"emotion": 7, "pixels": pixels, "Usage": "Training"}])
        df.to_csv(csv_path, index=False)

        with pytest.raises(AssertionError):
            prepare_dataset(csv_path=csv_path, output_dir=output_dir)
