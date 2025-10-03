"""
Dataset fetching utilities for various computer vision datasets.
"""

import os
import requests
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
import numpy as np


class KITTIDatasetFetcher:
    """Utility class for downloading and managing KITTI Visual Odometry dataset."""
    
    BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_"
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize KITTI dataset fetcher.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_sequence(self, sequence: str = "00", 
                         download_images: bool = True,
                         download_poses: bool = True) -> Path:
        """
        Download a specific KITTI sequence.
        
        Args:
            sequence: Sequence number (00-21)
            download_images: Whether to download image data
            download_poses: Whether to download ground truth poses
            
        Returns:
            Path to the downloaded sequence directory
        """
        seq_dir = self.data_dir / "kitti" / f"sequence_{sequence}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Check if data is already extracted ---
        # The final directory for images is the most reliable indicator
        final_image_dir = seq_dir / "dataset" / "sequences" / sequence / "image_0"
        if final_image_dir.exists():
            print(f"Dataset for sequence '{sequence}' already exists. Skipping download.")
            return seq_dir
        
        if download_images:
            self._download_images(sequence, seq_dir)
            
        if download_poses:
            self._download_poses(seq_dir)
            
        # Download calibration files
        self._download_calibration(seq_dir)
        
        return seq_dir
    
    def _download_file(self, url: str, filepath: Path, 
                      description: str = "Downloading") -> None:
        """Download a file with progress bar."""
        if filepath.exists():
            print(f"File {filepath.name} already exists, skipping download.")
            return
            
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _download_images(self, sequence: str, seq_dir: Path) -> None:
        """Download image data for a sequence."""
        # KITTI sequences 00-10 are in training set, 11-21 in testing
        if int(sequence) <= 10:
            dataset_type = "gray"  # Training sequences
            zip_name = f"data_odometry_{dataset_type}.zip"
        else:
            dataset_type = "gray"  # Test sequences  
            zip_name = f"data_odometry_{dataset_type}.zip"
            
        zip_path = seq_dir / zip_name
        url = f"{self.BASE_URL}{dataset_type}.zip"
        
        print(f"Downloading KITTI sequence {sequence} images...")
        self._download_file(url, zip_path, f"Downloading {zip_name}")
        
        # Extract specific sequence
        print(f"Extracting sequence {sequence}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only the specific sequence we need
            for member in zip_ref.namelist():
                if f"sequences/{sequence}/" in member:
                    zip_ref.extract(member, seq_dir)
        
        # Clean up zip file
        zip_path.unlink()
    
    def _download_poses(self, seq_dir: Path) -> None:
        """Download ground truth poses."""
        poses_url = f"{self.BASE_URL}poses.zip"
        poses_zip = seq_dir / "poses.zip"
        
        print("Downloading ground truth poses...")
        self._download_file(poses_url, poses_zip, "Downloading poses")
        
        # Extract poses
        with zipfile.ZipFile(poses_zip, 'r') as zip_ref:
            zip_ref.extractall(seq_dir)
            
        poses_zip.unlink()
    
    def _download_calibration(self, seq_dir: Path) -> None:
        """Download calibration files."""
        calib_url = f"{self.BASE_URL}calib.zip"
        calib_zip = seq_dir / "calib.zip"
        
        print("Downloading calibration files...")
        self._download_file(calib_url, calib_zip, "Downloading calibration")
        
        # Extract calibration
        with zipfile.ZipFile(calib_zip, 'r') as zip_ref:
            zip_ref.extractall(seq_dir)
            
        calib_zip.unlink()
    
    def load_poses(self, sequence: str = "00") -> np.ndarray:
        """
        Load ground truth poses for a sequence.
        
        Args:
            sequence: Sequence number
            
        Returns:
            Array of 4x4 transformation matrices
        """
        seq_dir = self.data_dir / "kitti" / f"sequence_{sequence}"
        poses_file = seq_dir / "dataset" / "poses" / f"{sequence}.txt"
        
        if not poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {poses_file}")
            
        poses_data = np.loadtxt(poses_file)
        
        # Convert 12-element vectors to 4x4 matrices
        poses = []
        for pose_vec in poses_data:
            pose_matrix = np.eye(4)
            pose_matrix[:3, :] = pose_vec.reshape(3, 4)
            poses.append(pose_matrix)
            
        return np.array(poses)
    
    def load_calibration(self, sequence: str = "00") -> dict:
        """
        Load calibration parameters for a sequence.
        
        Args:
            sequence: Sequence number
            
        Returns:
            Dictionary containing calibration matrices
        """
        seq_dir = self.data_dir / "kitti" / f"sequence_{sequence}"
        calib_file = seq_dir / "dataset" / "sequences" / sequence / "calib.txt"
        
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
            
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f:
                if line.strip():
                    key, values = line.strip().split(':', 1)
                    calib[key] = np.array([float(x) for x in values.split()])
                    
        # Reshape projection matrices
        for key in ['P0', 'P1', 'P2', 'P3']:
            if key in calib:
                calib[key] = calib[key].reshape(3, 4)
                
        return calib
    
    def get_image_paths(self, sequence: str = "00", camera: str = "image_0") -> list:
        """
        Get list of image file paths for a sequence.
        
        Args:
            sequence: Sequence number
            camera: Camera name (image_0 or image_1 for stereo)
            
        Returns:
            List of image file paths
        """
        seq_dir = self.data_dir / "kitti" / f"sequence_{sequence}"
        img_dir = seq_dir / "dataset" / "sequences" / sequence / camera
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        img_paths = sorted(img_dir.glob("*.png"))
        return [str(p) for p in img_paths]


def download_kitti_sequence(sequence: str = "00", data_dir: str = "data") -> Path:
    """
    Convenience function to download a KITTI sequence.
    
    Args:
        sequence: Sequence number (00-21)
        data_dir: Directory to store data
        
    Returns:
        Path to downloaded sequence
    """
    fetcher = KITTIDatasetFetcher(data_dir)
    return fetcher.download_sequence(sequence)


if __name__ == "__main__":
    # Example usage
    print("Downloading KITTI sequence 00...")
    seq_path = download_kitti_sequence("00")
    print(f"Downloaded to: {seq_path}")
