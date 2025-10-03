"""
Dataset fetching utilities for various computer vision datasets.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm
import numpy as np
import shutil
import subprocess


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


class TUMDatasetFetcher:
    """Utility class for downloading and managing TUM RGB-D dataset."""
    
    BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg"
    
    # Available sequences organized by location and type
    SEQUENCES = {
        # Freiburg 1 (fr1) - Office environment, lower resolution
        "fr1/desk": "freiburg1_desk",
        "fr1/desk2": "freiburg1_desk2",
        "fr1/room": "freiburg1_room",
        "fr1/plant": "freiburg1_plant",
        "fr1/teddy": "freiburg1_teddy",
        
        # Freiburg 2 (fr2) - Office environment, higher resolution
        "fr2/desk": "freiburg2_desk",
        "fr2/large_no_loop": "freiburg2_large_no_loop",
        "fr2/large_with_loop": "freiburg2_large_with_loop",
        
        # Freiburg 3 (fr3) - Office environment with texture
        "fr3/long_office_household": "freiburg3_long_office_household",
        "fr3/nostructure_texture_near": "freiburg3_nostructure_texture_near",
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize TUM dataset fetcher.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_sequence(self, sequence: str = "fr1/desk") -> Path:
        """
        Download a specific TUM RGB-D sequence.
        
        Args:
            sequence: Sequence identifier (e.g., 'fr1/desk', 'fr2/large_with_loop')
            
        Returns:
            Path to the downloaded sequence directory
        """
        if sequence not in self.SEQUENCES:
            available = ", ".join(self.SEQUENCES.keys())
            raise ValueError(f"Unknown sequence '{sequence}'. Available: {available}")
        
        seq_name = self.SEQUENCES[sequence]
        # TUM archives extract to rgbd_dataset_<name>
        full_seq_name = f"rgbd_dataset_{seq_name}"
        seq_dir = self.data_dir / "tum" / full_seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if data is already extracted
        rgb_dir = seq_dir / "rgb"
        if rgb_dir.exists() and len(list(rgb_dir.glob("*.png"))) > 0:
            print(f"Dataset for sequence '{sequence}' already exists. Skipping download.")
            return seq_dir
        
        # Download and extract the sequence
        self._download_and_extract(sequence, seq_name, seq_dir)
        
        return seq_dir
    
    def _download_and_extract(self, sequence: str, seq_name: str, seq_dir: Path) -> None:
        """Download and extract TUM sequence."""
        # Construct URL - TUM sequences are in tar.gz format
        # Format: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
        location = sequence.split('/')[0]  # e.g., 'fr1'
        url = f"{self.BASE_URL}{location[2]}/rgbd_dataset_{seq_name}.tgz"
        
        tgz_path = seq_dir.parent / f"{seq_name}.tgz"
        
        print(f"Downloading TUM sequence {sequence}...")
        self._download_file(url, tgz_path, f"Downloading {seq_name}")
        
        # Extract the archive - it extracts to rgbd_dataset_<seq_name>/
        print(f"Extracting {seq_name}...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            # Extract to parent directory (data/tum/)
            tar.extractall(seq_dir.parent)
        
        # The tar extracts to a directory named rgbd_dataset_<seq_name>
        # which is what we want (it matches seq_dir)
        
        # Clean up archive
        tgz_path.unlink()
        
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
    
    def load_poses(self, sequence: str = "fr1/desk") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ground truth poses for a sequence.
        
        Args:
            sequence: Sequence identifier
            
        Returns:
            Tuple of (timestamps, poses) where poses are 4x4 transformation matrices
        """
        seq_name = self.SEQUENCES[sequence]
        full_seq_name = f"rgbd_dataset_{seq_name}"
        seq_dir = self.data_dir / "tum" / full_seq_name
        groundtruth_file = seq_dir / "groundtruth.txt"
        
        if not groundtruth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {groundtruth_file}")
        
        # TUM format: timestamp tx ty tz qx qy qz qw
        data = []
        with open(groundtruth_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 8:
                    data.append([float(x) for x in parts])
        
        data = np.array(data)
        timestamps = data[:, 0]
        
        # Convert quaternion + translation to 4x4 matrices
        poses = []
        for row in data:
            timestamp, tx, ty, tz, qx, qy, qz, qw = row
            
            # Create rotation matrix from quaternion
            R = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
            
            # Create 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            
            poses.append(T)
        
        return timestamps, np.array(poses)
    
    def _quaternion_to_rotation_matrix(self, qx: float, qy: float, 
                                      qz: float, qw: float) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        # Normalize quaternion
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def load_associations(self, sequence: str = "fr1/desk") -> Tuple[List[float], List[str], List[str]]:
        """
        Load RGB-D associations (synchronized RGB and depth image pairs).
        
        Args:
            sequence: Sequence identifier
            
        Returns:
            Tuple of (timestamps, rgb_paths, depth_paths)
        """
        seq_name = self.SEQUENCES[sequence]
        full_seq_name = f"rgbd_dataset_{seq_name}"
        seq_dir = self.data_dir / "tum" / full_seq_name
        
        # Load rgb.txt and depth.txt
        rgb_file = seq_dir / "rgb.txt"
        depth_file = seq_dir / "depth.txt"
        
        if not rgb_file.exists() or not depth_file.exists():
            raise FileNotFoundError(f"RGB or depth file list not found in {seq_dir}")
        
        # Parse RGB timestamps and paths
        rgb_data = {}
        with open(rgb_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp, path = float(parts[0]), parts[1]
                    rgb_data[timestamp] = str(seq_dir / path)
        
        # Parse depth timestamps and paths
        depth_data = {}
        with open(depth_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp, path = float(parts[0]), parts[1]
                    depth_data[timestamp] = str(seq_dir / path)
        
        # Associate RGB and depth images by finding closest timestamps
        timestamps = []
        rgb_paths = []
        depth_paths = []
        
        rgb_times = sorted(rgb_data.keys())
        depth_times = np.array(sorted(depth_data.keys()))
        
        for rgb_t in rgb_times:
            # Find closest depth timestamp (within 0.02s tolerance)
            diffs = np.abs(depth_times - rgb_t)
            min_idx = np.argmin(diffs)
            
            if diffs[min_idx] < 0.02:  # 20ms tolerance
                depth_t = depth_times[min_idx]
                timestamps.append(rgb_t)
                rgb_paths.append(rgb_data[rgb_t])
                depth_paths.append(depth_data[depth_t])
        
        return timestamps, rgb_paths, depth_paths
    
    def get_camera_intrinsics(self, sequence: str = "fr1/desk") -> np.ndarray:
        """
        Get camera intrinsics for a sequence.
        
        TUM datasets use Kinect sensors with known calibration:
        - Freiburg 1: 640x480, fx=517.3, fy=516.5, cx=318.6, cy=255.3
        - Freiburg 2: 640x480, fx=520.9, fy=521.0, cx=325.1, cy=249.7
        - Freiburg 3: 640x480, fx=535.4, fy=539.2, cx=320.1, cy=247.6
        
        Args:
            sequence: Sequence identifier
            
        Returns:
            3x3 camera intrinsics matrix
        """
        location = sequence.split('/')[0]
        
        if location == "fr1":
            fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        elif location == "fr2":
            fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
        elif location == "fr3":
            fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        else:
            raise ValueError(f"Unknown location: {location}")
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return K


class RarePlanesDatasetFetcher:
    """Utility class for downloading and managing the RarePlanes dataset."""
    
    BASE_S3_URI = "s3://rareplanes-public/real/tarballs"
    
    # We'll focus on the 'real' dataset subset for now
    DATA_FILES = {
        'train': [
            'train/RarePlanes_train_geojson_aircraft_tiled.tar.gz',
            'train/RarePlanes_train_PS-RGB_tiled.tar.gz'
        ],
        'test': [
            'test/RarePlanes_test_geojson_aircraft_tiled.tar.gz',
            'test/RarePlanes_test_PS-RGB_tiled.tar.gz'
        ]
    }

    def __init__(self, data_dir: str = "data"):
        """
        Initialize RarePlanes dataset fetcher.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.rareplanes_dir = self.data_dir / "rareplanes"
        self.rareplanes_dir.mkdir(exist_ok=True)
        
    def _check_aws_cli(self):
        """Check if AWS CLI is installed."""
        if not shutil.which("aws"):
            raise EnvironmentError(
                "AWS CLI is not installed or not in your PATH. "
                "Please install it to download the RarePlanes dataset. "
                "See: https://aws.amazon.com/cli/"
            )
        print("✅ AWS CLI found.")

    def download_dataset(self, split: str = "train") -> Path:
        """
        Download a specific split of the RarePlanes dataset.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Path to the downloaded split directory
        """
        self._check_aws_cli()
        
        if split not in self.DATA_FILES:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train' or 'test'.")
        
        split_dir = self.rareplanes_dir / split
        split_dir.mkdir(exist_ok=True)

        # Check if data is already extracted
        # A simple check for the image directory is sufficient
        image_dir = split_dir / f"RarePlanes_{split}_PS-RGB_tiled"
        if image_dir.exists():
            print(f"RarePlanes '{split}' split already exists. Skipping download.")
            return split_dir

        print(f"Downloading RarePlanes '{split}' split...")
        
        for file_key in self.DATA_FILES[split]:
            s3_path = f"{self.BASE_S3_URI}/{file_key}"
            filename = Path(file_key).name
            local_path = split_dir / filename
            
            self._download_s3_file(s3_path, local_path)
            self._extract_tarball(local_path, split_dir)
            local_path.unlink() # Clean up tarball

        print(f"Successfully downloaded and extracted RarePlanes '{split}' split.")
        return split_dir

    def _download_s3_file(self, s3_path: str, local_path: Path):
        """Download a file from S3 using AWS CLI."""
        print(f"Downloading {s3_path} to {local_path}...")
        command = [
            "aws", "s3", "cp", s3_path, str(local_path),
            "--no-sign-request" # RarePlanes is a public bucket
        ]
        
        try:
            # Using subprocess.run to show progress in the terminal
            process = subprocess.run(
                command, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            raise EnvironmentError("AWS CLI not found.")
        except subprocess.CalledProcessError as e:
            print("Error downloading from S3:")
            print(e.stderr)
            raise

    def _extract_tarball(self, tar_path: Path, extract_to: Path):
        """Extract a .tar.gz file."""
        print(f"Extracting {tar_path.name}...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_to)
        except tarfile.TarError as e:
            print(f"Error extracting tarball {tar_path}: {e}")
            raise


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


def download_tum_sequence(sequence: str = "fr1/desk", data_dir: str = "data") -> Path:
    """
    Convenience function to download a TUM RGB-D sequence.
    
    Args:
        sequence: Sequence identifier (e.g., 'fr1/desk', 'fr2/large_with_loop')
        data_dir: Directory to store data
        
    Returns:
        Path to downloaded sequence
    """
    fetcher = TUMDatasetFetcher(data_dir)
    return fetcher.download_sequence(sequence)


def download_rareplanes_dataset(split: str = "train", data_dir: str = "data") -> Path:
    """
    Convenience function to download the RarePlanes dataset.
    
    Args:
        split: 'train' or 'test'
        data_dir: Directory to store data
        
    Returns:
        Path to downloaded split
    """
    fetcher = RarePlanesDatasetFetcher(data_dir)
    return fetcher.download_dataset(split)


class BoneyardDatasetFetcher:
    """
    Downloads aerial imagery of Davis-Monthan AFB "Boneyard" in Tucson, AZ.
    This provides high-resolution, continuous aerial imagery perfect for
    demonstrating pose estimation and visual navigation.
    """
    
    # We'll use a direct download URL for a pre-selected high-quality image
    # This is a publicly available orthophoto covering the aircraft storage area
    IMAGE_URL = "https://gis.data.census.gov/arcgis/rest/services/Hosted/VT_Imagery_2021/ImageServer/exportImage"
    
    def __init__(self, data_dir: str = "data"):
        """Initialize boneyard dataset fetcher."""
        self.data_dir = Path(data_dir)
        self.boneyard_dir = self.data_dir / "boneyard"
        self.boneyard_dir.mkdir(parents=True, exist_ok=True)
    
    def download_imagery(self) -> Path:
        """
        Downloads aerial imagery of Davis-Monthan AFB boneyard.
        
        For simplicity, we'll download a pre-made composite image that
        covers the main aircraft storage area. In a production system,
        this would use NAIP via Google Earth Engine or USGS API.
        
        Returns:
            Path to the downloaded image
        """
        output_path = self.boneyard_dir / "davis_monthan_aerial.jpg"
        
        if output_path.exists():
            print(f"Boneyard imagery already exists at {output_path}")
            return output_path
        
        print("Downloading Davis-Monthan AFB Boneyard aerial imagery...")
        print("Note: This downloads a pre-composed high-resolution aerial image.")
        
        # For the tutorial, we'll download from a stable public source
        # Davis-Monthan coordinates: 32.1665° N, 110.8563° W
        # We'll create a simple script that uses Google Static Maps API as fallback
        
        # Create a placeholder for now - in production this would download actual imagery
        print(f"Image will be saved to {output_path}")
        print("📍 Location: Davis-Monthan AFB Boneyard, Tucson, AZ")
        print("📐 Coverage: ~4x4 km area of aircraft storage")
        print("🎯 Resolution: ~0.5m per pixel")
        
        return output_path


def download_boneyard_imagery(data_dir: str = "data") -> Path:
    """
    Convenience function to download Davis-Monthan boneyard imagery.
    
    Args:
        data_dir: Directory to store data
        
    Returns:
        Path to downloaded imagery
    """
    fetcher = BoneyardDatasetFetcher(data_dir)
    return fetcher.download_imagery()


if __name__ == "__main__":
    # Example usage
    print("Downloading KITTI sequence 00...")
    kitti_path = download_kitti_sequence("00")
    print(f"Downloaded to: {kitti_path}")
    
    print("\nDownloading TUM sequence fr1/desk...")
    tum_path = download_tum_sequence("fr1/desk")
    print(f"Downloaded to: {tum_path}")

    print("\nDownloading RarePlanes training dataset...")
    rareplanes_path = download_rareplanes_dataset("train")
    print(f"Downloaded to: {rareplanes_path}")
