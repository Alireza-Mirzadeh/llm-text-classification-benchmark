"""
Data loader module for loading Rotten Tomatoes dataset using the Higgibg Face datasets library.

This module handles:

    - Downloading dataset from Hugging Face
    - Saving raw splits to local
    - Loading dataset from local storage
"""
import logging
from pathlib import Path
from datasets import load_dataset, load_from_disk
import yaml

# Load configuration from a YAML file
class ConfigLoader:

    @staticmethod

    def load_config(config_path="config/config.yaml"):
        """ 
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration file.
        """

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

# Data loader class for Rotten Tomatoes dataset
class RottenTomatoesDataLoader:
    """
    This class handles downloading, saving, and loading the dataset.
    """

    def __init__(self, config_path="config/config.yaml"):
        """ 
        Initialize the data loader with configuration.
        """
        self.config = ConfigLoader.load_config(config_path)
        self.data_path = Path(self.config["data"]["raw_path"])
        self.dataset_name = self.config["data"]["dataset_name"]

        self.data_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        
    def download_dataset(self):
        """ 
        Download dataset from Hugging Face.
        """
        self.logger.info(f"Downloading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name)
        return dataset

    def save_dataset_local(self):
        """ 
        Save dataset to local storage.
        """
        self.logger.info(f"Saving dataset to local path: {self.data_path}")
        dataset.save_to_disk(str(self.data_path))
        self.logger.info(f"Dataset saved to {self.data_path}")

    def load_dataset_local(self):
        """ 
        Load dataset from local storage
        """ 
        self.logger.info(f"Loading dataset from local_path: {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(
                "Local dataset not found. Run download_dataset() first."
            )

        return load_from_disk(str(self.data_path))

if __name__ == "__main__":

    loader = RottenTomatoesDataLoader()

    dataset = loader.download_dataset()
    loader.save_dataset_local()

    # Sanity check
    loaded = loader.load_dataset_local()
    print(f"Loaded dataset from local storage: {loaded}")


    