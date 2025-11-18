"""
Checkpoint Manager for resumable processing
Allows saving and loading progress to resume interrupted operations
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class CheckpointManager:
    """
    Manages checkpoints for long-running operations
    Supports automatic saving and resuming from last checkpoint
    """

    def __init__(self, checkpoint_dir, checkpoint_name="default"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_name: Name identifier for this checkpoint series
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}_checkpoint.pkl"
        self.metadata_file = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"

    def save(self, data, metadata=None, verbose=True):
        """
        Save checkpoint with data and optional metadata

        Args:
            data: Data to checkpoint (dict, DataFrame, list, etc.)
            metadata: Optional metadata dict (e.g., iteration number, timestamp)
            verbose: Print save confirmation
        """
        try:
            # Save main data
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(data, f)

            # Save metadata
            if metadata is None:
                metadata = {}

            metadata['timestamp'] = datetime.now().isoformat()
            metadata['checkpoint_name'] = self.checkpoint_name

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            if verbose:
                print(f"✓ Checkpoint saved: {self.checkpoint_file.name}")
                if 'iteration' in metadata:
                    print(f"  Iteration: {metadata['iteration']}")

        except Exception as e:
            print(f"✗ Error saving checkpoint: {e}")

    def load(self, verbose=True):
        """
        Load checkpoint data

        Args:
            verbose: Print load confirmation

        Returns:
            Tuple of (data, metadata) or (None, None) if no checkpoint exists
        """
        if not self.exists():
            if verbose:
                print(f"No checkpoint found: {self.checkpoint_file.name}")
            return None, None

        try:
            # Load data
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            # Load metadata
            metadata = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)

            if verbose:
                print(f"✓ Checkpoint loaded: {self.checkpoint_file.name}")
                if 'timestamp' in metadata:
                    print(f"  Saved at: {metadata['timestamp']}")
                if 'iteration' in metadata:
                    print(f"  Iteration: {metadata['iteration']}")

            return data, metadata

        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return None, None

    def exists(self):
        """Check if checkpoint exists"""
        return self.checkpoint_file.exists()

    def delete(self, verbose=True):
        """Delete checkpoint files"""
        deleted = False

        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            deleted = True

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            deleted = True

        if verbose and deleted:
            print(f"✓ Checkpoint deleted: {self.checkpoint_name}")

        return deleted

    def get_metadata(self):
        """Get checkpoint metadata without loading data"""
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file, 'r') as f:
            return json.load(f)


class DataFrameCheckpointManager:
    """
    Specialized checkpoint manager for pandas DataFrames
    Uses more efficient CSV/parquet storage
    """

    def __init__(self, checkpoint_dir, checkpoint_name="df_checkpoint", format='parquet'):
        """
        Initialize DataFrame checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_name: Name identifier for this checkpoint
            format: 'csv' or 'parquet' (parquet is more efficient)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name
        self.format = format

        if format == 'parquet':
            self.checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.parquet"
        else:
            self.checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.csv"

        self.metadata_file = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"

    def save(self, df, metadata=None, verbose=True):
        """
        Save DataFrame checkpoint

        Args:
            df: pandas DataFrame to save
            metadata: Optional metadata dict
            verbose: Print confirmation
        """
        try:
            # Save DataFrame
            if self.format == 'parquet':
                df.to_parquet(self.checkpoint_file, index=False)
            else:
                df.to_csv(self.checkpoint_file, index=False)

            # Save metadata
            if metadata is None:
                metadata = {}

            metadata['timestamp'] = datetime.now().isoformat()
            metadata['checkpoint_name'] = self.checkpoint_name
            metadata['rows'] = len(df)
            metadata['columns'] = len(df.columns)

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            if verbose:
                print(f"✓ DataFrame checkpoint saved: {self.checkpoint_file.name}")
                print(f"  Shape: {df.shape}")

        except Exception as e:
            print(f"✗ Error saving DataFrame checkpoint: {e}")

    def load(self, verbose=True):
        """
        Load DataFrame checkpoint

        Returns:
            Tuple of (DataFrame, metadata) or (None, None) if not exists
        """
        if not self.exists():
            if verbose:
                print(f"No checkpoint found: {self.checkpoint_file.name}")
            return None, None

        try:
            # Load DataFrame
            if self.format == 'parquet':
                df = pd.read_parquet(self.checkpoint_file)
            else:
                df = pd.read_csv(self.checkpoint_file)

            # Load metadata
            metadata = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)

            if verbose:
                print(f"✓ DataFrame checkpoint loaded: {self.checkpoint_file.name}")
                print(f"  Shape: {df.shape}")

            return df, metadata

        except Exception as e:
            print(f"✗ Error loading DataFrame checkpoint: {e}")
            return None, None

    def exists(self):
        """Check if checkpoint exists"""
        return self.checkpoint_file.exists()

    def delete(self, verbose=True):
        """Delete checkpoint files"""
        deleted = False

        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            deleted = True

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            deleted = True

        if verbose and deleted:
            print(f"✓ Checkpoint deleted: {self.checkpoint_name}")

        return deleted


def checkpoint_wrapper(checkpoint_manager, operation_func, *args, **kwargs):
    """
    Wrapper function to automatically checkpoint an operation

    Args:
        checkpoint_manager: CheckpointManager instance
        operation_func: Function to execute
        *args, **kwargs: Arguments to pass to operation_func

    Returns:
        Result from operation_func
    """
    # Try to load existing checkpoint
    result, metadata = checkpoint_manager.load()

    if result is not None:
        print("Resuming from checkpoint...")
        return result

    # No checkpoint, run the operation
    print("No checkpoint found, running operation...")
    result = operation_func(*args, **kwargs)

    # Save checkpoint
    checkpoint_manager.save(result)

    return result


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Example 1: Basic checkpoint
    print("Example 1: Basic checkpoint")
    print("-" * 50)
    cm = CheckpointManager("./checkpoints", "example_basic")

    # Save some data
    data = {'processed_ids': [1, 2, 3, 4, 5], 'last_index': 5}
    cm.save(data, metadata={'iteration': 5, 'status': 'in_progress'})

    # Load it back
    loaded_data, metadata = cm.load()
    print(f"Loaded data: {loaded_data}")
    print(f"Metadata: {metadata}")

    # Clean up
    cm.delete()

    print("\n")

    # Example 2: DataFrame checkpoint
    print("Example 2: DataFrame checkpoint")
    print("-" * 50)
    df_cm = DataFrameCheckpointManager("./checkpoints", "example_df", format='csv')

    # Create and save a DataFrame
    df = pd.DataFrame({
        'id': range(1000),
        'value': np.random.randn(1000)
    })
    df_cm.save(df, metadata={'processing_stage': 'cleaned'})

    # Load it back
    loaded_df, metadata = df_cm.load()
    print(f"Loaded DataFrame shape: {loaded_df.shape}")
    print(f"Metadata: {metadata}")

    # Clean up
    df_cm.delete()
