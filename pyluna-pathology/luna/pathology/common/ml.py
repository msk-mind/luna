import pandas as pd

import torch
from torch import nn

from torch.utils.data import Dataset
from PIL import Image

class BaseTorchTileDataset(Dataset):
    """Base class for a tile dataset
    
    Impliments the usual torch dataset methods, and additionally provides a decoding of the binary tile data.
    PIL images can be further preprocessed before becoming torch tensors via an abstract preprocess method
    """ 
    
    def __init__(self, tile_path):
        """Initialize BaseTileDataset
        
        Args:
            tile_path (str): Base path of tile data
        """
        self.tile_manifest = pd.read_csv(tile_path + 'address.slice.csv')
        self.pil_data = tile_path + 'tiles.slice.pil'
        
    def __len__(self):
        return len(self.tile_manifest)
    
    def __repr__(self):
        return f"TileDataset with {len(self.tile_manifest)} tiles"
    
    def __getitem__(self, idx: int):
        """Tile accessor
        
        Loads a tile image from the tile manifest

        Args:
            idx (int): Integer index 

        Returns:
            address, image, (str, torch.tensor): tuple of the tile index and corresponding tile as a torch tensor
        """ 
            
        if not type(idx)==int: raise TypeError(f"BaseTileDataset only accepts interger indicies, got {type(idx)}")
        row = self.tile_manifest.iloc[idx]
        with open(self.pil_data, "rb") as fp:
            fp.seek(int(row.tile_image_offset))
            img = Image.frombytes(
                row.tile_image_mode,
                (int(row.tile_image_size_xy), int(row.tile_image_size_xy)),
                fp.read(int(row.tile_image_length)),
            )                    
        return row.address, self.preprocess(img)
       
    def preprocess(self, input_tile: Image):
        """Preprocessing method called for each tile patch
        
        Loads a tile image from the tile manifest

        Args:
            input_tile (Image): Integer index 

        Returns:
            output_tile (torch.tensor): Output tile as preprocess tensor
        """ 
        raise NotImplementedError("preprocess() has not be implimented in the subclass!")
        
class BaseTorchTileClassifier(nn.Module):
    def __init__(self, **kwargs):
        """Initialize BaseTorchTileClassifier 
        
        Args:
            kwargs: Keyward arguements passed onto the subclass method
        """
        super(BaseTorchTileClassifier, self).__init__()
        
        self.cuda_is_available = torch.cuda.is_available()

        self.setup(**kwargs)
        self.eval()
        
        if self.cuda_is_available: self.cuda()
    def forward(self, index, tile_data):
        """Forward pass for base classifier class
        
        Loads a tile image from the tile manifest

        Args:
            index (list[str]): Tile address indicies with length B
            tile_data (torch.tensor): Input tiles of shape (B, *)

        Returns:
            pd.DataFrame: Dataframe of output features
        """ 
        if self.cuda_is_available: tile_data = tile_data.cuda()

        return pd.DataFrame(self.predict(tile_data).cpu().numpy(), index=index)
    
    def setup(self, **kwargs):
        """Set classifier modules
        
        Template/abstract method where individual modules that make up the forward pass are configured

        Args:
            kwargs: Keyward arguements passed onto the subclass method
        """
        raise NotImplementedError("setup() has not be implimented in the subclass!")
    
    def predict(self, input_tiles: torch.tensor):
        """predict method
        
        Loads a tile image from the tile manifest

        Args:
            input_tiles (torch.tensor): Input tiles of shape (B, *)

        Returns:
            torch.tensor: 2D tensor with (B, C) where B is the batch dimension and C are output classes or features
        """ 
        raise NotImplementedError("predict() has not be implimented in the subclass!")
