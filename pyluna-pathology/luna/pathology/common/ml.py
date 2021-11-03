import pandas as pd

import torch
from torch import nn

from torch.utils.data import Dataset
from PIL import Image

class BaseTorchTileDataset(Dataset):
    """Base class for a tile dataset
    
    Impliments the usual torch dataset methods, and additionally provides a decoding of the binary tile data.
    PIL images can be further preprocessed before becoming torch tensors via an abstract preprocess method

    Will send the tensors to gpu if available, on the device specified by CUDA_VISIBLE_DEVICES="1"
    """ 
    
    def __init__(self, tile_manifest=None, tile_path=None, label_cols=[]):
        """Initialize BaseTileDataset

        Can accept either a tile dataframe or a path to tile data
        
        Args:
            tile_manifest (pd.DataFrame): Dataframe of tile data
            tile_path (str): Base path of tile data
            label_cols (list[str]): (Optional) label columns to return as tensors, e.g. for training
        """

        if tile_manifest is not None:
            self.tile_manifest = tile_manifest 
        elif tile_path is not None: 
            self.tile_manifest = pd.read_csv(tile_path + 'address.slice.csv').set_index("address")
            self.tile_manifest['data_path'] = tile_path + 'tiles.slice.pil'
        else:
            raise RuntimeError("Must specifiy either tile_manifest or tile_path")
        
        self.label_cols = label_cols
        
    def __len__(self):
        return len(self.tile_manifest)
    
    def __repr__(self):
        return f"TileDataset with {len(self.tile_manifest)} tiles, indexed by {self.tile_manifest.index.names}, returning label columns: {self.label_cols}"
    
    def __getitem__(self, idx: int):
        """Tile accessor
        
        Loads a tile image from the tile manifest.  Returns a batch of the indicies of the input dataframe, the tile data always. 
        If label columns where specified, the 3rd position of the tuple is a tensor of the label data.

        Args:
            idx (int): Integer index 

        Returns:
            (str, torch.tensor, optional torch.tensor): tuple of the tile index and corresponding tile as a torch tensor, and metadata labels if specified
        """ 
            
        if not type(idx)==int: raise TypeError(f"BaseTileDataset only accepts interger indicies, got {type(idx)}")
        row = self.tile_manifest.iloc[idx]
        with open(row.data_path, "rb") as fp:
            fp.seek(int(row.tile_image_offset))
            img = Image.frombytes(
                row.tile_image_mode,
                (int(row.tile_image_size_xy), int(row.tile_image_size_xy)),
                fp.read(int(row.tile_image_length)),
            )    

        if len(self.label_cols):                
            return row.name, self.preprocess(img), torch.tensor(row[self.label_cols].to_list())
        else:
            return row.name, self.preprocess(img)
       
    def preprocess(self, input_tile: Image):
        """Preprocessing method called for each tile patch
        
        Loads a tile image from the tile manifest, must be manually implimented to accept a single PIL image and return a torch tensor.

        Args:
            input_tile (Image): Integer index 

        Returns:
            torch.tensor: Output tile as preprocessed tensor
        """ 
        raise NotImplementedError("preprocess() has not be implimented in the subclass!")
        
class BaseTorchTileClassifier(nn.Module):
    def __init__(self, **kwargs):
        """Initialize BaseTorchTileClassifier 

        Will run on cuda if available, on the device specified by CUDA_VISIBLE_DEVICES="1"
        
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

        with torch.no_grad():
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
        
        Loads a tile image from the tile manifest, must be manually implimented to pass the input tensor through the modules specified in setup()

        Args:
            input_tiles (torch.tensor): Input tiles of shape (B, *)

        Returns:
            torch.tensor: 2D tensor with (B, C) where B is the batch dimension and C are output classes or features
        """ 
        raise NotImplementedError("predict() has not be implimented in the subclass!")
