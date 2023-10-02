from typing import Optional

import pandera as pa
from pandera.typing import Series
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel


class Slide(BaseModel):
    id: str
    project_name: str = ""
    comment: str = ""
    slide_size: int
    uuid: str
    url: str
    channel0_R: Optional[float]
    channel0_G: Optional[float]
    channel0_B: Optional[float]
    channel1_R: Optional[float]
    channel1_G: Optional[float]
    channel1_B: Optional[float]

    class Config:
        extra = "allow"


class ShapeFeaturesSchema(pa.DataFrameModel):
    slide_id: Series[str]
    Parent: Series[str]
    Class: Series[str]
    variable: Series[str]
    value: Series = pa.Field(nullable=True)


class SlideSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(Slide)
        coerce = True
        strict = False


class Tile(BaseModel):
    address: str
    x_coord: int
    y_coord: int
    xy_extent: int
    tile_size: int
    tile_units: str

    class Config:
        extra = "allow"


class StoredTile(Tile):
    tile_store: str


class LabeledTile(StoredTile):
    Classification: str


class TileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(Tile)
        coerce = True


class StoredTileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(StoredTile)
        coerce = True


class LabeledTileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(LabeledTile)
        coerce = True
