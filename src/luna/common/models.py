import uuid
from typing import Optional

import pandera as pa
from pandera.engines.pandas_engine import PydanticModel
from pydantic import BaseModel


class Slide(BaseModel):
    id: str
    project_name: str = ""
    comment: str = ""
    size: int
    uuid: uuid.UUID
    url: str
    channel0_R: Optional[float]
    channel0_G: Optional[float]
    channel0_B: Optional[float]
    channel1_R: Optional[float]
    channel1_G: Optional[float]
    channel1_B: Optional[float]

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


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
    size: int
    units: str


class TileSchema(pa.DataFrameModel):
    class Config:
        dtype = PydanticModel(Tile)
        coerce = True
