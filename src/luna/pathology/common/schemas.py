import pandas as pd


class SchemaMismatchError(Exception):
    pass


class SlideTiles:
    REQ_COLUMNS = set(
        ["address", "x_coord", "y_coord", "xy_extent", "tile_size", "tile_units"]
    )

    @classmethod
    def check(self, slide_tiles):
        """Returns True if the given path is readable as "SlideTiles <slide_tiles>", else, reaises SchemaMismatchError"""
        df = pd.read_parquet(slide_tiles).reset_index()

        if not set(df.columns).intersection(self.REQ_COLUMNS) == self.REQ_COLUMNS:
            raise SchemaMismatchError(
                "SlideTile failed schema check: missing columns: ",
                (set(df.columns).intersection(self.REQ_COLUMNS)).symmetric_difference(
                    self.REQ_COLUMNS
                ),
            )

        return True
