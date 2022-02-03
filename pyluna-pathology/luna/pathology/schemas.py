import pandas as pd


class SchemaMismatchError(Exception): pass

class SlideTiles:
    REQ_COLUMNS = set(['address', 'tile_size', 'tile_units', 'x_coord', 'y_coord'])
    @classmethod
    def check(self, slide_tiles):
        df = pd.read_csv(slide_tiles)

        if not set(df.columns).intersection(self.REQ_COLUMNS) == self.REQ_COLUMNS:
            raise SchemaMismatchError("SlideTile failed schema check: missing columns: ", (set(df.columns).intersection(self.REQ_COLUMNS)).symmetric_difference(self.REQ_COLUMNS))
        
        return True