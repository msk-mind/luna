import os
import pandas as pd

import pyarrow.parquet as pq
import pyarrow as pa

class SaveParquet:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    def __call__(self, input_data, segment_id, extra_metadata=None):

        os.makedirs(self.dataset_dir, exist_ok=True)

        df = pd.read_csv(input_data)

        for col in df.columns:
            df[col] = df[col].astype(np, errors='ignore')
        
        df['segment_id'] = segment_id
        df = df.set_index('segment_id')

        for key, val in extra_metadata.items():
            df[key] = val

        output_filename = os.path.join(self.dataset_dir, f"{segment_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_filename)

        return output_filename

