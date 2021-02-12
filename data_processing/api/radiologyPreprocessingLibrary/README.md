## Radiology Preprocessing Services

### Setup

- Install dependencies

`pip3 install -r data_processing/api/radiologyPreprocessingLibrary/requirements.txt`

- Update configs `app.cfg.template` and save as `app.cfg`

- Start Flask server


`python3 -m data_processing.api.radiologyPreprocessingLibrary.app.app`

- Checkout the API documentation at `http://127.0.0.1:5000/`

### Usage

- Send requests!

```
import requests

param = {
  "paths": ['/Users/rosed2/Downloads/Vittorio_Louise_E_19600825_32210602/PreContrast/IM-0013-0135.dcm',
'/Users/rosed2/Downloads/Vittorio_Louise_E_19600825_32210602/PreContrast/IM-0013-0136.dcm',
'/Users/rosed2/Downloads/Vittorio_Louise_E_19600825_32210602/PreContrast/IM-0013-0137.dcm',
'/Users/rosed2/Downloads/Vittorio_Louise_E_19600825_32210602/PreContrast/IM-0013-0138.dcm',
'/Users/rosed2/Downloads/Vittorio_Louise_E_19600825_32210602/PreContrast/IM-0013-0139.dcm'],
  "width": 512,
  "height": 512
}

# Process and Save results in the object store
res = requests.post("http://127.0.0.1:5000/radiology/images/breast-mri/some-scan-id", json=param)

# Download results from the object store
res = requests.get("http://127.0.0.1:5000/radiology/images/breast-mri/some-scan-id/test-download.parquet")
```

- Load results in Pandas DF

```
import pyarrow.parquet as pq

table = pq.read_table('test-download.parquet')
df = table.to_pandas()
```
