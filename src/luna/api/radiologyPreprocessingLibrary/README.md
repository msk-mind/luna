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

# Process and Save results in the object store
param = {
  "paths": ['1.dcm', '2.dcm', '3.dcm', '4.dcm', '5.dcm'],
  "width": 512,
  "height": 512
}

res = requests.post("http://127.0.0.1:5000/radiology/images/breast-mri/some-scan-id", json=param)
```
```
# Download results from the object store
param = {
  "output_location": "/path/to/test-download.parquet"
}
res = requests.get("http://127.0.0.1:5000/radiology/images/breast-mri/some-scan-id", json=param)
```
```
# Delete images from the object store
res = requests.delete("http://127.0.0.1:5000/radiology/images/breast-mri/some-scan-id")
```

- Load results in Pandas DF

```
import pyarrow.parquet as pq

table = pq.read_table('test-download.parquet')
df = table.to_pandas()
```

### Notes

- Embedding binaries in parquet format, didn't work well with Minio select API, 
as the payload is binary, and the 'content' field of the parquet is also binary.