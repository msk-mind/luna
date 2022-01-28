import os
from pathlib import Path
from urllib.parse import urlparse
from functools import partial

import shutil
from minio import Minio

class WriteAdapter:
    def __init__(self): pass

class FileWriteAdatper(WriteAdapter):
    def __init__(self, store_url, bucket, dry_run=False):
        """ Return a WriteAdapter for a given file I/O scheme and URL
        
        Args:
            store_url (str): root URL for the storage location (e.g. s3://localhost:9000 or file:///data)
            bucket (str): the "bucket" or "parent folder" for the storage location
            dry_run (bool): don't actually copy any data
        """
        self.dry_run = dry_run
        
        url_result = urlparse(store_url)
        
        # All we need is the base path
        self.store_path = url_result.path
        self.bucket = bucket

        # Define base URL
        self.base_url = f'file://{Path(self.store_path)}'
        print("Configured FileAdatper with base URL=" + self.base_url)
        
        print ("Available buckets:")
        for x in os.listdir(self.store_path): print (f' - {x}')

    def write(self, input_data, prefix):
        """ Perform write operation to a posix file system
        
        Args:
            input_data (str): path to input file
            prefix (str): relative path prefix for destination
        """
        input_path  = Path(input_data)
        prefix_path = Path(prefix)
        filename    = input_path.name
        
        if prefix_path.is_absolute(): raise RuntimeError("Prefix paths must be relative!")

        output_data_url = os.path.join(self.base_url, self.bucket, prefix, filename)
        
        output_dir = os.path.join(self.store_path, self.bucket, prefix)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.dry_run: 
            shutil.copy(input_data, output_dir)
            size = os.stat(os.path.join(output_dir, filename)).st_size
        else: 
            print (input_data + "->" + output_data_url)
            size = 0

        return {'data_url': output_data_url, 'size': size}

class MinioWriteAdatper(WriteAdapter):
    def __init__(self, store_url, bucket, dry_run=False):
        """ Return a WriteAdapter for a given file I/O scheme and URL
        
        Args:
            store_url (str): root URL for the storage location (e.g. s3://localhost:9000 or file:///data)
            bucket (str): the "bucket" or "parent folder" for the storage location
            dry_run (bool): don't actually copy any data
        """       
        self.dry_run = dry_run

        url_result = urlparse(store_url)

        # We need a bit more detail here
        self.store_hostname = url_result.hostname
        self.store_port = url_result.port
        self.bucket = bucket

        self.client_init = partial(Minio, f'{self.store_hostname}:{self.store_port}', access_key=os.environ['MINIO_USER'], secret_key=os.environ['MINIO_PASSWORD'], secure=True)

        self.base_url = os.path.join(f"s3://{self.store_hostname}:{self.store_port}")
        print("Configured MinioAdatper with base URL=" + self.base_url)
        
        client = self.client_init()
        
        print ("Available buckets:")
        for x in client.list_buckets(): print (f' - {x.name}')

    def write(self, input_data, prefix):
        """ Perform write operation to a s3 file system
        
        Args:
            input_data (str): path to input file
            prefix (str): relative path prefix for destination
        """
        input_path  = Path(input_data)
        prefix_path = Path(prefix)
        filename    = input_path.name
        
        if prefix_path.is_absolute(): raise RuntimeError("Prefix paths must be relative!")

        output_data_url = os.path.join(self.base_url, self.bucket, prefix, filename)

        client = self.client_init()
        
        if not self.dry_run: 
            client.fput_object(self.bucket, f"{prefix_path}/{filename}", input_path, part_size=250000000)
            size = client.stat_object(self.bucket, f"{prefix_path}/{filename}").size
        else: 
            print (input_data + "->" + output_data_url)
            size = 0


        return {'data_url': output_data_url, 'size': size}
    

class IOAdapter:
    """ Interface for IO 

    Exposes a write and read method via scheme specific classes:

    IOAdapter.writer(<scheme>).write(<data>) -> url

    IOAdapter.reader(<scheme>).read(<url>)   -> data
    """
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
    
    def writer(self, store_url, bucket):
        """ Return a WriteAdapter for a given file I/O scheme and URL
        
        Args:
            store_url (str): root URL for the storage location (e.g. s3://localhost:9000 or file:///data)
            bucket (str): the "bucket" or "parent folder" for the storage location
            
        Returns
            WriteAdapter: object capable of writing to the location at store_url
            
        """
        url_result = urlparse(store_url)
        
        if url_result.scheme == 'file':
            return FileWriteAdatper(store_url=store_url, bucket=bucket, dry_run=self.dry_run)
        elif url_result.scheme == 's3':
            return MinioWriteAdatper(store_url=store_url, bucket=bucket, dry_run=self.dry_run)
        else:
            raise RuntimeError("Unsupported slide store schemes, please try s3:// or file://")
