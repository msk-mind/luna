import os
from pathlib import Path
from urllib.parse import urlparse
from functools import partial

import shutil
from minio import Minio
# from minio.error import NoSuchKey
import calendar
from datetime import datetime


def get_object_stats(client, bucket, key):
    try:
        stat_object = client.stat_object(bucket, key)
    except:
        return -1, -1
    return stat_object.size, calendar.timegm(stat_object.last_modified.timetuple())

def get_file_stats(path):
    try:
        stat_file   = os.stat(path)
    except FileNotFoundError:
        return -1, -1
    return stat_file.st_size, stat_file.st_mtime

class WriteAdapter:
    def __init__(self): pass

class NoWriteAdapter(WriteAdapter):
    def __init__(self):
        self.base_url = "file://"
        print("Configured NoWriteAdatper with base URL=" + self.base_url)

    def write(self, input_data, prefix) -> dict:
        """ Returns input_data as written data 
        
        Args:
            input_data (str): path to input file
            prefix (str): relative path prefix for destination, ignored
        Returns:
            dict: key-value pairs containing metadata about the write operation
        """
        input_path  = Path(input_data)
        filename    = input_path.name

        if not os.path.exists(input_path):
            return {}
        
        output_data_url = f"{self.base_url}{input_path}"
        
        input_size,  input_mtime   = get_file_stats  (input_data)
    
        return {'readable': True, 'data_url': output_data_url, 'size': input_size, 'ingest_time': datetime.fromtimestamp(input_mtime)}


class FileWriteAdatper(WriteAdapter):
    def __init__(self, store_url, bucket):
        """ Return a WriteAdapter for a given file I/O scheme and URL
        
        Args:
            store_url (str): root URL for the storage location (e.g. s3://localhost:9000 or file:///data)
            bucket (str): the "bucket" or "parent folder" for the storage location
        """
        url_result = urlparse(store_url)
        
        # All we need is the base path
        self.store_path = url_result.path
        self.bucket = bucket

        # Define base URL
        self.base_url = f'file://{Path(self.store_path)}'
        print("Configured FileAdatper with base URL=" + self.base_url)
        
        print ("Available buckets:")
        for x in os.listdir(self.store_path): print (f' - {x}')

    def write(self, input_data, prefix) -> dict:
        """ Perform write operation to a posix file system
        
        Will not perform write if :
            the content length matches (full copy) and the input modification time is earlier than the ingest time (with a 1 min. grace period)
        
        Args:
            input_data (str): path to input file
            prefix (str): relative path prefix for destination
        Returns:
            dict: key-value pairs containing metadata about the write operation
        """
        input_path  = Path(input_data)
        prefix_path = Path(prefix)
        filename    = input_path.name

        if not os.path.exists(input_path):
            return {}
        
        if prefix_path.is_absolute(): raise RuntimeError("Prefix paths must be relative!") # User needs to know prefixes are relative paths

        output_data_url = os.path.join(self.base_url, self.bucket, prefix, filename)
        
        output_dir = os.path.join(self.store_path, self.bucket, prefix)
        output_data = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        input_size,  input_mtime   = get_file_stats  (input_data)
        output_size, output_mtime  = get_file_stats  (output_data)

        needs_write = ( output_size != input_size or (input_mtime - 60) > output_mtime ) # 60 second grace period

        if needs_write: 
            shutil.copy(input_data, output_data)
        
        output_size, output_mtime  = get_file_stats  (output_data)

        if not output_size> 0: return {'readable': False}

        return {'readable': True, 'data_url': output_data_url, 'size': output_size, 'ingest_time': datetime.fromtimestamp(output_mtime)}
        
class MinioWriteAdatper(WriteAdapter):
    def __init__(self, store_url, bucket, secure=False):
        """ Return a WriteAdapter for a given file I/O scheme and URL
        
        Args:
            store_url (str): root URL for the storage location (e.g. s3://localhost:9000 or file:///data)
            bucket (str): the "bucket" or "parent folder" for the storage location
        """       

        url_result = urlparse(store_url)

        # We need a bit more detail here
        self.store_hostname = url_result.hostname
        self.store_port = url_result.port
        self.bucket = bucket
        self.secure = secure

        self.client_init = partial(Minio, f'{self.store_hostname}:{self.store_port}', access_key=os.environ['MINIO_USER'], secret_key=os.environ['MINIO_PASSWORD'], secure=secure)

        self.base_url = os.path.join(f"s3://{self.store_hostname}:{self.store_port}")
        print("Configured MinioAdatper with base URL=" + self.base_url)
        
        client = self.client_init()
        
        print ("Available buckets:")
        for x in client.list_buckets(): print (f' - {x.name}')

    def write(self, input_data, prefix) -> dict:
        """ Perform write operation to a s3 file system
        
        Will not perform write if :
            the content length matches (full copy) and the input modification time is earlier than the ingest time (with a 1 min. grace period)

        Args:
            input_data (str): path to input file
            prefix (str): relative path prefix for destination
        Returns:
            dict: key-value pairs containing metadata about the write operation
        """
        input_path  = Path(input_data)
        prefix_path = Path(prefix)
        filename    = input_path.name

        if not os.path.exists(input_path):
            return {}
        
        if prefix_path.is_absolute(): raise RuntimeError("Prefix paths must be relative!") # User needs to know prefixes are relative paths

        output_data_url = os.path.join(self.base_url, self.bucket, prefix, filename)

        client = self.client_init()

        object_size, object_mtime = get_object_stats(client, self.bucket, f"{prefix_path}/{filename}")
        input_size,  input_mtime  = get_file_stats  (input_data)
        needs_write = ( object_size != input_size or (input_mtime - 60) > object_mtime ) # 60 second grace period

        if needs_write: 
            client.fput_object(self.bucket, f"{prefix_path}/{filename}", input_path, part_size=250000000)
        
        object_size, object_mtime = get_object_stats(client, self.bucket, f"{prefix_path}/{filename}")

        if not object_size > 0: return {'readable': False}

        return {'readable': True, 'data_url': output_data_url, 'size': object_size, 'ingest_time': datetime.fromtimestamp(object_mtime)}
    

class IOAdapter:
    """ Interface for IO 

    Exposes a write and read method via scheme specific classes:

    IOAdapter.writer(<scheme>).write(<data>) -> url

    IOAdapter.reader(<scheme>).read(<url>)   -> data (TODO)
    """
    def __init__(self, no_write=False):
        self.no_write = no_write
    
    def writer(self, store_url, bucket):
        """ Return a WriteAdapter for a given file I/O scheme and URL
        
        Args:
            store_url (str): root URL for the storage location (e.g. s3://localhost:9000 or file:///data)
            bucket (str): the "bucket" or "parent folder" for the storage location
            
        Returns
            WriteAdapter: object capable of writing to the location at store_url
            
        """
        if self.no_write: return NoWriteAdapter()

        url_result = urlparse(store_url)
        
        if url_result.scheme == 'file':
            return FileWriteAdatper(store_url=store_url, bucket=bucket)
        elif url_result.scheme == 's3':
            return MinioWriteAdatper(store_url=store_url, bucket=bucket)
        elif url_result.scheme == 's3+https':
            return MinioWriteAdatper(store_url=store_url, bucket=bucket, secure=True)
        else:
            raise RuntimeError("Unsupported slide store schemes, please try s3:// or file://")
