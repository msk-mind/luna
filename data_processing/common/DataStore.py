from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.Node import Node
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger   import init_logger

import os, socket, pathlib, logging, shutil
from minio import Minio

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


class DataStore_v2:
    def __init__(self):
        self.backend = '/gpfs/mskmindhdp_emc/data_dev/'
        os.makedirs(self.backend, exist_ok=True)
        print ("File backend=", self.backend)

    def _generate_qualified_path(self, store_id, namespace_id, data_type, data_tag):
        return os.path.join (self.backend, store_id, namespace_id, data_type, data_tag)

    def put(self, filepath, store_id, namespace_id, data_type, data_tag='data'):
        dest_dir = os.path.join (self.backend, store_id, namespace_id, data_type, data_tag)
        os.makedirs(dest_dir, exist_ok=True)
        print ("Save", filepath, "->", dest_dir)
        shutil.copy(filepath, dest_dir )
    
    def write(self, iostream, store_id, namespace_id, data_type, data_tag, dtype='w'):
        dest_path_dir  = os.path.join (store_id, namespace_id, data_type)
        dest_path_file = os.path.join (dest_path_dir, data_tag)

        dest_dir  = os.path.join (self.backend, dest_path_dir)
        dest_file = os.path.join (self.backend, dest_path_file)

        os.makedirs(dest_dir, exist_ok=True)
        print ("Save ->", dest_file)
        with open(dest_file, dtype) as fp:
            fp.write(iostream)

        return dest_path_file





def bootstrap (container_id): 
    logger = logging.getLogger(__name__)
    logger.info(f"Bootstrapping pipeline for {container_id}")
    return 1

class DataStore(object):
    """
    DataStore: an abstraction with an id, name, namespace, type, and a list of associated data nodes

    Interfaces with a metadata store (graph DB) and raw file stores (gpfs, potentially others)

    Handles the matching and creation of metadata

    Example usage:
    $ container = data_processing.common.GraphEnum.DataStore( params ).setNamespace("test").setContainer("1.2.840...")
        > Connecting to: neo4j://localhost:7687
        > Connection successfull: True
        > Running on: localhost
        > Lookup ID: 1.2.840...
        > Found: [<Record id(container)=7091 labels(container)=['scan'] container.type='scan' container.name='1.2.840...>]
        > Match on: WHERE id(container) = 7091 
        > Successfully attached to: scan 1.2.840...

    $ node = Node("dicom", "DCM-0123", {"Modality":"CT", "path":"file:/some/path/1.dcm"})

    $ container.put(node)
        > Adding: test-0000
          DataStore has 1 pending commits

    $ 
        > Committing dicom:globals{ hash: 'abc123' name: 'my-dicom', qualified_address: 'test::1.2.840::my-dicom', namespace: 'test', type: 'dicom' , path: 'file:/some/path/1.dcm'}

    $ container.get("dicom", "my-dicom").path
        > /some/path/1.dcm

    $ container.get("dicom", "my-dicom").properties['Modality']
        > 'CT'
    
    The container includes a logging method:
    $ container.logger.info("I am processing the CT")
        > 'yyyy-mm-dd h:m:s,ms - DataStore [1] - INFO - I am processing the CT'

    
    """
    # TODO: worried about schema issues? like making sure name, namespace, type and qualified path are present, neo4j offers schema enforcment. 
    # TODO: testing
    # TODO: error checking

    def __init__(self, params):
        """
        Initialize the container object.
            Connects to the graph DB
            Figure out what host this code is running on

        :params: params - dictonary of important configuration, right now, only the graph URI connection parameters are needed.
        """

        self.logger = logging.getLogger(__name__)

        if isinstance(params, ConfigSet):
            params=params.get_config_set("APP_CFG")

        # Connect to graph DB
        self.logger.debug ("Connecting to: %s", params['GRAPH_URI'])
        self._conn = Neo4jConnection(uri=params['GRAPH_URI'], user=params['GRAPH_USER'], pwd=params['GRAPH_PASSWORD'])
        self.logger.debug ("Connection test: %s", self._conn.test_connection())

        if params.get('OBJECT_STORE_ENABLED',  False):
            self.logger.debug ("Connecting to: %s", params['MINIO_URI'])
            self._client = Minio(params['MINIO_URI'], access_key=params['MINIO_USER'], secret_key=params['MINIO_PASSWORD'], secure=False)
            try:
                for bucket in self._client.list_buckets():
                    self.logger.debug("Found bucket %s", bucket.name )
                self.logger.debug("OBJECT_STORE_ENABLED=True")
                params['OBJECT_STORE_ENABLED'] = True
            except:
                self.logger.warning("Could not connect to object store")
                self.logger.warning("Set OBJECT_STORE_ENABLED=False")
                params['OBJECT_STORE_ENABLED'] = False

        self._host = socket.gethostname() # portable to *docker* containers
        self.logger.debug ("Running on: %s", self._host)

        self.params = params
        self._attached = False

    def createNamespace(self, namespace_id: str):
        """
        Creates a namesapce, if it doesn't exist, else, tells you it exists

        :params: namespace_id - namespace value 
        """
        cohort = Node("cohort", namespace_id)
        create_res = self._conn.query(f""" MERGE (co:{cohort.get_create_str()}) RETURN co""")

        if len(create_res) == 1:
            self.logger.info(f"Namespace [{namespace_id}] created successfully")

        return self

    def setNamespace(self, namespace_id: str):
        """
        Sets the namespace for this container's commits, if it exists

        :params: namespace_id - namespace value 
        """
        self._namespace_id   = namespace_id
        self._namespace_node = Node("cohort", namespace_id)
        self._bucket_id      = namespace_id.lower().replace('_','-')
        
        self.logger.debug(f"Checking if [{namespace_id}] exists...")
        
        match_res = self._conn.query(f""" MATCH (co:{self._namespace_node.get_match_str()}) RETURN co""")

        if not len(match_res) == 1:
            raise RuntimeError( f"Namespace [{namespace_id}] does not exist, call .createNamespace() first!")

        if self.params.get('OBJECT_STORE_ENABLED',  False):
            if not self._client.bucket_exists(self._bucket_id):
                self._client.make_bucket(self._bucket_id)

        return self

    def createDatastore(self, container_id, container_type):
        """
        Checks if the node referenced by container_id is a valid container, queries the metastore for relevant metadata

        :params: container_id - unique container ID
        "params: type - the type of the container
        """

        if not container_type in ['generic', 'patient', 'accession', 'scan', 'slide', 'parquet']: 
            self.logger.warning (f"DataStore type [{container_type}] invalid, please choose from ['generic', 'patient', 'accession', 'scan', 'slide', 'parquet']" )

        if ":" in container_id: 
            self.logger.warning (f"Invalid container_id [{container_id}], only use alphanumeric characters")
        
        node = Node(container_type, container_id)
        node.set_namespace( self._namespace_id )

        create_res = self._conn.query(f""" MERGE (container:{node.get_create_str()}) RETURN container""")

        if len(create_res)==1: 
            self.logger.info(f"DataStore [{container_id}] of type [{container_type}] created or matched successfully!")
        else:
            self.logger.error("The container does not exists")
        
        return self
    
    def setDatastore(self, container_id):
        """
        Checks if the node referenced by container_id is a valid datastore, queries the metastore for relevant metadata

        :params: container_id - the unique container ID, either as an integer (neo4j autopopulated ID) or as a string (the Qualified Path)
        """
        self._attached = False
        self.logger.info ("Lookup ID: %s", container_id)

        # Figure out how to match the node
        if isinstance(container_id, str) and not "uid://" in container_id: 
            node = Node("generic", container_id)
            node.set_namespace( self._namespace_id )
            print (node.get_address())
            match_clause = f"""WHERE container.qualified_address = '{node.get_address()}' """
        elif isinstance(container_id, str) and "uid://" in container_id: 
            match_clause = f"""WHERE id(container) = {container_id.replace('uid://', '')} """
        elif isinstance(container_id, int):
            match_clause = f"""WHERE id(container) = {container_id} """
        else:
            raise RuntimeError("Invalid container_id type not (str, int)")

        # Run query
        res = self._conn.query(f"""
            MATCH (container) {match_clause}
            RETURN id(container), labels(container), container.type, container.name, container.namespace, container.qualified_address"""
        )
        
        # Check if the results are singleton (they should be... since we only query unique IDs!!!) 
        if res is None or len(res) == 0: 
            self.logger.warning (f"DataStore [{container_id}] does not exist, you can try creating it first with createContainer()")
            return self

        # Set some potentially import parameters
        self._datastore_id  = res[0]["id(container)"]
        self._name          = res[0]["container.name"]
        self._qualifiedpath = res[0]["container.qualified_address"]
        self._type          = res[0]["container.type"]
        self._labels        = res[0]["labels(container)"]
        self.address        = res[0]["container.qualified_address"]

        # Containers need to have a qualified path
        if self._qualifiedpath is None: 
            self.logger.warning ("Found, however not valid container object, containers must have a name, namespace, and qualified path")
            return self
        
        # Set match clause to id
        self.logger.debug ("Match on: %s", self._datastore_id)

        # Attach
        cohort = Node("cohort", self._namespace_id)
        if not len(self._conn.query(f""" MATCH (co:{cohort.get_match_str()}) MATCH (container) WHERE id(container) = {self._datastore_id} MERGE (co)-[:INCLUDE]->(container) RETURN co,container """)) == 1:
            self.logger.warning ( "Cannot attach, tried [%s]", f""" MATCH (co:{cohort.get_match_str()}) MATCH (container) WHERE id(container) = {self._datastore_id} MERGE (co)-[:INCLUDE]->(container) RETURN co,container """)
            return self

        # Let us know attaching was a success! :)
        self.logger = init_logger(f'container-{self.address}.log')
        self.logger.info ("Successfully attached to %s container id=%s @ %s", self._type, self._datastore_id, self.address)
        self._attached = True

        return self

    def isAttached(self):
        """
        Returns true if container was properly attached (i.e. checks in setDatastore succeeded), else False
        """
        self.logger.debug ("Attached: %s", self._attached)
        return self._attached


    def get(self, type, name):
        """
        Query graph DB container node for dependent data nodes, and return one 
        Parses the path field URL for various cases, and sets the node.data an node.aux attribute with a corrected path
        Note: namespace is not a default filter for get nodes, but is for adding them (i.e., one can write data under a different namespace)

        :params: type - the type of data designed 
            e.g. radiomics, mha, dicom, png, svs, geojson, etc.
        :params: name - can be used to filter nodes
            e.g. name of the node in the subspace of the container (e.g. generate-mhd)
        :example: get("mhd", "generate-mhd") gets data of type "mhd" generated from the method "generate-mhd" in this container's context/subspace
        """
        assert self.isAttached()

        query = f"""MATCH (container)-[:HAS_DATA]-(data:{type}) WHERE id(container) = {self._datastore_id} AND data.name='{type}-{name}' AND data.namespace='{self._namespace_id}' RETURN data"""

        self.logger.debug(query)
        res = self._conn.query(query)

        # Catches bad queries
        # If successfull query, reconstruct a Node object
        if res is None:
            self.logger.warning(f"get() query failed, data.name='{type}-{name}' returning None")
            return None
        elif len(res) == 0: 
            self.logger.warning(f"get() found no nodes, data.name='{type}-{name}' returning None")
            return None
        elif len(res) > 1: 
            self.logger.warning(f"get() found many nodes (?), data.name='{type}-{name}' returning None")
            return None
        else:
            node = Node(res[0]['data']['type'], res[0]['data']['name'], dict(res[0]['data'].items()))
            self.logger.debug ("Query Successful:")
            self.logger.debug (node)

        node.set_data(node.properties.get('data', None))
        node.set_aux (node.properties.get('aux', None))

        return node
    
    @staticmethod
    def run(namespace, container_id, pipeline):
        """
        Runner for pipelined jobs
        """
        for func in pipeline: 
            module = func[0]
            params = func[1]
            module (cohort_id=namespace, container_id=container_id, method_data=params)

    def runLocal(self, pipeline):
        """ 
        Run a pipeline in the main thread, blocking.

        :params: pipeline - an ordered list of (function, params) tuples to execute
        """
        self.run (self._namespace_id, self._name, pipeline)

    def runProcessPoolExecutor(self, pipeline, executor):
        """ 
        Use a process pool executor to run full pipelines in background

        :params: pipeline - an ordered list of (function, params) tuples to execute
        :params: executor - a ProcessPoolExecutor passed from a parent script
        """

        assert isinstance(executor, ProcessPoolExecutor)
        return executor.submit(self.run, self._namespace_id, self._name, pipeline)

    def runDaskDistributed(self, pipeline, client):
        """ 
        Submit functions to dask workers.
        Dask can track dependencies via a semaphore future, so we pass that explicitly and submit each function individually 

        :params: pipeline - an ordered list of (function, params) tuples to execute
        :params: client - a dask client
        """
        from dask.distributed   import Client
        
        assert isinstance(client, Client)
        future = client.submit (bootstrap, self._name)
        for func in pipeline: 
            module = func[0]
            params = func[1]
            future = client.submit (module, self._namespace_id, self._name, params, semaphore=future)
        return future


    def put(self, node: Node):
        """
        Adds a node to a temporary dictonary that will be used to save/commit nodes to the relevant databases
        If you add the same node under the same name, no change as the 
        Decorates the node with the container's namespace

        :param: node - node object
        """
        assert isinstance(node, Node)
        assert self.isAttached()

        self.logger.info(f"Adding node: {node}")

        # Decorate with the container namespace 
        node.set_namespace( self._namespace_id, self._name )
        node._datastore_id = self._datastore_id

        # Set node data object(s) only if there is a path and the object store is enabled
        node.objects = []
        if node.data is not None and self.params.get("OBJECT_STORE_ENABLED", False):
            node.properties['object_bucket'] = f"{self._bucket_id}"
            node.properties['object_folder'] = f"{self._name}/{node.name}"

            data_path = pathlib.Path( node.data )

            if data_path.is_file(): 
                node.objects.append( data_path )
            
            if data_path.is_dir(): 
                # TODO: enable extention in glob via something?
                for path in data_path.glob("*.*"): 
                    node.objects.append(path)        

            self.logger.info ("Node has %s pending object commits",  len(node.objects))

        # Set node aux object only if a path and the object store is enabled
        if node.aux is not None and self.params.get("OBJECT_STORE_ENABLED", False):
            node.properties['object_bucket'] = f"{self._bucket_id}"
            node.properties['object_folder'] = f"{self._name}/{node.name}"
            node.objects.append( pathlib.Path( node.aux ))
            self.logger.info ("Node has %s pending object commits",  len(node.objects))

        # Add to node commit dictonary
        self.logger.info ("Adding: %s", node.get_address())
                
        self._conn.query( f""" 
            MATCH (container) WHERE id(container) = {node._datastore_id}
            MERGE (container)-[:HAS_DATA]->(da:{node.get_match_str()})
                ON MATCH  SET da = {node.get_map_str()}
                ON CREATE SET da = {node.get_map_str()} 
            """ )
                
        if self.params.get("OBJECT_STORE_ENABLED", False):
            future_uploads = []
            executor = ThreadPoolExecutor(max_workers=4)

            object_bucket = node.properties.get("object_bucket")
            object_folder = node.properties.get("object_folder")
            for p in node.objects:
                future = executor.submit(self._client.fput_object, object_bucket, f"{object_folder}/{p.name}", p, part_size=250000000)
                future_uploads.append(future)
           
        n_count_futures = 0
        n_total_futures = len (future_uploads)
        for future in as_completed(future_uploads):
            try:
                data = future.result()
            except:
                self.logger.exception('Bad upload: generated an exception:')
            else:
                n_count_futures += 1
                if n_count_futures < 10: self.logger.info("Upload successful with etag: %s", data[0])
                if n_count_futures < 1000 and n_count_futures % 100 == 0: self.logger.info("Uploaded [%s/%s]", n_count_futures, n_total_futures)
                if n_count_futures % 1000 == 0: self.logger.info("Uploaded [%s/%s]", n_count_futures, n_total_futures)

        self.logger.info("Uploaded [%s/%s]", n_count_futures, n_total_futures)
        if self.params.get("OBJECT_STORE_ENABLED", False):
            self.logger.info("Shutdown executor %s", executor)                
            executor.shutdown()    
        self.logger.info("Done saving all records!!")

    def add(self, *args): 
        self.logger.warning ("Datastore.add() has been depreciated")
    def saveAll(self, *args): 
        self.logger.warning ("Datastore.saveAll() has been depreciated")
