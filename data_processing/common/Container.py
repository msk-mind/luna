from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.Node import Node
from data_processing.common.config import ConfigSet

import os, socket, pathlib, logging
from minio import Minio

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

class Container(object):
    """
    Container: an abstraction with an id, name, namespace, type, and a list of associated data nodes

    Interfaces with a metadata store (graph DB) and raw file stores (gpfs, potentially others)

    Handles the matching and creation of metadata

    Example usage:
    $ container = data_processing.common.GraphEnum.Container( params ).setNamespace("test").lookupAndAttach("1.2.840...")
        > Connecting to: neo4j://localhost:7687
        > Connection successfull: True
        > Running on: localhost
        > Lookup ID: 1.2.840...
        > Found: [<Record id(container)=7091 labels(container)=['scan'] container.type='scan' container.name='1.2.840...>]
        > Match on: WHERE id(container) = 7091 
        > Successfully attached to: scan 1.2.840...

    $ node = Node("dicom", "DCM-0123", {"Modality":"CT", "path":"file:/some/path/1.dcm"})

    $ container.add(node)
        > Adding: test-0000
          Container has 1 pending commits

    $ container.saveAll()
        > Committing dicom:globals{ hash: 'abc123' name: 'my-dicom', qualified_address: 'test::1.2.840::my-dicom', namespace: 'test', type: 'dicom' , path: 'file:/some/path/1.dcm'}

    $ node = container.listData("dicom")
        > ----------------------------------------------------------------------------------------------------
          name: DCM-0123
          type: dicom
          properties: 
          - type: 'dicom'
          - qualified_address: 'est::1.2.840::my-dicom'
          - path: 'file:/some/path/1.dcm'
          - namespace: '3'
          - Modality: 'CT'
          - name: 'my-dicom'
          ----------------------------------------------------------------------------------------------------
    $ container.get("dicom", "my-dicom").path
        > /some/path/1.dcm

    $ container.get("dicom", "my-dicom").properties['Modality']
        > 'CT'
    
    The container includes a logging method:
    $ container.logger.info("I am processing the CT")
        > 'yyyy-mm-dd h:m:s,ms - Container [1] - INFO - I am processing the CT'

    
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

        #self.logger = init_logger("common-container.log", "Container [empty]")
        self.logger = logging.getLogger(__name__)

        if isinstance(params, ConfigSet):
            params=params.get_config_set("APP_CFG")

        # Connect to graph DB
        self.logger.info ("Connecting to: %s", params['GRAPH_URI'])
        self._conn = Neo4jConnection(uri=params['GRAPH_URI'], user=params['GRAPH_USER'], pwd=params['GRAPH_PASSWORD'])
        self.logger.info ("Connection test: %s", self._conn.test_connection())

        if params.get('OBJECT_STORE_ENABLED',  False):
            self.logger.info ("Connecting to: %s", params['MINIO_URI'])
            self._client = Minio(params['MINIO_URI'], access_key=params['MINIO_USER'], secret_key=params['MINIO_PASSWORD'], secure=False)
            try:
                for bucket in self._client.list_buckets():
                    self.logger.debug("Found bucket %s", bucket.name )
                self.logger.info("OBJECT_STORE_ENABLED=True")
                params['OBJECT_STORE_ENABLED'] = True
            except:
                self.logger.warning("Could not connect to object store")
                self.logger.warning("Set OBJECT_STORE_ENABLED=False")
                params['OBJECT_STORE_ENABLED'] = False

        self._host = socket.gethostname() # portable to *docker* containers
        self.logger.info ("Running on: %s", self._host)

        self.params = params
        self._node_commits    = {}

    
    def setNamespace(self, namespace_id: str):
        """
        Sets the namespace for this container's commits

        :params: namespace_id - namespace value 
        """
        self._namespace_id = namespace_id
        self._bucket_id    = namespace_id.lower().replace('_','-')
        self.logger.info ("Container namespace: %s", self._namespace_id)

        if self.params.get('OBJECT_STORE_ENABLED',  False):
            if not self._client.bucket_exists(self._bucket_id):
                self._client.make_bucket(self._bucket_id)

        return self
    
    def lookupAndAttach(self, container_id):
        """
        Checks if the node referenced by container_id is a valid container, queries the metastore for relevant metadata

        :params: container_id - the unique container ID, either as an integer (neo4j autopopulated ID) or as a string (the Qualified Path)
        """
        self._attached = False
        self.logger.info ("Lookup ID: %s", container_id)

        # Figure out how to match the node
        if isinstance(container_id, str) and not container_id.isdigit(): 
            match_clause = f"""WHERE container.qualified_address = '{container_id.lower()}'"""
        elif (isinstance(container_id, str) and container_id.isdigit()) or (isinstance(container_id, int)):
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
            self.logger.warning ("Not found")
            return self

        # Set some potentially import parameters
        self.logger.info ("Found: %s", res)
        self._container_id  = res[0]["id(container)"]
        self._name          = res[0]["container.name"]
        self._qualifiedpath = res[0]["container.qualified_address"]
        self._type          = res[0]["container.type"]
        self._labels        = res[0]["labels(container)"]

        # Containers need to have a qualified path
        if self._qualifiedpath is None: 
            self.logger.warning ("Found, however not valid container object, containers must have a name, namespace, and qualified path")
            return self
        
        # Set match clause to id
        self.logger.debug ("Match on: %s", self._container_id)

        # Attach
        cohort = Node("cohort", self._namespace_id)
        if not len(self._conn.query(f""" MATCH (co:{cohort.get_match_str()}) MATCH (container) WHERE id(container) = {self._container_id} MERGE (co)-[:INCLUDE]->(container) RETURN co,container """ ))==1: 
            self.logger.warning ( "Cannot attach, tried [%s]", f""" MATCH (co:{cohort.get_match_str()}) MATCH (container) WHERE id(container) = {self._container_id} MERGE (co)-[:INCLUDE]->(container) RETURN co,container """)
            return self

        # Let us know attaching was a success! :)
        self.logger = logging.getLogger(f'Container [{self._container_id}]')
        self.logger.info ("Successfully attached to: %s %s %s", self._type, self._container_id, self._qualifiedpath)
        self._attached = True
        return self

    def isAttached(self):
        """
        Returns true if container was properly attached (i.e. checks in lookupAndAttach succeeded), else False
        """
        self.logger.info ("Attached: %s", self._attached)
        return self._attached


    def listTypes(self):
        """
        Query graph DB container node for dependent data nodes, and list them  

        :example: listTypes()
            > INFO - Available types: {'radiomics', 'dicom', 'mha', 'globals', 'dataset', 'nrrd', 'mhd'}
        """

        assert self.isAttached()

        res = self._conn.query(f"""
            MATCH (container)-[:HAS_DATA]-(data) 
            WHERE id(container) = {self._container_id}
            RETURN labels(data)"""
        )
        types = set()
        [types.update(rec['labels(data)']) for rec in res ]
        self.logger.info("Available types: %s", types)

    def listData(self, type, view=""):
        """
        Query graph DB container node for dependent data nodes, and list them  

        :params: type - the type of data designed 
            e.g. radiomics, mha, dicom, png, svs, geojson, etc.
        :params: view - can be used to filter nodes
            e.g. data.source='generateMHD'
            e.g. data.label='Right'
            e.g. data.namespace in ['default', 'my_cohort']

        :example: ls("png") gets data nodes of type "png" and prints the repr of each node
        """
        assert self.isAttached()

        # Prepend AND since the query runs with a WHERE on the container ID by default
        if view is not "": view = "AND " + view

        # Run query, subject to SQL injection attacks (but right now, our entire system is)
        res = self._conn.query(f"""
            MATCH (container)-[:HAS_DATA]-(data:{type}) 
            WHERE id(container) = {self._container_id}
            {view}
            RETURN data"""
        )
        # Catches bad queries
        # If successfull query, reconstruct a Node object
        if res is None or len(res) == 0: 
            return None
        else:
            [self.logger.info(Node(rec['data']['type'], rec['data']['name'], dict(rec['data'].items()))) for rec in res]


    def get(self, type, name):
        """
        Query graph DB container node for dependent data nodes, and return one 
        Parses the path field URL for various cases, and sets the node.path attribute with a corrected path
        Note: namespace is not a default filter for get nodes, but is for adding them (i.e., one can write data under a different namespace)

        :params: type - the type of data designed 
            e.g. radiomics, mha, dicom, png, svs, geojson, etc.
        :params: name - can be used to filter nodes
            e.g. name of the node in the subspace of the container (e.g. generate-mhd)
        :example: get("mhd", "generate-mhd") gets data of type "mhd" generated from the method "generate-mhd" in this container's context/subspace
        """
        assert self.isAttached()

        query = f"""MATCH (container)-[:HAS_DATA]-(data:{type}) WHERE id(container) = {self._container_id} AND data.name='{type}-{name}' AND data.namespace='{self._namespace_id}' RETURN data"""

        self.logger.debug(query)
        res = self._conn.query(query)

        # Catches bad queries
        # If successfull query, reconstruct a Node object
        if res is None:
            self.logger.error(f"get() query failed, data.name='{type}-{name}' returning None")
            return None
        elif len(res) == 0: 
            self.logger.error(f"get() found no nodes, data.name='{type}-{name}' returning None")
            return None
        elif len(res) > 1: 
            self.logger.error(f"get() found many nodes (?), data.name='{type}-{name}' returning None")
            return None
        else:
            node = Node(res[0]['data']['type'], res[0]['data']['name'], dict(res[0]['data'].items()))
            self.logger.debug ("Query Successful:")
            self.logger.debug (node)

        
        # Parse path (filepath) URI: more sophistic path logic to come (pulling from S3, external mounts, etc!!!!)
        # For instance, if it was like s3://bucket/test.dcm, this should pull the dicom to a temporary directory and set .path to that dir
        # This also might be dependent on filetype
        # Checks:
            # If like file:/path/to.dcm, strip file:
            # Else, just return path
        # TODO: change 'path' field to 'filepath_uri"?
        
        if "path" in node.properties.keys(): 
            path = node.properties["path"]
            if path.split(":")[0] == "file":
                node.path = pathlib.Path(path.split(":")[-1])
            else:
                node.path = pathlib.Path(path)
            
            node.static_path = str(node.path)
        
            # Output and check
            self.logger.info ("Resolved %s -> %s", node.properties["path"], node.path)

            # Check that we got it right, and this path is readable on the host system
            if not os.path.exists(node.path):
                raise RuntimeError("Invalid pathspec", node.path)
            self.logger.info ("Filepath is valid: %s", os.path.exists(node.path))

        if "file" in node.properties.keys(): 

            node.file = pathlib.Path(node.properties["file"])
            
            node.static_file = str(node.file)
        
            # Output and check
            self.logger.info ("Resolved %s -> %s", node.properties["file"], node.static_file)

            # Check that we got it right, and this path is readable on the host system
            if not os.path.exists(node.file):
                raise RuntimeError("Invalid filespec", node.file)
            self.logger.info ("File object is valid: %s", os.path.exists(node.file))

        return node
    
    def add(self, node: Node):
        """
        Adds a node to a temporary dictonary that will be used to save/commit nodes to the relevant databases
        If you add the same node under the same name, no change as the 
        Decorates the node with the container's namespace

        :param: node - node object
        """
        assert isinstance(node, Node)
        assert self.isAttached()

        # Decorate with the container namespace 
        node.set_namespace( self._namespace_id, self._name )
        node._container_id = self._container_id

        # Set node objects only if there is a path and the object store is enabled
        node.objects = []
        if "path" in node.properties.keys() and self.params.get("OBJECT_STORE_ENABLED", False):
            node.properties['object_bucket'] = f"{self._bucket_id}"
            node.properties['object_folder'] = f"{self._name}/{node.name}"
            for path in pathlib.Path(node.properties['path']).glob("*"): node.objects.append(path)        
            self.logger.info ("Node has %s pending object commits",  len(node.objects))

        if "file" in node.properties.keys() and self.params.get("OBJECT_STORE_ENABLED", False):
            node.properties['object_bucket'] = f"{self._bucket_id}"
            node.properties['object_folder'] = f"{self._name}/{node.name}"
            node.objects.append(pathlib.Path(node.properties['file']))
            self.logger.info ("Node has %s pending object commits",  len(node.objects))

        # Add to node commit dictonary
        self.logger.info ("Adding: %s", node.get_address())
        self._node_commits[node.get_address()] = node
        
        self.logger.info ("Container has %s pending node commits",  len(self._node_commits))

      
    def saveAll(self):
        """
        Tries to create nodes for all committed nodes
        """
        # Loop through all nodes in commit dictonary, and run query
        # Will fully overwrite existing nodes, since we assume changes in the FS already occured
        self.logger.info("Detaching container...")
        self._attached = False
        self.logger = logging.getLogger(__name__)


        future_uploads = []
        for n in self._node_commits.values():
            self.logger.info ("Committing %s", n.get_match_str())
            self._conn.query(f""" 
                MATCH (container) WHERE id(container) = {n._container_id}
                MERGE (container)-[:HAS_DATA]->(da:{n.get_match_str()})
                    ON MATCH  SET da = {n.get_map_str()}
                    ON CREATE SET da = {n.get_map_str()}
                """
            )

            if self.params.get("OBJECT_STORE_ENABLED", False):
                self.logger.info("Started minio executor with 4 threads")
                executor = ThreadPoolExecutor(max_workers=4)

                object_bucket = n.properties.get("object_bucket")
                object_folder = n.properties.get("object_folder")
                for p in n.objects:
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
        self.logger.info("Shutdown executor %s", executor)                
        executor.shutdown()    
        self.logger.info("Done saving all records!!")
