from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.Node import Node

from minio import Minio
from concurrent.futures import ThreadPoolExecutor

import os, socket, pathlib, logging


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

        self.params=params

        # Connect to graph DB
        self.logger.info ("Connecting to: %s", params['GRAPH_URI'])
        self._conn = Neo4jConnection(uri=params['GRAPH_URI'], user=params['GRAPH_USER'], pwd=params['GRAPH_PASSWORD'])
        self.logger.info ("Connection test: %s", self._conn.test_connection())
        self._host = socket.gethostname() # portable to *docker* containers
        self.logger.info ("COnnecting to: %s", params['MINIO_URI'])
        self._client = Minio(params['MINIO_URI'], access_key=params['MINIO_USER'], secret_key=params['MINIO_PASSWORD'], secure=False)
        self.logger.info ("Running on: %s", self._host)
    
    def setNamespace(self, namespace_id: str):
        """
        Sets the namespace for this container's commits

        :params: namespace_id - namespace value 
        """
        self._namespace_id = namespace_id
        self.logger.info ("Container namespace: %s", self._namespace_id)
        if not self._client.bucket_exists(self._namespace_id):
            self._client.make_bucket(self._namespace_id)

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
            if not "::" in container_id: self.logger.warning ("Qualified path %s doesn't look like one...", container_id)
            self._match_clause = f"""WHERE container.qualified_address = '{container_id}'"""
        elif (isinstance(container_id, str) and container_id.isdigit()) or (isinstance(container_id, int)):
            self._match_clause = f"""WHERE id(container) = {container_id} """
        else:
            raise RuntimeError("Invalid container_id type not (str, int)")

        # Run query
        res = self._conn.query(f"""
            MATCH (container) {self._match_clause}
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
        self._node_commits    = {}

        # Containers need to have a qualified path
        if self._qualifiedpath is None: 
            self.logger.warning ("Found, however not valid container object, containers must have a name, namespace, and qualified path")
            return self
        
        # Set match clause to id
        self._match_clause = f"""WHERE id(container) = {self._container_id} """
        self.logger.debug ("Match on: %s", self._match_clause)

        # Attach
        cohort = Node("cohort", self._namespace_id)
        if not len(self._conn.query(f""" MATCH (co:{cohort.get_match_str()}) MATCH (container) {self._match_clause} MERGE (co)-[:INCLUDE]->(container) RETURN co,container """ ))==1: 
            self.logger.warning ( "Cannot attach")
            return self

        # Let us know attaching was a success! :)
        self.logger = logging.getLogger(f'Container [{self._container_id}]')
        self.logger.info ("Successfully attached to: %s %s", self._type, self._qualifiedpath)
        self._attached = True

        return self
    
    def isAttached(self):
        """
        Returns true if container was properly attached (i.e. checks in lookupAndAttach succeeded), else False
        """
        self.logger.info ("Attached: %s", self._attached)


    def listTypes(self):
        """
        Query graph DB container node for dependent data nodes, and list them  

        :example: listTypes()
            > INFO - Available types: {'radiomics', 'dicom', 'mha', 'globals', 'dataset', 'nrrd', 'mhd'}
        """

        # Run query, subject to SQL injection attacks (but right now, our entire system is)
        res = self._conn.query(f"""
            MATCH (container)-[:HAS_DATA]-(data) 
            {self._match_clause}
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
        # Prepend AND since the query runs with a WHERE on the container ID by default
        if view is not "": view = "AND " + view

        # Run query, subject to SQL injection attacks (but right now, our entire system is)
        res = self._conn.query(f"""
            MATCH (container)-[:HAS_DATA]-(data:{type}) 
            {self._match_clause}
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
        query = f"""MATCH (container)-[:HAS_DATA]-(data:{type})  {self._match_clause} AND data.name='{name}' AND data.namespace='{self._namespace_id}' RETURN data"""

        self.logger.info(query)
        res = self._conn.query(query)

        # Catches bad queries
        # If successfull query, reconstruct a Node object
        if res is None:
            self.logger.error("get() query failed, returning None")
            return None
        elif len(res) == 0: 
            self.logger.error("get() found no nodes, returning None")
            return None
        elif len(res) > 1: 
            self.logger.error("get() found many nodes (?) returning None")
            return None
        else:
            node = Node(res[0]['data']['type'], res[0]['data']['name'], dict(res[0]['data'].items()))
            self.logger.info ("Query Successful:")
            self.logger.info (node)

        
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
        
            # Output and check
            self.logger.info ("Resolved %s -> %s", node.properties["path"], node.path)

            # Check that we got it right, and this path is readable on the host system
            self.logger.info ("Filepath is valid: %s", os.path.exists(node.path))

        return node
    
    def add(self, node: Node):
        """
        Adds a node to a temporary dictonary that will be used to save/commit nodes to the relevant databases
        If you add the same node under the same name, no change as the 
        Decorates the node with the container's namespace

        :param: node - node object
        """
        assert isinstance(node, Node)

        # Add to node commit dictonary
        self.logger.info ("Adding: %s", node.name)
        self._node_commits[node.name] = node
        
        # Decorate with the container namespace 
        self._node_commits[node.name].set_namespace( self._namespace_id )
        self.logger.info ("Container has %s pending node commits",  len(self._node_commits))

        # Set node objects
        if "path" in node.properties.keys():
            node.objects = []
            node.properties['state'] = 'PENDING'
            node.properties['object_uri'] = f"s3://{self._namespace_id}/{node.name}"
            for path in pathlib.Path(node.properties['path']).glob("*"): node.objects.append(path)        

        self.logger.info ("Node has %s pending object commits",  len(node.objects))


    def saveAll(self):
        """
        Tries to create nodes for all committed nodes
        """
        # Loop through all nodes in commit dictonary, and run query
        # Will fully overwrite existing nodes, since we assume changes in the FS already occured
         
        for n in self._node_commits.values():
            self.logger.info ("Committing %s", n.get_create_str())
            self._conn.query(f""" 
                MATCH (container) {self._match_clause}
                MERGE (container)-[:HAS_DATA]->(da:{n.get_match_str()})
                    ON MATCH  SET da = {n.get_map_str()}
                    ON CREATE SET da = {n.get_map_str()}
                """
            )

            with ThreadPoolExecutor(max_workers=8) as executor:
                self.logger.info("Started object upload with 8 threads...")
                futures = []
                for p in n.objects:
                    futures.append(executor.submit(self._client.fput_object, self._namespace_id, f"{n.name}/{p.name}", p))
            
            self._conn.query(f""" 
                MATCH (container) {self._match_clause}
                MATCH (container)-[:HAS_DATA]->(da:{n.get_match_str()})
                SET da.state = 'VALID'
                """
            )
            self.logger.info("Done.")
           
