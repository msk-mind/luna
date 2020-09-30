from neo4j import GraphDatabase
from neo4j import __version__ as neo4j_version

from pyspark.sql.types import StringType,StructType,StructField
from termcolor import colored 

def pretty_path(path): 
    to_print = ''
    for i,x in enumerate(path): 
        if type(x)==dict: 
            node_desc = '(' + ','.join([key+":"+x[key] for key in x.keys()]) + ')'
            if   i==0: 		   to_print += "SOURCE:" + node_desc # First
            elif i==(len(path)-1): to_print += "SINK:" + node_desc # Last
            else:                  to_print += node_desc # Middle
        if type(x)==str: 
            to_print += '-[' + x + ']-'
    return to_print

class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None):
        """
        Runs a cyper query against the initalized driver
        """
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

    def test_count(self):
        """
        Get node(s) given a value input
        """
        result = self.query(f"""
            MATCH (n) RETURN n
        """
        )
        node_count = len(result)
        print (f"Successfully connected to {node_count} nodes!")
    def match_concept_node(self, QUERY_ID):
        """
        Get node(s) given a value input
        """
        result = self.query(f"""
            MATCH (concept_node)
            WHERE concept_node.value = '{QUERY_ID}'
            RETURN concept_node"""
        )
        return [(x.data(),) for x in result]

    def commute_source_id_to_spark(self, sc, sqlc, SOURCE_TYPE, SINK_TYPE, QUERY_ID ):
        """
        Spark connector for an input source id
        Returns a dataframe with one column named the sink/target ID
        """
        result = self.query(f"""
            MATCH (source:{SOURCE_TYPE})-[:ID_LINK*]-(sink:{SINK_TYPE}) \
            WHERE source.value = '{QUERY_ID}' \
            RETURN source,sink """
        )
        cSchema = StructType([StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(sc.parallelize([(x.data()['sink']['value'],) for x in result]),schema=cSchema)

    def commute_source_id_to_spark_query(self, sc, sqlc, WHERE_CLAUSE, SINK_TYPE ):
        """
        Spark connector for an input source id
        Returns a dataframe with one column named the sink/target ID
        """
        result = self.query(f"""
            MATCH path=(source)-[r*]-(sink:{SINK_TYPE}) \
            {WHERE_CLAUSE} \
            RETURN source,sink,path """
        )
        for x in result: print (pretty_path(x.data()['path']))
        cSchema = StructType([StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(([(x.data()['sink']['value'],) for x in result]),schema=cSchema)

    def create_id_lookup_table_where(self, sqlc, source, sink, r="ID_LINK|HAS_RECORD", WHERE_CLAUSE=""):
        """
        Spark connector for an input source id
        Returns a dataframe with one column named the sink/target ID
        """
        print (f""">>> QUERY >>> \n\tMATCH (source:{source})-[r:{r}*]-(sink:{sink}), \n\tpath=shortestPath( (source)-[:{r}*..15]-(sink) ) \n\t{WHERE_CLAUSE} RETURN DISTINCT source,sink,path""")

        result = self.query(f"""
            MATCH (source:{source})-[r:{r}*]-(sink:{sink}), path=shortestPath( (source)-[:{r}*..15]-(sink) ) \
            {WHERE_CLAUSE} \
            RETURN DISTINCT source,sink,path 
            """
        )
        if result is None: 
            print ("Improper query returning null")
            return None 
        cSchema = StructType([StructField(source, StringType(), True), StructField(sink, StringType(), True), StructField("pathspec", StringType(), True)])
        return sqlc.createDataFrame(([(x.data()['source']['value'],x.data()['sink']['value'], pretty_path(x.data()['path'])) for x in result]),schema=cSchema)

    def create_id_lookup_table(self, sc, sqlc, SOURCE_TYPE, SINK_TYPE, QUERY_ID ):
        """
        Spark connector for an input source id
        Returns a dataframe with two ID columns as specified by SOURCE_TYPE and SINK_TYPE
        """
        result = self.query(f"""
            MATCH (source:{SOURCE_TYPE})-[:ID_LINK|HAS_RECORD*]-(sink:{SINK_TYPE}) \
            WHERE source.value = '{QUERY_ID}' \
            RETURN source,sink """
        )
        cSchema = StructType([StructField(SOURCE_TYPE, StringType(), True), StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(sc.parallelize([(x.data()['source']['value'],x.data()['sink']['value']) for x in result]),schema=cSchema)

    def commute_cohort_id_to_spark (self, sc, sqlc, SOURCE_TYPE, SINK_TYPE, QUERY_ID):
        """
        Spark connector for an input cohort id
        Returns a dataframe with one column named the sink/target ID
        """
        result = self.query(f"""
            MATCH (source:{SOURCE_TYPE})-[:COHORT_LINK*1..1]-(entry_node)
            WHERE source.value = '{QUERY_ID}' \
            MATCH (entry_node)-[:ID_LINK*]-(sink:{SINK_TYPE})
            RETURN source,sink """
        )
        cSchema = StructType([StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(sc.parallelize([(x.data()['sink']['value'],) for x in result]),schema=cSchema)

    def commute_record_id_to_spark (self, sc, sqlc, SOURCE_TYPE, SINK_TYPE):
        """
        Spark connector for an input cohort id
        Returns a dataframe with one column named the sink/target ID
        """
        result = self.query(f"""
            MATCH (source:{SOURCE_TYPE})-[:HAS_RECORD*1..1]-(sink:{SINK_TYPE})
            RETURN source,sink """
        )
        cSchema = StructType([StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(sc.parallelize([(x.data()['sink']['value'],) for x in result]),schema=cSchema)

    def commute_sink_id_to_spark (self, sc, sqlc, SINK_TYPE, QUERY_ID):
        """
        A pass through function for consistency
        """
        cSchema = StructType([StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(sc.parallelize([(x,) for x in [QUERY_ID]]),schema=cSchema)

    def commute_all_sink_id(self, sc, sqlc, SINK_TYPE):
        """
        Spark connector for an input source id
        Returns a dataframe with one column named the sink/target ID
        """
        result = self.query(f"""
            MATCH (sink:{SINK_TYPE}) \
            RETURN sink """
        )
        cSchema = StructType([StructField(SINK_TYPE, StringType(), True)])
        return sqlc.createDataFrame(sc.parallelize([(x.data()['sink']['value'],) for x in result]),schema=cSchema)
