import pandas as pd
from pyarrow import flight


class RedcapToMinioConnector:
    """
    A connector to load tables from REDcap to Minio/S3
    Parameters:
    ----------
    REDCAP_URI: Redcap API URI
    REDCAP_API_TOKEN: Redcap API token
    MINIO_URI: Minio URI
    MINIO_USER: Minio username
    MINIO_PASSWORD: Associated password
    """
    def __init__(self, REDCAP_URI, REDCAP_API_TOKEN, MINIO_URI, MINIO_USER, MINIO_PASSWORD ):
        self.REDCAP_URI = REDCAP_URI
        self.REDCAP_API_TOKEN = REDCAP_API_TOKEN
        self.MINIO_URI = MINIO_URI
        self.MINIO_USER = MINIO_USER
        self.MINIO_PASSWORD = MINIO_PASSWORD

    def _pull_redcap_instrument(self, report_id, fields_to_include=None):

        data_meta = {
            'token': self.REDCAP_API_TOKEN,
            'content': 'metadata',
            'format': 'json',
            'returnFormat': 'json'
        }
        r = requests.post(self.REDCAP_URI, data=data_meta)
        df = pd.DataFrame(data=r.json())
        exlude_phi_columns = df [ df['identifier']=='y']['field_name']

        data_report = {
            'token': self.REDCAP_API_TOKEN,
            'content': 'report',
            'format': 'json',
            'report_id': report_id,
            'csvDelimiter': '',
            'rawOrLabel': 'raw',
            'rawOrLabelHeaders': 'raw',
            'exportCheckboxLabel': 'false',
            'returnFormat': 'json'
        }
        r = requests.post(self.REDCAP_URI, data=data_report)
        print('HTTP Status: ' + str(r.status_code))
        df = pd.DataFrame(data=r.json()).drop(columns=exlude_phi_columns, errors='ignore' )
        if fields_to_include is None: 
            return df
        else:
            return df[fields_to_include]

    def _push_dataframe(self, df, project="LUNG_18-193", table_name="ID_INVENTORY"):
        print (f"Writing to: s3://{project}/tables/{table_name}")
        dd.to_parquet(
            df=dd.from_pandas(df, npartitions=2), 
            path=f's3://{project}/tables/{table_name}', 
            storage_options={"key":self.MINIO_USER, "secret": self.MINIO_PASSWORD, "client_kwargs": {"endpoint_url":self.MINIO_URI}}
        )
        
    def run(self, input_report_id, project, output_table_name, fields_to_include=None ):
        """
        Write REDcap report as a parquet table at project/tables/table_name
        Parameters:
        ----------
        input_report_id: The report ID of the REDcap report to ingest
        project: Project ID
        output_table_name: Destination table name
        fields_to_include: Optional - filter fields of report to writeout
        """
        df = self._pull_redcap_instrument(input_report_id, fields_to_include)
        print (f"Loaded {len(df)} records!")
        self._push_dataframe(df, project, output_table_name)
        

class DremioClientAuthMiddlewareFactory(flight.ClientMiddlewareFactory):
    """A factory that creates DremioClientAuthMiddleware(s)."""

    def __init__(self):
        self.call_credential = []

    def start_call(self, info):
        return DremioClientAuthMiddleware(self)

    def set_call_credential(self, call_credential):
        self.call_credential = call_credential


class DremioClientAuthMiddleware(flight.ClientMiddleware):
    """
    A ClientMiddleware that extracts the bearer token from 
    the authorization header returned by the Dremio 
    Flight Server Endpoint.
    Parameters
    ----------
    factory : ClientHeaderAuthMiddlewareFactory
        The factory to set call credentials if an
        authorization header with bearer token is
        returned by the Dremio server.
    """

    def __init__(self, factory):
        self.factory = factory

    def received_headers(self, headers):
        auth_header_key = 'authorization'
        authorization_header = []
        for key in headers:
            if key.lower() == auth_header_key: authorization_header = headers.get(auth_header_key)
        self.factory.set_call_credential([
            b'authorization', authorization_header[0].encode("utf-8")])

        
class DremioDataframeConnector:
    """
    A connector that interfaces with a Dremio instance/cluster via Apache Arrow Flight for fast read performance
    Parameters
    ----------
    scheme: connection scheme
    hostname: host of main dremio name
    flightport: which port dremio exposes to flight requests
    dremio_user: username to use
    dremio_password: associated password
    connection_args: anything else to pass to the FlightClient initialization
    """
    def __init__(self, scheme, hostname, flightport, dremio_user, dremio_password, connection_args):
        # Skipping tls...

        # Two WLM settings can be provided upon initial authneitcation
        # with the Dremio Server Flight Endpoint:
        # - routing-tag
        # - routing queue
        initial_options = flight.FlightCallOptions(headers=[
            (b'routing-tag', b'test-routing-tag'),
            (b'routing-queue', b'Low Cost User Queries')
        ])
        client_auth_middleware = DremioClientAuthMiddlewareFactory()
        client = flight.FlightClient(f"{scheme}://{hostname}:{flightport}", middleware=[client_auth_middleware], **connection_args)
        self.bearer_token = client.authenticate_basic_token(dremio_user, dremio_password, initial_options)
        print('[INFO] Authentication was successful')
        self.client = client
        
    def run(self, project, table_name):
        """
        Return the virtual table at project(or "space").table_name as a pandas dataframe
        Parameters:
        ----------
        project: Project ID to read from
        table_name:  Table name to load
        """
        sqlquery = f'''SELECT * FROM "{project}"."{table_name}"'''
        # Get table from our dicom segments
        flight_desc = flight.FlightDescriptor.for_command(sqlquery)
        print('[INFO] Query: ', sqlquery)

        options = flight.FlightCallOptions(headers=[self.bearer_token])
        schema = self.client.get_schema(flight_desc, options)
        print('[INFO] GetSchema was successful')
        print('[INFO] Schema: ', schema)

        # Get the FlightInfo message to retrieve the Ticket corresponding
        # to the query result set.
        flight_info = self.client.get_flight_info(flight.FlightDescriptor.for_command(sqlquery),
            options)
        print('[INFO] GetFlightInfo was successful')
        print('[INFO] Ticket: ', flight_info.endpoints[0].ticket)

        # Retrieve the result set as a stream of Arrow record batches.
        reader = self.client.do_get(flight_info.endpoints[0].ticket, options)
        print('[INFO] Reading query results from Dremio')
        return reader.read_pandas()

    def get_table(self, sqlquery):
        """
        Return the virtual table at project(or "space").table_name as a pandas dataframe
        Parameters:
        ----------
        project: Project ID to read from
        table_name:  Table name to load
        """
        # Get table from our dicom segments
        flight_desc = flight.FlightDescriptor.for_command(sqlquery)
        print('[INFO] Query: ', sqlquery)

        options = flight.FlightCallOptions(headers=[self.bearer_token])
        schema = self.client.get_schema(flight_desc, options)
        print('[INFO] GetSchema was successful')
        print('[INFO] Schema: ', schema)

        # Get the FlightInfo message to retrieve the Ticket corresponding
        # to the query result set.
        flight_info = self.client.get_flight_info(flight.FlightDescriptor.for_command(sqlquery),
            options)
        print('[INFO] GetFlightInfo was successful')
        print('[INFO] Ticket: ', flight_info.endpoints[0].ticket)

        # Retrieve the result set as a stream of Arrow record batches.
        reader = self.client.do_get(flight_info.endpoints[0].ticket, options)
        print('[INFO] Reading query results from Dremio')
        return reader.read_pandas()
