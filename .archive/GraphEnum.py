from enum import Enum


class NodeType(object):
    """
    NodeType object defines the schema used to populate a graph node.

    :param: node_type: node type. e.g. scan
    :param: name: required node name. e.g. scan-123
    :param: schema: list of column names. e.g. slide_id
    """

    def __init__(self, node_type, name, schema=[]):

        self.type = node_type
        self.name = name
        self.schema = schema

    def get_all_schema(self):
        """
        Name is a required field, but it's still a property of this node.
        Return the properties including the name property!
        """
        return self.schema + [self.name]


class Graph(object):
    """
    Graph object that stores src-[relationship]-target information.
    This object is used get data from existing delta tables to populate the graph database.

    :param: src: NodeType - source node
    :param: relationship: str - relationship name e.g. HAS_PX, HAS_RECORD etc
    :param: target: NodeType - target node
    """

    def __init__(self, src, relationship, target):

        self.src = src
        self.relationship = relationship
        self.target = target


class GraphEnum(Enum):
    """
    Defines Graph relationships.
    We assume that all properties come from the table defined in GraphEnum.name

    name: data type
    value: list of Graphs - to accomodate multiple relationship update

    >>> GraphEnum['DICOM'].value[0].src.type
    'patient'
    """

    accession_radiology_dicom = NodeType("accession", "metadata.AccessionNumber")
    accession_radiology = NodeType("accession", "accession_number")
    scan = NodeType("scan", "metadata.SeriesInstanceUID")

    # radiology
    DICOM = [
        Graph(
            NodeType("patient", "metadata.PatientID"),
            "HAS_CASE",
            accession_radiology_dicom,
        ),
        Graph(accession_radiology_dicom, "HAS_SCAN", scan),
        Graph(
            scan,
            "HAS_DATA",
            NodeType("DicomSeries", "metadata.SeriesInstanceUID", ["path"]),
        ),
    ]

    MHA = [
        Graph(
            accession_radiology,
            "HAS_DATA",
            NodeType(
                "VolumetricLabel", "scan_annotation_record_uuid", ["path", "label"]
            ),
        )
    ]

    MHD = [
        Graph(
            accession_radiology,
            "HAS_DATA",
            NodeType("VolumetricImage", "scan_annotation_record_uuid", ["path"]),
        )
    ]

    PNG = [
        Graph(
            scan,
            "HAS_DATA",
            NodeType(
                "DicomImageSeries",
                "png_record_uuid",
                ["metadata.SeriesInstanceUID", "label"],
            ),
        )
    ]
    FEATURE = [
        Graph(
            scan,
            "HAS_DATA",
            NodeType(
                "DicomImageSeries",
                "feature_record_uuid",
                ["metadata.SeriesInstanceUID", "label"],
            ),
        )
    ]

    # pathology
    slide = NodeType("slide", "slide_id", ["sv_project_id", "slideviewer_path"])
    REGIONAL_BITMASK = [
        Graph(
            slide,
            "HAS_DATA",
            NodeType(
                "RegionalAnnotationBitmap",
                "bmp_record_uuid",
                ["user", "date_updated", "latest"],
            ),
        )
    ]

    # regional concat geojson table contains regional geojson table + concatenated jsons
    REGIONAL_CONCAT_GEOJSON = [
        Graph(
            slide,
            "HAS_DATA",
            NodeType(
                "RegionalAnnotationJSON",
                "concat_geojson_record_uuid",
                ["labelset", "date_updated", "latest"],
            ),
        )
    ]

    POINT_RAW_JSON = [
        Graph(
            slide,
            "HAS_DATA",
            NodeType(
                "PointAnnotation",
                "sv_json_record_uuid",
                ["user", "date_updated", "latest"],
            ),
        )
    ]

    POINT_GEOJSON = [
        Graph(
            slide,
            "HAS_DATA",
            NodeType(
                "PointAnnotationJson",
                "geojson_record_uuid",
                ["labelset", "date_updated", "latest"],
            ),
        )
    ]
