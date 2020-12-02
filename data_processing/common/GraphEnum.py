from enum import Enum

class Graph(object):
	"""
	Graph object that stores src-[relationship]-target information.
	This object is used get data from existing delta tables to populate the graph database.
	src_column_name and target_column_name are "id" columns for querying the tables.

	:params:
	src: source node name
	relationship: relationship name e.g. HAS_PX, HAS_RECORD etc
	target: target node name
	src_column_name: (optional) source column name. if not provided, this is set to `src`.
	target_column_name: (optional) target column name. if not provided, this is set to `target`
	"""
	def __init__(self, src, relationship, target, src_column_name=None, target_column_name=None):
		# src node
		self.src = src
		self.relationship = relationship
		# target node
		self.target = target

		# optional fields
		if src_column_name is None:
			self.src_column_name = src
		else:
			self.src_column_name = src_column_name

		if target_column_name is None:
			self.target_column_name = target
		else:
			self.target_column_name = target_column_name



class GraphEnum(Enum):
	"""
	name: table name
	value: list of Graphs - to accomodate multiple relationship update

	>>> GraphEnum['DICOM'].value[0].src
	'xnat_patient_id'
	"""
	DICOM = [Graph("xnat_patient_id", "HAS_SCAN", "SeriesInstanceUID", "metadata.PatientName", "metadata.SeriesInstanceUID")]
	#PROJECT = Graph("project_name", "HAS_PX", "dmp_patient_id")
