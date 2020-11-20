from enum import Enum

class Graph(object):

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
	value: Graph object - this could be a list of Graphs if needed..

	>>> GraphEnum['DICOM'].value.src
	'xnat_patient_id'
	"""
	DICOM = Graph("xnat_patient_id", "HAS_SCAN", "SeriesInstanceUID", "metadata.PatientName", "metadata.SeriesInstanceUID")
	PROJECT = Graph("project_name", "HAS_PX", "some_patient_id")
