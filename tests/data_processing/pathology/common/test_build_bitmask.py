from data_processing.pathology.common.utils import *
import numpy as np

xml_data_path = 'tests/data_processing/testdata/data/test-project/pathology_annotations/123456_annotation_from_halo.xml'

def test_convert_halo_xml_to_roi():
   roi = convert_halo_xml_to_roi(xml_data_path)
   assert roi == ([1, 10], [40, 1])

def test_convert_xml_to_mask():
   roi = convert_xml_to_mask(xml_data_path, shape=(20,50), annotation_name="Tumor")
   assert list(np.bincount(roi.flatten())) == [929,  71] 
   assert list(roi[5,:7]) == [False, False,  True,  True, True,  True, False]
