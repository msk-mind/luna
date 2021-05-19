from data_processing.pathology.common.build_bitmask import *
import numpy as np

xml_data_path = 'tests/data_processing/testdata/data/test-project/pathology_annotations/123456_annotation_from_halo.xml'

def test_get_roi():
   roi = xml2roi(xml_data_path)
   assert roi == ([1, 10], [40, 1])

def test_get_mask():
   roi = xml2mask(xml_data_path, shape=(20,50))
   assert list(np.bincount(roi.flatten())) == [929,  71] 
   assert list(roi[5,:7]) == [False, False,  True,  True, True,  True, False]
