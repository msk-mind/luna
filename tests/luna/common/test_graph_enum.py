# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: aukermaa@mskcc.org
'''
import pytest

from luna.common.Node import Node, CONTAINER_TYPES, ALL_DATA_TYPES

def test_patient_create():
    create_string = Node("patient", "my_patient", properties={"namespace":"my_cohort", "Description":"a patient"}).get_create_str()
    assert "patient:globals" in create_string    
    assert "qualified_address: 'my_cohort::my_patient'"   in create_string
    assert "name"  in create_string
    assert "Description"   in create_string

def test_patient_match():
    match_string  = Node("patient", "my_patient", properties={"namespace":"my_cohort", "Description":"a patient"}).get_match_str()
    assert "patient" in match_string    
    assert "qualified_address: 'my_cohort::my_patient'"   in match_string
    assert "Description"  not in match_string


def test_cohort_create():
    create_string = Node("cohort", "my_cohort", properties={"Description":"a cohort"}).get_create_str()
    assert "cohort:globals" in create_string    
    assert "qualified_address: 'my_cohort::my_cohort'"  in create_string
    assert "name"  in create_string
    assert "Description"  in create_string

def test_cohort_match():
    match_string = Node("cohort", "my_cohort", properties={"Description":"a cohort"}).get_match_str()
    assert "cohort" in match_string    
    assert "qualified_address: 'my_cohort::my_cohort'"   in match_string
    assert "Description"  not in match_string


def test_metadata_create():
    properties = {}
    properties['namespace'] = "my_cohort"
    properties['dim'] = 3

    create_string = Node("VolumetricImage", "SCAN-001", properties=properties).get_create_str()
    assert "VolumetricImage:globals" in create_string    
    assert "qualified_address: 'my_cohort::VolumetricImage-SCAN-001'"  in create_string
    assert "dim"  in create_string

def test_metadata_match():
    properties = {}
    properties['namespace'] = "my_cohort"
    properties['dim'] = 3

    match_string = Node("VolumetricImage", "SCAN-002", properties=properties).get_match_str()
    assert "qualified_address: 'my_cohort::VolumetricImage-SCAN-002"  in match_string
    assert "dim"  not in match_string


def test_get_all_properties():
    properties = {}
    properties['namespace'] = "my_cohort"
    properties['dim'] = 3

    all_props = Node("VolumetricImage", "SCAN-001", properties=properties).get_all_props()
    assert "name" in all_props

def test_patient_no_namespace():
    node = Node("patient", "pid", properties={})
    assert node.properties['qualified_address'] == 'pid'

def test_cohort_wrong_properties():
    with pytest.raises(TypeError):
        Node("cohort", properties={"Description":"a cohort"})


def test_patient_bad_id():
    with pytest.raises(ValueError):
        Node("patient", "my:patient", properties={"namespace":"my_cohort", "Description":"a patient"})

def test_cohort_bad_id():
    with pytest.raises(ValueError):
        Node("cohort", "my:cohort", properties={"Description":"a cohort"})
      
@pytest.mark.parametrize(('node_type'),CONTAINER_TYPES)
def test_container_types(node_type):
    node = Node(node_type, "my_node")
    assert node.properties["type"] == node_type

@pytest.mark.parametrize(('node_type'),ALL_DATA_TYPES)
def test_container_types(node_type):
    node = Node(node_type, "my_node")
    assert node.properties["type"] == node_type
    assert node.properties["qualified_address"] == node_type + '-' + "my_node"
