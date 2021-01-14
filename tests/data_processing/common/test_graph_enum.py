# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: aukermaa@mskcc.org
'''
import pytest

from data_processing.common.Node import Node

def test_patient_create():
    create_string = Node("patient", "my_patient", properties={"Namespace":"my_cohort", "Description":"a patient"}).get_create_str()
    assert "patient:globals" in create_string    
    assert "QualifiedPath: 'my_cohort::my_patient'"   in create_string
    assert "name"  in create_string
    assert "Description"   in create_string

def test_patient_match():
    match_string  = Node("patient", "my_patient", properties={"Namespace":"my_cohort", "Description":"a patient"}).get_match_str()
    assert "patient" in match_string    
    assert "QualifiedPath: 'my_cohort::my_patient'"   in match_string
    assert "Description"  not in match_string


def test_cohort_create():
    create_string = Node("cohort", "my_cohort", properties={"Description":"a cohort"}).get_create_str()
    assert "cohort:globals" in create_string    
    assert "QualifiedPath: 'my_cohort::my_cohort'"  in create_string
    assert "name"  in create_string
    assert "Description"  in create_string

def test_cohort_match():
    match_string = Node("cohort", "my_cohort", properties={"Description":"a cohort"}).get_match_str()
    assert "cohort" in match_string    
    assert "QualifiedPath: 'my_cohort::my_cohort'"   in match_string
    assert "Description"  not in match_string


def test_metadata_create():
    properties = {}
    properties['Namespace'] = "my_cohort"
    properties['dim'] = 3

    create_string = Node("mhd", "SCAN-001", properties=properties).get_create_str()
    assert "mhd:globals" in create_string    
    assert "QualifiedPath: 'my_cohort::SCAN-001'"  in create_string
    assert "dim"  in create_string

def test_metadata_match():
    properties = {}
    properties['Namespace'] = "my_cohort"
    properties['dim'] = 3

    match_string = Node("mhd", "SCAN-001", properties=properties).get_match_str()
    assert "QualifiedPath: 'my_cohort::SCAN-001'"  in match_string
    assert "dim"  not in match_string


def test_cohort_wrong_properties():
    with pytest.raises(TypeError):
        Node("cohort", properties={"Description":"a cohort"})

def test_patient_wrong_properties():
    with pytest.raises(RuntimeError):
        Node("patient", "pid", properties={})

def test_patient_bad_id():
    with pytest.raises(ValueError):
        Node("patient", "my:patient", properties={"Namespace":"my_cohort", "Description":"a patient"})

def test_cohort_bad_id():
    with pytest.raises(ValueError):
        Node("cohort", "my:cohort", properties={"Description":"a cohort"})
       
