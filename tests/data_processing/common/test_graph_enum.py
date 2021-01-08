# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: aukermaa@mskcc.org
'''
import pytest

from data_processing.common.GraphEnum import Node

def test_patient_create():
    create_string = Node("patient", properties={"PatientID":"my_patient", "Namespace":"my_cohort", "Description":"a patient"}).create()
    assert "patient:globals" in create_string    
    assert "QualifiedPath: 'my_cohort::my_patient'"   in create_string
    assert "PatientID"  in create_string
    assert "Description"   in create_string

def test_patient_match():
    match_string  = Node("patient", properties={"PatientID":"my_patient", "Namespace":"my_cohort", "Description":"a patient"}).match()
    assert "patient" in match_string    
    assert "QualifiedPath: 'my_cohort::my_patient'"   in match_string
    assert "Description"  not in match_string


def test_cohort_create():
    create_string = Node("cohort", properties={"CohortID":"my_cohort", "Description":"a cohort"}).create()
    assert "cohort:globals" in create_string    
    assert "QualifiedPath: 'my_cohort::my_cohort'"  in create_string
    assert "CohortID"  in create_string
    assert "Description"  in create_string

def test_cohort_match():
    match_string = Node("cohort", properties={"CohortID":"my_cohort", "Description":"a cohort"}).match()
    assert "cohort" in match_string    
    assert "QualifiedPath: 'my_cohort::my_cohort'"   in match_string
    assert "Description"  not in match_string

def test_cohort_wrong_properties():
    with pytest.raises(RuntimeError):
        Node("cohort", properties={"Description":"a cohort"})

def test_patient_wrong_properties():
    with pytest.raises(RuntimeError):
        Node("patient", properties={"Description":"a patient"})

def test_patient_bad_id():
    with pytest.raises(ValueError):
        Node("patient", properties={"PatientID":"my:patient", "Namespace":"my_cohort", "Description":"a patient"})

def test_cohort_bad_id():
    with pytest.raises(ValueError):
        Node("cohort", properties={"CohortID":"my:cohort", "Description":"a cohort"})
       