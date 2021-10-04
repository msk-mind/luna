from luna.pathology.spatial.transforms import generate_k_function_statistics
import os

method_data = {
        "phenotype1" : {
                "name" : "phenotype_short",
                'value' : 'CD68+'
        },
        "phenotype2" : {
                "name" : "phenotype_short",
                'value' : 'panCK+'
        },
        "count" : True,
        "radius" : 60,
        "intensity" : 'Cell: PDL1 sum',
        "distance" : True
}

def test():

    output = generate_k_function_statistics(
        f"pyluna-pathology/tests/luna/testdata/data/spatial/example_cell_data.csv", 
        method_data, 
        "PX-001")
    assert output.index[0] == "PX-001"

