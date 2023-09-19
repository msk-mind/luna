import fire
import fsspec
from fsspec import open
import pandas as pd
import re

from luna.common.utils import get_config
from luna.common.models import ShapeFeaturesSchema


def create_query(variables):
    query = '''
select * from (
    select slide_id, Parent || ' ' ||  Class || ' ' || variable as variable, shape_features."value" as val
    from shape_features
)
pivot (
    max(val) for variable in ({})
)
'''.format(", ".join(["'" + var + "' as " + re.sub('[^_a-zA-Z0-9 \n]', '', re.sub(r'(: |:| )', '_', var).replace('µ','u')) for var in variables]))
    return query


def create_wide_shape_features_query(
    shape_features_urlpath: str,
    storage_options: dict = {},
):
    """Gets wide shape features query for dremio

    Args:
        shape_features_urlpaths (List[str]): URL/path to shape feature parquet files
        storage_options (dict): storage options to pass to reading functions
    """
    with open(shape_features_urlpath, **storage_options) as of:
        df = pd.read_parquet(of)
    ShapeFeaturesSchema.validate(df)
    df['merged_variable'] = df.Parent + " " + df.Class + " " + df.variable
    return create_query(df['merged_variable'].unique())

def cli(
    shape_features_urlpath: str,
    storage_options: dict = {}
):
    """Prints wide shape features query for Dremio

    Args:
        shape_features_urlpaths (List[str]): URL/path to shape features parquet files
        storage_options (dict): storage options to pass to reading functions
    """
    config = get_config(vars())
    query = create_wide_shape_features_query(
        config['shape_features_urlpath'],
        config['storage_options']
    )

    print(query)

def fire_cli():
    fire.Fire(cli)

if __name__ == "__main__":
    fire_cli()

