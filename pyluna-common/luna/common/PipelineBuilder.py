import yaml, json
import luna.transforms

def load(stream):
    if isinstance(stream, str):
        stream = open(stream, 'r')
    pipeline_config = yaml.safe_load( stream )
    pipeline = []
    for stage, job_config in enumerate(pipeline_config['stages']):
        print (f"STAGE {stage}: {job_config['job']}")
        print (json.dumps(job_config, indent=4))
        method_to_call = getattr(luna.transforms, job_config['job'])
        pipeline.append ( (method_to_call, job_config) )

    return pipeline

