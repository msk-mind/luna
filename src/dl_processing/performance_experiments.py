import databricks.koalas as ks  

# Using Sparkdf + Parallel processing
results = Parallel(n_jobs=8)(delayed(process_patient)(row, target_spacing) for row in df.rdd.collect())

# Using Pandas DF and applyInPandas() [Apache Arrow] - best option:
print("** Testing Pandas UDF **")
df.groupBy("feature_uuid").applyInPandas(process_patient_pandas_udf, schema = df.schema).show()

# koalas - apply udf [single processing]
print("  ** Testing: Apply Koalas UDF **")
ks.set_option("compute.default_index_type", "distributed") 
df = df.to_koalas()         
df.groupby('feature_uuid').apply(process_patient_koalas_udf)
df = df.to_spark()

# koalas - batch apply udf - and PARALLEL processing on batches
print("  ** Testing: Koalas apply_batch udf and parallel processing on each batch **")
ks.set_option("compute.default_index_type", "distributed") 
df = df.to_koalas()         
df.koalas.apply_batch(process_patient_koalas_udf_iterate)
df = df.to_spark()