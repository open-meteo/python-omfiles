"""
Databricks example: Load Open-Meteo .om files as a Spark DataFrame.

This script demonstrates how to use the ``omfiles`` PySpark custom data source
to read spatial weather data from Open-Meteo's public S3 bucket and persist it
as a reusable Delta table.

Prerequisites (install on the Databricks cluster):
    %pip install omfiles[pyspark,grids]

Requirements:
    - Databricks Runtime 15.2 or later
    - Python 3.10+
"""

# -- 0. Install (run once per cluster session) --------------------------------
# %pip install omfiles[pyspark,grids]

import datetime as dt

# -- 1. Register the custom data source --------------------------------------
from omfiles.pyspark import OmFileDataSource

spark.dataSource.register(OmFileDataSource)  # type: ignore[name-defined]  # noqa: F821

# -- 2. Build the S3 URI for a recent spatial file ----------------------------
MODEL_DOMAIN = "dwd_icon"
date_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)
S3_URI = (
    f"s3://openmeteo/data_spatial/{MODEL_DOMAIN}/{date_time.year}/"
    f"{date_time.month:02}/{date_time.day:02}/0000Z/"
    f"{date_time.strftime('%Y-%m-%d')}T0000.om"
)
print(f"Reading: {S3_URI}")

# -- 3. Read selected weather variables into a Spark DataFrame ----------------
df = (
    spark.read.format("om")  # type: ignore[name-defined]  # noqa: F821
    .option("path", S3_URI)
    .option("variables", "temperature_2m,wind_speed_10m")
    .option("s3_anon", "true")
    .option("s3_block_size", "65536")
    .option("include_coordinates", "true")
    .load()
)

# Show a sample of the data
df.show(20)
df.printSchema()
print(f"Total rows: {df.count()}")

# -- 4. (Optional) Save as a Delta table for fast reuse ----------------------
# df.write.format("delta").mode("overwrite").saveAsTable("weather.dwd_icon_spatial")

# -- 5. Run queries directly -------------------------------------------------
# Example: find the hottest locations
df.createOrReplaceTempView("weather")
hottest = spark.sql(  # type: ignore[name-defined]  # noqa: F821
    """
    SELECT latitude, longitude, temperature_2m
    FROM weather
    WHERE temperature_2m IS NOT NULL
    ORDER BY temperature_2m DESC
    LIMIT 10
    """
)
hottest.show()
