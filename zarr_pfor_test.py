import numpy as np
import zarr
from omfiles.omfiles_numcodecs import PforCodec
from zarr.registry import register_codec

# Register custom codec with zarr!
register_codec("numcodecs.pfor", PforCodec)

# --- Writing ---
file_path = 'data_with_custom_codec.zarr'

print(f"\n--- Writing to {file_path} ---")
# Create a zarr group at the specified path
zarr_group = zarr.open_group(file_path, mode='w')

# Create a dataset within the zarr group with the custom codec
z_array = zarr_group.create_array(
    'my_data',  # Name within the Zarr group
    shape=(1000, 1000),
    chunks=(1000, 1000),
    dtype='int16',  # Match the dtype you specified for the codec
    compressors=[PforCodec(dtype='int8', length=1_000_000 * 2)],  # Registered codec instance
    overwrite=True
)

# Write random data
data_to_write = np.random.randint(0, 10000, size=(1000, 1000), dtype=np.int16)
print(f"Writing data with shape {data_to_write.shape} and dtype {data_to_write.dtype}")
z_array[:] = data_to_write
print(f"Data written. Compressors used: {z_array.compressors}")

# --- Reading ---
print(f"\n--- Reading from {file_path} ---")
# Open the zarr group
zarr_group_read = zarr.open_group(file_path, mode='r')

# Access the specific array
z_array_read = zarr_group_read['my_data']
print(f"Accessed Zarr array. Shape: {z_array_read.shape}, Chunks: {z_array_read.chunks}, Dtype: {z_array_read.dtype}")
print(f"Compressors from metadata: {z_array_read.compressors}")

# Read the data
retrieved_data = z_array_read[:]
print(f"Read data back successfully. Shape: {retrieved_data.shape}")

# Verify the data matches
print(f"Arrays match: {np.array_equal(data_to_write, retrieved_data)}")

print("\n--- Done ---")
