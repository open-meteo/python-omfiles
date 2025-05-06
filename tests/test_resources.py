import fsspec
import omfiles

from .test_utils import FileDescriptorCounter


def test_no_file_handle_leaks(temp_om_file):
    file_descriptor_counter = FileDescriptorCounter()
    file_descriptor_counter.count_before()

    # Create and use multiple readers
    for _ in range(10):  # Open and close multiple times
        reader = omfiles.OmFilePyReader(temp_om_file)
        _ = reader[0:5, 0:5]
        reader.close()

    # Also test with fsspec
    fs = fsspec.filesystem("file")
    for _ in range(10):
        with fs.open(temp_om_file, "rb") as f:
            reader = omfiles.OmFilePyReader(f)
            _ = reader[0:5, 0:5]
            reader.close()

    file_descriptor_counter.assert_no_leaks()
