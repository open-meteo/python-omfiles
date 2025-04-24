import asyncio

from fsspec.spec import AbstractBufferedFile, AbstractFileSystem


class AsyncFsSpecWrapper:
    """
    Wraps a synchronous fsspec file-like object to provide
    an asynchronous interface using asyncio's run_in_executor.
    """
    def __init__(self, sync_file: AbstractBufferedFile):
        self._sync_file = sync_file
        # Try to get common attributes directly if they exist (like size)
        # These are assumed non-blocking
        self.size = getattr(sync_file, 'size', None)
        self.path = getattr(sync_file, 'path', None)
        self.fs: AbstractFileSystem = getattr(sync_file, 'fs')
        self.mode = getattr(sync_file, 'mode', 'rb') # Assume binary read default

    def _run_in_executor(self, func, *args):
        # Helper to run synchronous methods in the default executor
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, func, *args)

    async def read_bytes(self, offset, count):
        """Atomic seek and read operation."""
        # Unfortunately fsspec does not currently provide any atomic operation on the
        # file to read random blocks of data from a file.
        # Therefore, we are using the underlying FileSystem-Object to perform the operation.
        data = await self._run_in_executor(self.fs.read_bytes, self.path, offset, offset+count)
        return data

    async def close(self):
        """Async version of close."""
        # Check if close exists and is callable
        if hasattr(self._sync_file, 'close') and callable(self._sync_file.close):
             await self._run_in_executor(self._sync_file.close)
