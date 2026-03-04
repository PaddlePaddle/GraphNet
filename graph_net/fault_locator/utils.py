from contextlib import contextmanager
import os
import tempfile


@contextmanager
def get_tmp_file_path():
    """
    Context manager that yields a temporary file path and
    ensures the file is deleted after the context exits.
    """
    # Create a named temporary file.
    # delete=False is used because some OS/File-systems might lock
    # the file if we try to open it by path while it's still open here.
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name

    try:
        # Close the file handle so other processes/functions can open it by path
        tmp.close()
        yield tmp_path
    finally:
        # Cleanup: Ensure the file is removed from the disk
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
