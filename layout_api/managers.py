"""
Se ofrece un context manager para manejar objetos BytesIO.
"""
from contextlib import contextmanager, asynccontextmanager
from io import BytesIO


@contextmanager
def bytes_io_manager(data: bytes = None):
    """
    Context manager para manejar objetos BytesIO.
    
    :param data: Datos iniciales opcionales para BytesIO.
    """
    bytes_io = BytesIO(data)
    try:
        yield bytes_io
    finally:
        bytes_io.close()

@asynccontextmanager
async def async_bytes_io_manager(data: bytes = None):
    """
    Context manager asíncrono para manejar objetos BytesIO.
    
    :param data: Datos iniciales opcionales para BytesIO.
    """
    bytes_io = BytesIO(data)
    try:
        yield bytes_io
    finally:
        bytes_io.close()