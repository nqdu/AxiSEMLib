import numpy as np
import struct 

class FortranIO:
    def __init__(self, filename,mode='r'):
        """
        Docstring for __init__
    
        :param filename: Description
        :param mode: [r/w/a] read or write mode
        """
        self.filename = filename
        if mode not in ['r','w','a']:
            raise ValueError("FortranIO: mode should be 'r', 'w' or 'a'")
        self.fio = open(self.filename, mode + 'b')

    def read_record(self,type_:np.dtype|str)->np.ndarray:
        """
        Reads a single Fortran unformatted record, including EOF handling.
        """
        dtype = np.dtype(type_)
        
        # 1. Read Start Marker (4-byte integer)
        # We request 1 element. If EOF is hit here, the file is exhausted.
        start_marker_arr = np.fromfile(self.fio, dtype=np.uint32, count=1)
        
        # Check for EOF: np.fromfile returns an empty array if EOF is reached.
        if start_marker_arr.size == 0:
            # Raise EOFError to signal the end of the file.
            raise EOFError("Reached end of file while attempting to read start marker.")
        
        nbytes = start_marker_arr[0]
        
        # 2. Calculate data size and read data
        size = nbytes // dtype.itemsize
        arr = np.fromfile(self.fio, dtype=dtype, count=size)
        
        # Sanity Check: Ensure we read the expected number of data items
        if arr.size != size:
            # This handles the case where the start marker was read, 
            # but EOF was hit inside the data block (a corrupted file).
            raise IOError(f"FortranIO: Corrupted file. Expected {size} items, but only read {arr.size}.")

        # 3. Read End Marker (4-byte integer)
        end_marker_arr = np.fromfile(self.fio, dtype=np.uint32, count=1)

        # Sanity Check: Ensure the trailing marker exists
        if end_marker_arr.size == 0:
             # This means the start marker and data were read, but the end marker is missing.
            raise IOError("FortranIO: Corrupted file. Missing end marker after data block.")
        
        nbytes_end = end_marker_arr[0]

        # 4. Record Size Mismatch Check
        if nbytes != nbytes_end:
            raise IOError(f"FortranIO: Record size mismatch. Expected {nbytes} bytes, found {nbytes_end}.")
        
        return arr
    

    def write_record(self, arr:np.ndarray):
        
        data = arr.tobytes()
        nbytes = len(data)
        self.fio.write(struct.pack('i', nbytes))
        self.fio.write(data)
        self.fio.write(struct.pack('i', nbytes))
    
    def close(self):
        self.fio.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

        return False

    def __enter__(self):
        return self
