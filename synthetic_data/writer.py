from typing import Union

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet


class Writer:
    def __init__(self, file_path, schema, format):
        """Initialize the writer with the file path and schema.

        Args:
            file_path (str): The path to the file.
            schema (pyarrow.Schema): The schema of the file.
            format (str): The format of the file to write.
        """
        self.file_path = file_path + "." + format
        self.schema = schema
        self.writer = None
        self.format = format
        self.open()

    def open(self):
        """Open the file for writing."""
        if self.format == "csv":
            self.writer = pa_csv.CSVWriter(self.file_path, self.schema)
        elif self.format == "parquet":
            self.writer = pa_parquet.ParquetWriter(self.file_path, self.schema)

    def write(self, batch):
        """Write a batch of data to the file."""
        self.writer.write(batch)

    def close(self):
        """Close the file."""
        if self.writer:
            self.writer.close()


def save_batch(batch, writer):
    """Save a batch of data to the file.

    Args:
        batch (List[dict]): The batch of data to save.
        writer (writer): The writer to use for saving the data
    """
    arrays = []
    schema = writer.schema
    for name, type in zip(schema.names, schema.types):
        arrays.append(pa.array([item[name] for item in batch], type=type))

    batch_table = pa.Table.from_arrays(arrays, schema=writer.schema)
    writer.write(batch_table)
