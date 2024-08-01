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
    # Create pa.array from the data
    index_array = pa.array([item["index"] for item in batch], type=pa.int64())
    audio_array = pa.array([item["audio"] for item in batch], type=pa.string())
    tokens_array = pa.array([item["tokens"] for item in batch], type=pa.string())

    # Create batch table
    batch_table = pa.Table.from_arrays(
        [index_array, audio_array, tokens_array], schema=writer.schema
    )

    writer.write(batch_table)
