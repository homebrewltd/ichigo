import pyarrow as pa
import pyarrow.csv as pa_csv


class CSVWriter:
    def __init__(self, file_path, schema):
        """Initialize the CSV writer with the file path and schema.

        Args:
            file_path (str): The path to the CSV file.
            schema (pyarrow.Schema): The schema of the CSV file.
        """
        self.file_path = file_path
        self.schema = schema
        self.writer = None
        self.open()

    def open(self):
        """Open the CSV file for writing."""
        self.writer = pa_csv.CSVWriter(self.file_path, self.schema)

    def write(self, batch):
        """Write a batch of data to the CSV file."""
        self.writer.write(batch)

    def close(self):
        """Close the CSV file."""
        if self.writer:
            self.writer.close()


def save_batch(batch, csv_writer):
    """Save a batch of data to the CSV file.

    Args:
        batch (List[dict]): The batch of data to save.
        csv_writer (CSVWriter): The CSV writer to use for saving the data
    """
    # Create pa.array from the data
    index_array = pa.array([item["index"] for item in batch], type=pa.int64())
    audio_array = pa.array([item["audio"] for item in batch], type=pa.string())
    tokens_array = pa.array([item["tokens"] for item in batch], type=pa.string())

    # Create batch table
    batch_table = pa.Table.from_arrays(
        [index_array, audio_array, tokens_array], schema=csv_writer.schema
    )

    csv_writer.write(batch_table)
