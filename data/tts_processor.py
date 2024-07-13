from whisperspeech.pipeline import Pipeline


class TTSProcessor:
    def __init__(self, device: str):
        """Initialize the TTS Processor with a specified device."""
        self.pipe = Pipeline(
            s2a_ref="collabora/whisperspeech:s2a-q4-tiny-en+pl.model", device=device
        )

    def convert_text_to_audio(self, text: str):
        """Convert text to audio.

        Args:
            text (str): The text to convert to audio.

        Returns:
            torch.Tensor: The generated audio.
        """
        return self.pipe.generate(text)

    def convert_text_to_audio_file(self, text: str, output_path: str):
        """Convert text to audio and save it to a file.

        Args:
            text (str): The text to convert to audio.
            output_path (str): The path to save the audio file.
        """
        self.pipe.generate_to_file(output_path, text)


# class TTSProcessorGroup:
#     def __init__(self, num_processes_per_gpu, gpu_ids, output_dir: str = "./audio"):
#         """Group of TTS Processors to generate TTS from prompts."""
#         self.num_processes_per_gpu = num_processes_per_gpu
#         self.gpu_ids = gpu_ids
#         self.completed_idx = []
#         self.failed_idx = []
#         self.output_dir = output_dir

#         # Prepare folders and distribute pipeline
#         self._prepare_folders()
#         print("Setting up", len(self._distribute_pipe()))

#     def _prepare_folders(self):
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)

#     def _distribute_pipe(self):
#         """Distribute the pipeline."""
#         self.pipes = []
#         # Distribute pipeline on multiple gpus
#         for gpu_id in self.gpu_ids:
#             # Distribute num_processes_per_gpu on a single gpu
#             for _ in range(self.num_processes_per_gpu):
#                 self.pipes.append(TTSProcessor(device=f"cuda:{gpu_id}"))

#     def convert_text_to_audio(
#         self,
#         dataset: Dataset,
#     ):
#         # Batch process based on the number of pipes using multiprocessing
#         total_num_pipes = len(self.pipes)
#         with ThreadPoolExecutor(max_workers=total_num_pipes) as executor:
#             for index, data in enumerate(dataset):
#                 pipe = self.pipes[
#                     index % total_num_pipes
#                 ]  # Get the pipe based on index
#                 executor.submit(
#                     self.process_single_prompt, data["prompt"], pipe, data["index"]
#                 )

#     def process_single_prompt(self, text, pipe, index):
#         """Process a single prompt.

#         Args:
#             text (str): The text to convert to audio.
#             pipe (Pipeline): The pipeline to use for conversion.
#             index (int): The index of the prompt.
#         """
#         # Check if audio already exists
#         file_path = os.path.join(self.output_dir, f"audio_{index}.wav")
#         if os.path.exists(file_path):
#             print(f"Skipping generation for {file_path} as it already exists.")
#             return {
#                 "audio_path": file_path
#             }  # Return existing path if audio already generated

#         # Generate audio
#         try:
#             pipe.convert_text_to_audio_file(file_path, text)
#             # Append to completed index if successful
#             self.completed_idx.append(index)
#         except Exception as e:
#             print(f"Error generating audio for text: {text}, error: {e}")
#             # Append to failed index if failed
#             self.failed_idx.append(index)
