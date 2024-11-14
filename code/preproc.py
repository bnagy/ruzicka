import logging
import os
import string
import pandas as pd


def _preprocess_text(text: str) -> str:
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def _overlapping_samples(text: str, sample_size, overlap_ratio: int) -> list[list[str]]:
    txt_ary: list[str] = text.split()
    num_samples = len(txt_ary) // (sample_size // overlap_ratio)
    step = sample_size // overlap_ratio
    samples = [txt_ary[i * step : (i * step) + sample_size] for i in range(num_samples)]
    return samples


def preprocess_and_slice_text_files(
    folder_path: str, sample_size: int = 1000, overlap_ratio: int = 1
) -> pd.DataFrame:
    """
    Preprocess a set of text samples. Files are lowercased, split into words,
    and then broken into chunks of the selected size (default 1000 words)

    In:
        folder_path (string): Folder to read for text files. Reads all files
            with a .txt extension.
        sample_size (int=1000): Number of words per chunk. Words at the end of
            the file that do not make a full chunk are discarded.
        overlap_ratio (int=1): Amount of overlap, higher is more overlap. A
            ratio of two advances the rolling window by half of the sample_size.

    Returns:
        dict[str,list]: The results. Each chunk will by indexed by a name
            constructed from the filename with the chunk number, e.g. MyFile.txt
            will produce MyFile_0, MyFile_1 ... The words in the chunk are
            returned as a list.

    """
    processed_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(folder_path, filename), "r") as file:
                    text = file.read()
                    processed_text = _preprocess_text(text)
                    samples = _overlapping_samples(
                        processed_text, sample_size, overlap_ratio
                    )
                    num_samples = len(samples)
                    for i, sample in enumerate(samples):
                        variable_name = filename[:-4] + "_" + str(i)
                        processed_texts[variable_name] = sample
                logging.info(
                    f"'{filename}' was processed and split into {num_samples} samples"
                )
            except Exception as e:
                logging.info(f"{filename} could not be processed, aborting.")
                raise e
    entries = []

    for k, txt in processed_texts.items():
        chunk = " ".join(txt)
        entries.append(
            {
                "tag": k,
                "chunk": chunk,
            }
        )
    chunk_df = pd.DataFrame(entries)

    return chunk_df
