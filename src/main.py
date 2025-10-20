from data.data_preprocessor import Data_Preprocessor
from config.data_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


if __name__ == "__main__":
    preprocessor = Data_Preprocessor(
        source_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, balance_threshold=1
    )
    preprocessor.run()
