import sys
from pathlib import Path

sys.path.append("..")

from src.processors import (
    GeoWikiProcessor,
    KenyaPVProcessor,
    KenyaNonCropProcessor,
    TogoProcessor,
    SudanProcessor,
    EthiopiaProcessor,
    LEMProcessor,
    MaliProcessor,
)


def process_files():
    for processor_class in [
        GeoWikiProcessor,
        KenyaPVProcessor,
        KenyaNonCropProcessor,
        TogoProcessor,
        SudanProcessor,
        EthiopiaProcessor,
        LEMProcessor,
        MaliProcessor,
    ]:
        processor = processor_class(Path("../data"))
        processor.process()


if __name__ == "__main__":
    process_files()
