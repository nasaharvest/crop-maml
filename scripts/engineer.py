import sys
from pathlib import Path

sys.path.append("..")

from src.engineer import (
    GeoWikiEngineer,
    PVKenyaEngineer,
    KenyaNonCropEngineer,
    TogoEngineer,
    EthiopiaEngineer,
    SudanEngineer,
    BrazilEngineer,
    MaliEngineer,
)


def engineer():
    for engineer_class in [
        GeoWikiEngineer,
        PVKenyaEngineer,
        KenyaNonCropEngineer,
        TogoEngineer,
        EthiopiaEngineer,
        SudanEngineer,
        BrazilEngineer,
        MaliEngineer,
    ]:
        engineer = engineer_class(Path("../data"))
        engineer.engineer()


if __name__ == "__main__":
    engineer()
