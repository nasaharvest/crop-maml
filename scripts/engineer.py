import sys
from pathlib import Path

sys.path.append("..")

from src.engineer import (
    GeoWikiEngineer,
    PVKenyaEngineer,
    KenyaNonCropEngineer,
    TogoEngineer,
    TogoEvaluationEngineer,
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
        TogoEvaluationEngineer,
    ]:
        engineer = engineer_class(Path("../../landcover-mapping-gabi/data"))
        print(f"Engineering: {engineer.dataset}")
        if isinstance(engineer, TogoEngineer):
            # all evaluation data will be
            # captured by the Togo evaluation set.
            # This is the only dataset where the train / val / test
            # distinction matters
            engineer.engineer(test_set_size=0)


if __name__ == "__main__":
    engineer()
