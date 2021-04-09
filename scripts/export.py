import sys
from pathlib import Path
from datetime import date

sys.path.append("..")

from src.exporters import (
    GeoWikiExporter,
    GeoWikiSentinelExporter,
    KenyaPVSentinelExporter,
    KenyaNonCropSentinelExporter,
    RegionalExporter,
    TogoSentinelExporter,
    BrazilSentinelExporter,
    SudanSentinelExporter,
    EthiopiaSentinelExporter,
    BrazilMunicipalitiesExporter,
    MaliSentinelExporter,
)


def export_geowiki():
    """
    Download the raw geowiki labels
    """
    exporter = GeoWikiExporter(Path("../data"))
    exporter.export()


def export_for_labels():
    """
    Download the tif files associated with labels
    from google earth engine
    """
    for export_class in [
        GeoWikiSentinelExporter,
        KenyaPVSentinelExporter,
        KenyaNonCropSentinelExporter,
        TogoSentinelExporter,
        BrazilSentinelExporter,
        SudanSentinelExporter,
        EthiopiaSentinelExporter,
        MaliSentinelExporter,
    ]:
        exporter = export_class(Path("../data"))
        exporter.export_for_labels(num_labelled_points=None, monitor=False, checkpoint=True)


def export_region():
    """
    Export for a specified region
    """
    exporter = RegionalExporter(Path("../data"))
    exporter.export_for_region(
        region_name="Busia",
        end_date=date(2020, 4, 16),
        monitor=False,
        checkpoint=True,
        metres_per_polygon=10000,
    )


def export_brazil_municipality():
    """
    Export for a specific Brazilian municipality
    """
    exporter = BrazilMunicipalitiesExporter(Path("../data"))
    exporter.export_for_region(end_date=date(2020, 4, 16))


if __name__ == "__main__":
    export_geowiki()
    export_for_labels()
    export_region()
    export_brazil_municipality()
