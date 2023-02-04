from buildingenvs.building import Building
from buildingenvs.dooebuilding import DOOEBuilding
from buildingenvs.typea import TypeABuilding
from buildingenvs.fivezone import FiveZoneBuilding

from enum import Enum


class BuildingEnvStrings(Enum):
    five_zone = "five_zone"
    denver = "denver"
    sf = "sf"
    dooe = "dooe"
