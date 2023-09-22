import os


class BaseDiversity():
    """ Base class for diversity handlers
    """
    @classmethod
    def is_diverse_training(cls, diversity_weight: float, diverse_policies_loc: str) -> bool:
        assert diversity_weight >= 0.0, "Diversity weight must be non-negative"
        if diversity_weight > 0.0:
            assert len(os.listdir(diverse_policies_loc)) > 0, \
                "Diverse policies must be provided for diversity training"

        if diversity_weight > 0.0 and len(os.listdir(diverse_policies_loc)) > 0:
            return True
        else:
            return False
