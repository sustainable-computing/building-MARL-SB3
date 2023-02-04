class BaseDiversity():
    @classmethod
    def is_diverse_training(cls, diversity_weight: float, diverse_policies: list) -> bool:
        assert diversity_weight >= 0.0, "Diversity weight must be non-negative"
        if diversity_weight > 0.0:
            assert len(diverse_policies) > 0, \
                "Diverse policies must be provided for diversity training"

        if diversity_weight > 0.0 and len(diverse_policies) > 0:
            return True
        else:
            return False
