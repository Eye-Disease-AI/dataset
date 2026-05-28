from enum import Enum


class HardPolicy(Enum):
    PASSTHROUGH = 1
    NO_HARD = 2
    ONLY_HARD = 3
    DOMINATE = 4

    def apply(
        self, packs_with_labels: list[list[dict[str, str]]]
    ) -> list[list[dict[str, str]]]:
        """
        Accepts pack_with_labels in following format (with string labels):
        ```python
        [
            [
                {
                    "path": "...",
                    "label": "..."
                },
                ...
            ],
            ...
        ]
        ```
        """
        if self == HardPolicy.PASSTHROUGH:
            return packs_with_labels
        elif self == HardPolicy.NO_HARD or self == HardPolicy.ONLY_HARD:
            return self.__apply_no_or_only_hard(packs_with_labels)
        elif self == HardPolicy.DOMINATE:
            return self.__apply_dominate(packs_with_labels)
        else:
            raise RuntimeError("Unsupported HardPolicy type")

    def __apply_no_or_only_hard(
        self, packs_with_labels: list[list[dict[str, str]]]
    ) -> list[list[dict[str, str]]]:
        filtered_packs = []

        for pack in packs_with_labels:
            classes_counts = self.__pack_classes_counts(pack)
            non_zero_classes = 0

            for _, count in classes_counts.items():
                if count > 0:
                    non_zero_classes += 1

            if self == HardPolicy.NO_HARD and non_zero_classes == 1:
                filtered_packs.append(pack)
            elif self == HardPolicy.ONLY_HARD and non_zero_classes > 1:
                filtered_packs.append(pack)

        return filtered_packs

    def __apply_dominate(
        self, packs_with_labels: list[list[dict[str, str]]]
    ) -> list[list[dict[str, str]]]:
        mapped_packs = []

        for pack in packs_with_labels:
            classes_counts = self.__pack_classes_counts(pack)
            max_count = 0
            dominating_class = "Zaćma"

            for cls, count in classes_counts.items():
                if count > max_count:
                    dominating_class = cls
                    max_count = count

            mapped_pack = []

            for sample in pack:
                mapped_pack.append({"path": sample["path"], "label": dominating_class})

            cnts = self.__pack_classes_counts(mapped_pack).values()
            assert max(cnts) == sum(cnts)

            mapped_packs.append(mapped_pack)

        return mapped_packs

    def __pack_classes_counts(
        self, pack_with_labels: list[dict[str, str]]
    ) -> dict[str, int]:
        cataracts_count = sum(
            [1 if sample["label"] == "Zaćma" else 0 for sample in pack_with_labels]
        )
        non_cataracts_count = len(pack_with_labels) - cataracts_count

        return {
            "Zaćma": cataracts_count,
            "Brak Zaćmy": non_cataracts_count,
        }
