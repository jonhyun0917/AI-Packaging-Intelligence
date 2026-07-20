"""
AI Packaging Intelligence System V5.0

Smart Box Optimizer

Author : Better Life For US
"""

from box_db import BOX_DATABASE


# ==========================================================
# Box Optimizer
# ==========================================================

class BoxOptimizer:

    def __init__(self, product_volume):

        self.product_volume = product_volume

    # ------------------------------------------------------

    def calculate_void(self, box):

        void = box.volume - self.product_volume

        if void < 0:
            return None

        return void

    # ------------------------------------------------------

    def utilization(self, box):

        return self.product_volume / box.volume

    # ------------------------------------------------------

    def estimate_pvi(self, box):

        ratio = self.utilization(box)

        score = 0

        # Box Fit (50점)

        if ratio >= 0.90:
            score += 50

        elif ratio >= 0.80:
            score += 45

        elif ratio >= 0.70:
            score += 40

        elif ratio >= 0.60:
            score += 34

        elif ratio >= 0.50:
            score += 28

        elif ratio >= 0.40:
            score += 20

        else:
            score += 10

        # ESG (30점)

        if box.carbon <= 0.20:
            score += 30

        elif box.carbon <= 0.40:
            score += 25

        elif box.carbon <= 0.60:
            score += 20

        else:
            score += 15

        # Cost (20점)

        if box.cost <= 800:
            score += 20

        elif box.cost <= 1200:
            score += 16

        elif box.cost <= 1800:
            score += 12

        else:
            score += 8

        return round(score, 2)

    # ------------------------------------------------------

    def recommend(self):

        candidates = []

        for box in BOX_DATABASE:

            void = self.calculate_void(box)

            if void is None:
                continue

            candidates.append({

                "box": box,

                "void": void,

                "utilization": self.utilization(box),

                "pvi": self.estimate_pvi(box)

            })

        if len(candidates) == 0:

            return None

        candidates.sort(

            key=lambda x: x["pvi"],

            reverse=True

        )

        return candidates[0]
    