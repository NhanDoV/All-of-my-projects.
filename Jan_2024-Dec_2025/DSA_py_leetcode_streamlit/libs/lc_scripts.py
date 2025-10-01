from math import sqrt
from itertools import combinations

class Solution:
    def compute_area_based_on_3_pts(self, a : float, b: float, c: float) -> float:
        p = (a + b + c) / 2
        area = p * (p - a) * (p - b) * (p - c)
        return area

    def distance(self, p1: list, p2: list) -> int:
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def largestTriangleArea(self, points: list[list[int]]) -> float:
        max_area = 0.0
        for p1, p2, p3 in combinations(points, 3):
            a, b, c = self.distance(p1, p2), self.distance(p2, p3), self.distance(p3, p1)
            area = self.compute_area_based_on_3_pts(a, b, c)
            max_area = max(max_area, area)
        
        return max_area**0.5      