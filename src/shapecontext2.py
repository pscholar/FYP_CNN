import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import thinplatespline as tps  # Ensure this package is installed

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def cart2logpolar(self):
        rho = math.sqrt(self.x**2 + self.y**2)
        theta = math.atan2(self.x, self.y)
        return math.log(rho + 1e-10), theta  # Avoid log(0) issues

    def dist2(self, other):
        return (self.x - other.x)**2 + (self.y - other.y)**2

class Shape:
    def __init__(self, shape=None, img=None):
        self.img = img
        self.shape = shape if shape is not None else canny_edge_shape(img)
        self.shape_pts = [Point(x, y) for x, y in self.shape]
        self.shape_contexts = self.get_shape_contexts()

    def get_shape_contexts(self, angular_bins=12, radial_bins=None):
        if radial_bins is None:
            max_dist2 = max(p1.dist2(p2) for p1 in self.shape_pts for p2 in self.shape_pts)
            radial_bins = int(math.log(math.sqrt(max_dist2) + 1e-10)) + 1

        shape_contexts = [np.zeros((radial_bins, angular_bins), dtype=float) for _ in self.shape_pts]

        for i, p1 in enumerate(self.shape_pts):
            for j, p2 in enumerate(self.shape_pts):
                if i == j:
                    continue
                r, theta = Point(p2.x - p1.x, p2.y - p1.y).cart2logpolar()
                x = max(0, min(int(r), radial_bins - 1))
                y = int(angular_bins * (theta + math.pi) / (2 * math.pi))
                shape_contexts[i][x][y] += 1

        return [sc.flatten() for sc in shape_contexts]

    def get_cost_matrix(self, Q, beta=0.1, dummy_cost=1):
        def normalize_histogram(hist):
            return hist / (hist.sum() + 1e-10)

        def shape_context_cost(h1, h2):
            h1, h2 = map(normalize_histogram, (h1, h2))
            return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))

        def tangent_angle_dissimilarity(p1, p2):
            return 0.5 * (1 - math.cos(math.atan2(p1.x, p1.y) - math.atan2(p2.x, p2.y)))

        n, m = len(self.shape_pts), len(Q.shape_pts)
        C = np.full((max(n, m), max(n, m)), dummy_cost)

        for i, p in enumerate(self.shape_pts):
            hist_p = self.shape_contexts[i]
            for j, q in enumerate(Q.shape_pts):
                hist_q = Q.shape_contexts[j]
                C[i, j] = (1 - beta) * shape_context_cost(hist_p, hist_q) + beta * tangent_angle_dissimilarity(p, q)

        return C

    def matching(self, Q):
        cost_matrix = self.get_cost_matrix(Q)
        perm = linear_sum_assignment(cost_matrix)[1]
        return np.array(self.shape)[perm], np.array(Q.shape)[perm]

    @staticmethod
    def estimate_transformation(source, target):
        T = tps.TPS()
        BE = T.fit(source, target)
        return BE, T

    def shape_context_distance(self, Q_transformed):
        cost_matrix = self.get_cost_matrix(Q_transformed)
        return cost_matrix.min(axis=1).mean() + cost_matrix.min(axis=0).mean()

    def appearance_cost(self, source, target_transformed, img_q, std=1, window_size=3):
        def gaussian_window():
            X, Y = np.meshgrid(np.arange(-window_size//2 + 1, window_size//2 + 1), repeat=True)
            return np.exp(-(X**2 + Y**2) / (2 * std**2)) / (2 * math.pi * std**2)

        G = gaussian_window()
        AC = 0
        for i, (sx, sy) in enumerate(source):
            tx, ty = target_transformed[i]
            for dx, dy in np.ndindex(G.shape):
                sx_, sy_ = min(sx + dx, self.img.shape[0] - 1), min(sy + dy, self.img.shape[1] - 1)
                tx_, ty_ = min(tx + dx, img_q.shape[0] - 1), min(ty + dy, img_q.shape[1] - 1)
                AC += G[dx, dy] * (self.img[sx_, sy_] - img_q[tx_, ty_]) ** 2
        return AC / len(source)

    def compute_distance(self, Q, w1=1.6, w2=1, w3=0.3, iterations=3):
        for _ in range(iterations):
            source, target = self.matching(Q)
            BE, T = Shape.estimate_transformation(source, target)
            transformed_target = T.transform(target)
            AC = self.appearance_cost(source, transformed_target, Q.img)
            SC = self.shape_context_distance(Q)
        return w1 * AC + w2 * SC + w3 * BE

def canny_edge_shape(img, max_samples=100, t1=100, t2=200):
    edges = cv2.Canny(img, t1, t2)
    x, y = np.where(edges > 0)
    if len(x) > max_samples:
        idx = np.random.choice(len(x), max_samples, replace=False)
        x, y = x[idx], y[idx]
    return np.column_stack((x, y)).tolist()

def distance(source_img, target_img, w1=1.6, w2=1, w3=0.3):
    P = Shape(img=source_img)
    Q = Shape(img=target_img)
    return P.compute_distance(Q, w1, w2, w3)
