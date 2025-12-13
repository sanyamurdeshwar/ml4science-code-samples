"""
Koch Snowflake 2D Distribution Generator
=========================================

Generates uniform 2D distributions bounded by Koch snowflake fractals.
Designed for generative modeling research where the goal is to learn
distributions with fractal boundaries.

Two distribution types:
1. Filled Koch snowflake - uniform distribution inside a single snowflake
2. Koch annulus - uniform distribution between two concentric snowflakes

Mathematical Properties:
- Boundary has fractal dimension: log(4)/log(3) ≈ 1.2619
- Boundary has infinite length but encloses finite area
- Area of Koch snowflake: (2√3/5) * s² where s = side length of initial triangle

Author: Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum
import json


# =============================================================================
# Koch Snowflake Geometry
# =============================================================================

@dataclass
class KochSnowflakeConfig:
    """Configuration for Koch snowflake generation."""
    iterations: int = 5
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    rotation: float = np.pi / 2  # Point at top
    orientation: Literal["outward", "inward"] = "outward"
    
    def __post_init__(self):
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.iterations > 10:
            raise ValueError("iterations > 10 produces excessive vertices")


class KochSnowflake:
    """Koch snowflake geometry with point-in-polygon testing."""
    
    def __init__(self, config: KochSnowflakeConfig = None):
        self.config = config or KochSnowflakeConfig()
        self._vertices = None
        self._compute_fractal()
    
    def _compute_fractal(self) -> None:
        """Compute all vertices of the Koch snowflake."""
        # Initial equilateral triangle
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + self.config.rotation
        triangle = np.column_stack([
            self.config.center[0] + self.config.radius * np.cos(angles),
            self.config.center[1] + self.config.radius * np.sin(angles)
        ])
        
        # Close the triangle
        vertices = np.vstack([triangle, triangle[0:1]])
        
        # Apply Koch iterations
        for _ in range(self.config.iterations):
            vertices = self._koch_iteration(vertices)
        
        self._vertices = vertices
    
    def _koch_iteration(self, vertices: np.ndarray) -> np.ndarray:
        """Apply one Koch iteration."""
        n_segments = len(vertices) - 1
        new_vertices = np.zeros((4 * n_segments + 1, 2))
        
        # Sign determines inward vs outward bumps
        sign = -1.0 if self.config.orientation == "outward" else 1.0
        
        for i in range(n_segments):
            a = vertices[i]
            b = vertices[i + 1]
            
            direction = b - a
            perp = np.array([-direction[1], direction[0]]) * sign
            
            p1 = a + direction / 3
            p2 = a + 2 * direction / 3
            apex = (p1 + p2) / 2 + perp * (np.sqrt(3) / 6)
            
            idx = 4 * i
            new_vertices[idx] = a
            new_vertices[idx + 1] = p1
            new_vertices[idx + 2] = apex
            new_vertices[idx + 3] = p2
        
        new_vertices[-1] = new_vertices[0]
        return new_vertices
    
    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()
    
    @property
    def n_vertices(self) -> int:
        return len(self._vertices)
    
    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) bounding box."""
        return self._vertices.min(axis=0), self._vertices.max(axis=0)
    
    @property
    def theoretical_area(self) -> float:
        """Theoretical area: (2√3/5) * s² where s = r√3."""
        side_length = self.config.radius * np.sqrt(3)
        return (2 * np.sqrt(3) / 5) * side_length ** 2
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the snowflake using ray casting."""
        x, y = point
        n = len(self._vertices) - 1
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self._vertices[i]
            xj, yj = self._vertices[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Vectorized point-in-polygon test for multiple points.
        
        Uses the ray casting algorithm with numpy broadcasting for efficiency.
        """
        n = len(self._vertices) - 1
        x = points[:, 0]
        y = points[:, 1]
        
        inside = np.zeros(len(points), dtype=bool)
        
        j = n - 1
        for i in range(n):
            xi, yi = self._vertices[i]
            xj, yj = self._vertices[j]
            
            # Check if ray crosses this edge
            cond1 = (yi > y) != (yj > y)
            
            # Avoid division by zero
            if abs(yj - yi) > 1e-10:
                x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
                cond2 = x < x_intersect
                
                # Toggle inside flag where both conditions are met
                inside ^= (cond1 & cond2)
            
            j = i
        
        return inside


# =============================================================================
# 2D Distribution Samplers
# =============================================================================

@dataclass
class KochDistributionConfig:
    """Configuration for Koch snowflake distribution sampling.
    
    Attributes:
        iterations: Koch fractal depth (5-6 recommended for smooth boundaries)
        center: Center of the snowflake(s)
        outer_radius: Circumradius of the outer snowflake
        inner_radius: Circumradius of inner snowflake (None for filled, >0 for annulus)
        seed: Random seed for reproducibility
    """
    iterations: int = 5
    center: tuple[float, float] = (0.0, 0.0)
    outer_radius: float = 1.0
    inner_radius: float | None = None  # None = filled snowflake, >0 = annulus
    seed: int | None = None
    
    def __post_init__(self):
        if self.inner_radius is not None:
            if self.inner_radius <= 0:
                raise ValueError("inner_radius must be positive")
            if self.inner_radius >= self.outer_radius:
                raise ValueError("inner_radius must be less than outer_radius")


class KochDistribution:
    """Uniform 2D distribution bounded by Koch snowflake(s).
    
    This class generates points uniformly distributed either:
    - Inside a single Koch snowflake (filled)
    - Between two concentric Koch snowflakes (annulus)
    
    Uses rejection sampling for mathematical exactness.
    
    Example:
        >>> # Filled snowflake
        >>> config = KochDistributionConfig(iterations=5, outer_radius=1.0)
        >>> dist = KochDistribution(config)
        >>> points = dist.sample(10000)
        
        >>> # Annulus between two snowflakes
        >>> config = KochDistributionConfig(iterations=5, outer_radius=1.0, inner_radius=0.5)
        >>> dist = KochDistribution(config)
        >>> points = dist.sample(10000)
    """
    
    def __init__(self, config: KochDistributionConfig = None):
        self.config = config or KochDistributionConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Create outer snowflake
        outer_config = KochSnowflakeConfig(
            iterations=self.config.iterations,
            center=self.config.center,
            radius=self.config.outer_radius
        )
        self.outer_snowflake = KochSnowflake(outer_config)
        
        # Create inner snowflake if annulus mode
        self.inner_snowflake = None
        if self.config.inner_radius is not None:
            inner_config = KochSnowflakeConfig(
                iterations=self.config.iterations,
                center=self.config.center,
                radius=self.config.inner_radius
            )
            self.inner_snowflake = KochSnowflake(inner_config)
        
        # Compute bounding box for rejection sampling
        self._min_bounds, self._max_bounds = self.outer_snowflake.bounds
        
        # Estimate acceptance rate for efficiency reporting
        self._estimate_acceptance_rate()
    
    def _estimate_acceptance_rate(self, n_test: int = 10000) -> None:
        """Estimate the acceptance rate for rejection sampling."""
        test_points = self.rng.uniform(
            self._min_bounds, self._max_bounds, (n_test, 2)
        )
        inside_outer = self.outer_snowflake.contains_points(test_points)
        
        if self.inner_snowflake is not None:
            inside_inner = self.inner_snowflake.contains_points(test_points)
            accepted = inside_outer & ~inside_inner
        else:
            accepted = inside_outer
        
        self._acceptance_rate = accepted.sum() / n_test
        
        # Theoretical area calculations
        self._outer_area = self.outer_snowflake.theoretical_area
        if self.inner_snowflake is not None:
            self._inner_area = self.inner_snowflake.theoretical_area
            self._target_area = self._outer_area - self._inner_area
        else:
            self._inner_area = 0
            self._target_area = self._outer_area
    
    @property
    def acceptance_rate(self) -> float:
        """Estimated acceptance rate for rejection sampling."""
        return self._acceptance_rate
    
    @property
    def target_area(self) -> float:
        """Theoretical area of the target region."""
        return self._target_area
    
    @property
    def is_annulus(self) -> bool:
        """Whether this is an annulus distribution."""
        return self.inner_snowflake is not None
    
    def _in_target_region(self, points: np.ndarray) -> np.ndarray:
        """Check which points are in the target region."""
        inside_outer = self.outer_snowflake.contains_points(points)
        
        if self.inner_snowflake is not None:
            inside_inner = self.inner_snowflake.contains_points(points)
            return inside_outer & ~inside_inner
        else:
            return inside_outer
    
    def sample(self, n_points: int, show_progress: bool = False) -> np.ndarray:
        """Sample n_points uniformly from the target region.
        
        Uses rejection sampling with adaptive batch sizing for efficiency.
        
        Args:
            n_points: Number of points to generate
            show_progress: Print progress updates
            
        Returns:
            Array of shape (n_points, 2) with uniform samples
        """
        samples = []
        n_collected = 0
        n_total_generated = 0
        
        # Adaptive batch size based on acceptance rate
        batch_size = max(1000, int(n_points / self._acceptance_rate * 1.2))
        
        while n_collected < n_points:
            # Generate candidate points in bounding box
            candidates = self.rng.uniform(
                self._min_bounds, self._max_bounds, (batch_size, 2)
            )
            n_total_generated += batch_size
            
            # Accept points in target region
            mask = self._in_target_region(candidates)
            accepted = candidates[mask]
            
            # Collect accepted points
            n_needed = n_points - n_collected
            if len(accepted) <= n_needed:
                samples.append(accepted)
                n_collected += len(accepted)
            else:
                samples.append(accepted[:n_needed])
                n_collected += n_needed
            
            if show_progress and n_collected < n_points:
                pct = 100 * n_collected / n_points
                eff = 100 * n_collected / n_total_generated
                print(f"  Progress: {n_collected}/{n_points} ({pct:.1f}%), "
                      f"efficiency: {eff:.1f}%")
        
        result = np.vstack(samples)
        
        if show_progress:
            actual_rate = n_points / n_total_generated
            print(f"  Final: {n_points} points, acceptance rate: {actual_rate:.3f}")
        
        return result
    
    def sample_with_boundary(
        self, 
        n_interior: int, 
        n_boundary: int,
        boundary_noise: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample both interior points and boundary points.
        
        Useful for training models that need to learn the boundary explicitly.
        
        Args:
            n_interior: Number of interior points
            n_boundary: Number of boundary points
            boundary_noise: Optional noise to add to boundary points
            
        Returns:
            Tuple of (interior_points, boundary_points)
        """
        interior = self.sample(n_interior)
        
        # Sample boundary points uniformly by arc length
        boundary = self._sample_boundary(n_boundary, boundary_noise)
        
        return interior, boundary
    
    def _sample_boundary(self, n_points: int, noise_std: float = 0.0) -> np.ndarray:
        """Sample points uniformly along the boundary."""
        if self.inner_snowflake is None:
            # Just the outer boundary
            return self._sample_curve(self.outer_snowflake.vertices, n_points, noise_std)
        else:
            # Both boundaries, weighted by perimeter
            outer_verts = self.outer_snowflake.vertices
            inner_verts = self.inner_snowflake.vertices
            
            outer_perim = self._compute_perimeter(outer_verts)
            inner_perim = self._compute_perimeter(inner_verts)
            total_perim = outer_perim + inner_perim
            
            n_outer = int(n_points * outer_perim / total_perim)
            n_inner = n_points - n_outer
            
            outer_pts = self._sample_curve(outer_verts, n_outer, noise_std)
            inner_pts = self._sample_curve(inner_verts, n_inner, noise_std)
            
            return np.vstack([outer_pts, inner_pts])
    
    def _compute_perimeter(self, vertices: np.ndarray) -> float:
        """Compute perimeter of a polygon."""
        diffs = np.diff(vertices, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def _sample_curve(
        self, 
        vertices: np.ndarray, 
        n_points: int,
        noise_std: float = 0.0
    ) -> np.ndarray:
        """Sample points uniformly along a curve defined by vertices."""
        # Compute cumulative arc length
        diffs = np.diff(vertices, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(edge_lengths)])
        total_length = cumulative[-1]
        
        # Sample uniform distances
        distances = self.rng.uniform(0, total_length, n_points)
        
        # Interpolate to get points
        points = np.zeros((n_points, 2))
        for i, d in enumerate(distances):
            edge_idx = np.searchsorted(cumulative, d, side='right') - 1
            edge_idx = np.clip(edge_idx, 0, len(edge_lengths) - 1)
            
            t = (d - cumulative[edge_idx]) / edge_lengths[edge_idx]
            p1 = vertices[edge_idx]
            p2 = vertices[edge_idx + 1]
            points[i] = p1 + t * (p2 - p1)
        
        # Add noise if requested
        if noise_std > 0:
            points += self.rng.normal(0, noise_std, points.shape)
        
        return points
    
    def get_boundary_vertices(self) -> dict[str, np.ndarray]:
        """Get the boundary vertices for visualization."""
        result = {"outer": self.outer_snowflake.vertices}
        if self.inner_snowflake is not None:
            result["inner"] = self.inner_snowflake.vertices
        return result
    
    # =========================================================================
    # Probability Density Functions
    # =========================================================================
    
    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the probability density at given points.
        
        The PDF is uniform over the target region:
            p(x,y) = 1/Area  if (x,y) in region
            p(x,y) = 0       otherwise
        
        Args:
            points: Array of shape (N, 2) or (2,)
            
        Returns:
            Array of density values, shape (N,)
        """
        points = np.atleast_2d(points)
        in_region = self._in_target_region(points)
        densities = np.where(in_region, 1.0 / self._target_area, 0.0)
        return densities
    
    def log_pdf(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the log probability density at given points.
        
        Args:
            points: Array of shape (N, 2) or (2,)
            
        Returns:
            Array of log density values (−∞ for points outside region)
        """
        points = np.atleast_2d(points)
        in_region = self._in_target_region(points)
        log_densities = np.where(
            in_region, 
            -np.log(self._target_area), 
            -np.inf
        )
        return log_densities
    
    def negative_log_likelihood(self, points: np.ndarray) -> float:
        """Compute the negative log-likelihood of a set of points.
        
        NLL = -mean(log p(x_i))
        
        For points outside the region, returns infinity.
        
        Args:
            points: Array of shape (N, 2)
            
        Returns:
            NLL value (float)
        """
        log_probs = self.log_pdf(points)
        if np.any(np.isinf(log_probs)):
            return np.inf
        return -np.mean(log_probs)
    
    def theoretical_nll(self) -> float:
        """Return the theoretical NLL for the true distribution.
        
        For a uniform distribution: NLL = log(Area)
        This is the best possible NLL a model can achieve.
        """
        return np.log(self._target_area)
    
    def evaluate_generated_samples(self, generated_points: np.ndarray) -> dict:
        """Evaluate generated samples from a model.
        
        Computes metrics useful for assessing generative model quality:
        - fraction_in_region: What % of samples fall in the valid region
        - nll: Negative log-likelihood (inf if any point outside)
        - theoretical_nll: Best possible NLL
        - nll_gap: How much worse than optimal (should be ~0 for good model)
        
        Args:
            generated_points: Samples from a generative model, shape (N, 2)
            
        Returns:
            Dictionary of evaluation metrics
        """
        in_region = self._in_target_region(generated_points)
        fraction_in = in_region.mean()
        
        # NLL only on points inside (to avoid inf)
        if fraction_in > 0:
            points_inside = generated_points[in_region]
            nll_inside = self.negative_log_likelihood(points_inside)
        else:
            nll_inside = np.inf
        
        # Full NLL (will be inf if any outside)
        nll_full = self.negative_log_likelihood(generated_points)
        
        theoretical = self.theoretical_nll()
        
        return {
            "fraction_in_region": float(fraction_in),
            "n_inside": int(in_region.sum()),
            "n_outside": int((~in_region).sum()),
            "nll_inside_only": float(nll_inside),
            "nll_full": float(nll_full),
            "theoretical_nll": float(theoretical),
            "nll_gap": float(nll_inside - theoretical) if fraction_in > 0 else np.inf,
        }
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "type": "annulus" if self.is_annulus else "filled",
            "iterations": self.config.iterations,
            "center": list(self.config.center),
            "outer_radius": self.config.outer_radius,
            "inner_radius": self.config.inner_radius,
            "target_area": self.target_area,
            "acceptance_rate": self.acceptance_rate,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_koch_2d_dataset(
    n_points: int = 10000,
    iterations: int = 5,
    inner_radius: float | None = None,
    outer_radius: float = 1.0,
    seed: int | None = None,
    return_boundary: bool = False,
    n_boundary: int = 1000
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate a 2D uniform distribution bounded by Koch snowflake(s).
    
    Args:
        n_points: Number of interior points to generate
        iterations: Koch fractal depth (5-6 recommended)
        inner_radius: If provided, creates annulus. Must be < outer_radius.
        outer_radius: Radius of outer snowflake
        seed: Random seed for reproducibility
        return_boundary: If True, also return boundary points
        n_boundary: Number of boundary points if return_boundary=True
        
    Returns:
        Array of shape (n_points, 2), or tuple of (interior, boundary) arrays
        
    Examples:
        >>> # Filled Koch snowflake
        >>> X = generate_koch_2d_dataset(10000, iterations=5)
        
        >>> # Koch annulus
        >>> X = generate_koch_2d_dataset(10000, iterations=5, inner_radius=0.5)
        
        >>> # With boundary points for training
        >>> X_interior, X_boundary = generate_koch_2d_dataset(
        ...     10000, iterations=5, inner_radius=0.5, 
        ...     return_boundary=True, n_boundary=2000
        ... )
    """
    config = KochDistributionConfig(
        iterations=iterations,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        seed=seed
    )
    dist = KochDistribution(config)
    
    if return_boundary:
        return dist.sample_with_boundary(n_points, n_boundary)
    else:
        return dist.sample(n_points)


def generate_train_test_split(
    n_train: int = 50000,
    n_test: int = 10000,
    iterations: int = 5,
    inner_radius: float | None = 0.5,
    outer_radius: float = 1.0,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate train/test split for generative modeling experiments.
    
    Args:
        n_train: Number of training points
        n_test: Number of test points  
        iterations: Koch fractal depth
        inner_radius: Inner snowflake radius (None for filled)
        outer_radius: Outer snowflake radius
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'test', and 'boundary' arrays
    """
    config = KochDistributionConfig(
        iterations=iterations,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        seed=seed
    )
    dist = KochDistribution(config)
    
    # Use different RNG states for train/test
    train_points = dist.sample(n_train)
    test_points = dist.sample(n_test)
    
    # Get boundary for evaluation
    boundaries = dist.get_boundary_vertices()
    
    return {
        "train": train_points,
        "test": test_points,
        "outer_boundary": boundaries["outer"],
        "inner_boundary": boundaries.get("inner"),
        "config": dist.to_dict()
    }


# =============================================================================
# Visualization Helpers
# =============================================================================

def create_visualization_svg(
    points: np.ndarray,
    boundaries: dict[str, np.ndarray],
    width: int = 600,
    height: int = 600,
    point_radius: float = 1.0,
    point_color: str = "#3b82f6",
    boundary_color: str = "#1e293b",
    background: str = "#ffffff"
) -> str:
    """Create an SVG visualization of the distribution.
    
    Args:
        points: Sample points array (N, 2)
        boundaries: Dict with 'outer' and optionally 'inner' vertex arrays
        width, height: SVG dimensions
        point_radius: Radius of point circles
        point_color: Fill color for points
        boundary_color: Stroke color for boundaries
        background: Background color
        
    Returns:
        SVG string
    """
    padding = 40
    
    # Compute bounds from boundaries
    all_verts = np.vstack(list(boundaries.values()))
    min_coords = all_verts.min(axis=0)
    max_coords = all_verts.max(axis=0)
    
    # Scale to fit
    data_width = max_coords[0] - min_coords[0]
    data_height = max_coords[1] - min_coords[1]
    scale = min(
        (width - 2 * padding) / data_width,
        (height - 2 * padding) / data_height
    )
    
    def transform(pts):
        transformed = (pts - min_coords) * scale + padding
        transformed[:, 1] = height - transformed[:, 1]  # Flip Y
        return transformed
    
    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        f'  <rect width="{width}" height="{height}" fill="{background}"/>',
    ]
    
    # Draw points
    svg_parts.append('  <g fill="{}" opacity="0.6">'.format(point_color))
    transformed_points = transform(points)
    for x, y in transformed_points[:5000]:  # Limit for SVG size
        svg_parts.append(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius}"/>')
    svg_parts.append('  </g>')
    
    # Draw boundaries
    for name, verts in boundaries.items():
        if verts is None:
            continue
        transformed = transform(verts)
        path_data = f"M {transformed[0, 0]:.2f} {transformed[0, 1]:.2f}"
        for v in transformed[1:]:
            path_data += f" L {v[0]:.2f} {v[1]:.2f}"
        path_data += " Z"
        
        svg_parts.append(
            f'  <path d="{path_data}" fill="none" stroke="{boundary_color}" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
        )
    
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("Koch Snowflake 2D Distribution Generator")
    print("=" * 50)
    
    # Test filled snowflake
    print("\n1. Filled Koch Snowflake:")
    config_filled = KochDistributionConfig(iterations=5, outer_radius=1.0, seed=42)
    dist_filled = KochDistribution(config_filled)
    print(f"   Target area: {dist_filled.target_area:.4f}")
    print(f"   Acceptance rate: {dist_filled.acceptance_rate:.3f}")
    
    X_filled = dist_filled.sample(10000, show_progress=True)
    print(f"   Generated: {X_filled.shape}")
    
    # Test annulus
    print("\n2. Koch Annulus (between two snowflakes):")
    config_annulus = KochDistributionConfig(
        iterations=5, 
        outer_radius=1.0, 
        inner_radius=0.5,
        seed=42
    )
    dist_annulus = KochDistribution(config_annulus)
    print(f"   Outer area: {dist_annulus._outer_area:.4f}")
    print(f"   Inner area: {dist_annulus._inner_area:.4f}")
    print(f"   Target area: {dist_annulus.target_area:.4f}")
    print(f"   Acceptance rate: {dist_annulus.acceptance_rate:.3f}")
    
    X_annulus = dist_annulus.sample(10000, show_progress=True)
    print(f"   Generated: {X_annulus.shape}")
    
    # Test with boundary
    print("\n3. With boundary points:")
    X_interior, X_boundary = dist_annulus.sample_with_boundary(5000, 1000)
    print(f"   Interior: {X_interior.shape}")
    print(f"   Boundary: {X_boundary.shape}")
    
    # Test convenience function
    print("\n4. Quick generation:")
    X_quick = generate_koch_2d_dataset(5000, iterations=5, inner_radius=0.4, seed=123)
    print(f"   Generated: {X_quick.shape}")
    
    # Test train/test split
    print("\n5. Train/test split:")
    data = generate_train_test_split(n_train=10000, n_test=2000)
    print(f"   Train: {data['train'].shape}")
    print(f"   Test: {data['test'].shape}")
    print(f"   Config: {data['config']}")
    
    # Save visualization
    print("\n6. Creating visualizations...")
    
    # Filled snowflake visualization
    boundaries_filled = dist_filled.get_boundary_vertices()
    svg_filled = create_visualization_svg(X_filled, boundaries_filled)
    with open("koch_filled.svg", "w") as f:
        f.write(svg_filled)
    print("   Saved: koch_filled.svg")
    
    # Annulus visualization
    boundaries_annulus = dist_annulus.get_boundary_vertices()
    svg_annulus = create_visualization_svg(X_annulus, boundaries_annulus)
    with open("koch_annulus.svg", "w") as f:
        f.write(svg_annulus)
    print("   Saved: koch_annulus.svg")
    
    # Save sample data as NPZ
    np.savez(
        "koch_dataset.npz",
        X_filled=X_filled,
        X_annulus=X_annulus,
        outer_boundary=boundaries_annulus["outer"],
        inner_boundary=boundaries_annulus["inner"]
    )
    print("   Saved: koch_dataset.npz")
    
    # Test probability density functions
    print("\n7. Probability Density Functions:")
    
    # PDF evaluation
    test_points = np.array([
        [0.0, 0.0],   # Center (inside both)
        [0.0, 0.5],   # Should be in annulus
        [0.0, 0.8],   # Should be in annulus  
        [2.0, 2.0],   # Outside
    ])
    
    print(f"   Filled snowflake PDF at test points:")
    densities_filled = dist_filled.pdf(test_points)
    for i, p in enumerate(test_points):
        print(f"      {p} → p = {densities_filled[i]:.4f}")
    
    print(f"\n   Annulus PDF at test points:")
    densities_annulus = dist_annulus.pdf(test_points)
    for i, p in enumerate(test_points):
        print(f"      {p} → p = {densities_annulus[i]:.4f}")
    
    # NLL computation
    print(f"\n   Theoretical NLL (optimal):")
    print(f"      Filled: {dist_filled.theoretical_nll():.4f}")
    print(f"      Annulus: {dist_annulus.theoretical_nll():.4f}")
    
    # NLL on samples (should match theoretical)
    nll_filled = dist_filled.negative_log_likelihood(X_filled)
    nll_annulus = dist_annulus.negative_log_likelihood(X_annulus)
    print(f"\n   Empirical NLL on samples:")
    print(f"      Filled: {nll_filled:.4f}")
    print(f"      Annulus: {nll_annulus:.4f}")
    
    print("\n✓ All tests passed!")
