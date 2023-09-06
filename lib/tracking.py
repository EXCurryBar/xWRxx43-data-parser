import numpy as np


class TrackingAlgorithm:
  """
  This class implements a tracking algorithm for tracking the movement of points in a time series of 3D points.
  """

  def track(self, scatter_points, center_points, bounding_boxes):
    """
    This function tracks the movement of points in a time series of 3D points.

    Args:
      scatter_points: A list of 3D points.
      center_points: A list of 3D points that represent the centers of the clusters.
      bounding_boxes: A list of bounding boxes that enclose the clusters.

    Returns:
      A dictionary that maps each point to its cluster ID and trajectory.
    """

    # Initialize the trajectories.
    trajectories = {}
    for point in scatter_points:
      trajectories[point] = Trajectory()

    # Track the movement of the points over time.
    for i in range(len(scatter_points) - 1):
      current_point = scatter_points[i]
      next_point = scatter_points[i + 1]

      # Find the cluster that the current point belongs to.
      cluster_id = self.find_cluster_id(current_point, center_points, bounding_boxes)

      # Update the trajectory of the current point.
      trajectories[current_point].add_point(next_point)
      trajectories[current_point].cluster_id = cluster_id

    return trajectories

  def find_cluster_id(self, point, center_points, bounding_boxes):
    """
    This function finds the cluster ID of a point.

    Args:
      point: A 3D point.
      center_points: A list of 3D points that represent the centers of the clusters.
      bounding_boxes: A list of bounding boxes that enclose the clusters.

    Returns:
      The cluster ID of the point.
    """

    # Find the closest cluster center.
    closest_cluster_center = min(center_points, key=lambda center_point: point.distance(center_point))

    # Find the bounding box of the closest cluster center.
    closest_cluster_bounding_box = bounding_boxes[center_points.index(closest_cluster_center)]

    # If the point is inside the bounding box, return the cluster ID of the closest cluster center.
    if point.inside(closest_cluster_bounding_box):
      return center_points.index(closest_cluster_center)

    # Otherwise, return -1.
    return -1


class Trajectory:
  """
  This class represents a trajectory of a point.
  """

  def __init__(self):
    """
    Constructor.
    """
    self.points = []
    self.cluster_id = -1

  def add_point(self, point):
    """
    This function adds a point to the trajectory.

    Args:
      point: A 3D point.
    """
    self.points.append(point)
