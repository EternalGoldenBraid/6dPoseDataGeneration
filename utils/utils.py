import numpy as np

#from https://github.com/google-research/kubric/blob/main/challenges/multiview_matting/worker.py
def sample_point_in_half_sphere_shell(
    inner_radius: float,
    outer_radius: float,
    rng: np.random.RandomState
    ):
  """Uniformly sample points that are in a given distance
     range from the origin and with z >= 0."""

  add_distractors = False
  while True:
    v = rng.uniform((-outer_radius, -outer_radius, 0),
                    (outer_radius, outer_radius, 0))
    len_v = np.linalg.norm(v)
    correct_angle = True
    if add_distractors:
      cam_dir = v[:2] / np.linalg.norm(v[:2])
      correct_angle = np.all(np.dot(distractor_dir, cam_dir) < np.cos(np.pi / 9.))
    if inner_radius <= len_v <= outer_radius and correct_angle:
      return tuple(v) 
