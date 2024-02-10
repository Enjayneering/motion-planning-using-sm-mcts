import numpy as np
from common import distance

def coll_count(traj_0, traj_1):
    coll_count = 0
    for i in range(len(traj_0)):
        for j in range(len(traj_1)):
            line_points_0 = np.linspace(traj_0[i][:2], traj_0[i+1][:2], num=10).tolist()
            line_points_1 = np.linspace(traj_1[j][:2], traj_1[j+1][:2], num=10).tolist()
            if any(distance(point_0, point_1) <= 0.5 for point_0 in line_points_0 for point_1 in line_points_1):
                coll_count += 1
    return coll_count