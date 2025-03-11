import rerun as rr
import time
import numpy as np

def read_xyz(filenm, path = 'pointclouds-500'):
    """
    Reading points
        filenm: the file name
    """
    filenm = path + '\\' + filenm

    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points

def visualize(pts, sleeptime=0.01):
    # -- init rerun viewer
    rr.init("Regiongrowing Results", spawn=True)

    # -- log pointcloud one-by-one
    for idx, pointcloud in enumerate(pts):
        subset = pointcloud[:, :3]
        rr.log(
            "segment_{}".format(idx),
            rr.Points3D(
                subset[:],
                colors=[
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                ],
                radii=0.1,
            ),
        )
        rr.log(
            "logs_{}".format(idx),
            rr.TextLog(
                "size segment_{}=={}".format(idx, subset.shape[0]),
                level=rr.TextLogLevel.TRACE,
            ),
        )
        time.sleep(sleeptime)

pointclouds = []
for i in range(500):
    i = str(i)
    if len(i) == 1:
        i = '00' + i + '.xyz'
    if len(i) == 2:
        i = '0' + i + '.xyz'
    if len(i) == 3:
        i = i + '.xyz'
    pointclouds.append(read_xyz(i))

visualize(pointclouds)