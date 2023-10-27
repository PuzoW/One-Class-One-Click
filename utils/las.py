import laspy
import numpy as np

def read_las(filename, label=True):
    """convert from a las file with no rgb"""
    #---read the ply file--------
    try:
        inFile = laspy.read(filename)
    except NameError:
        raise ValueError("laspy package not found. uncomment import in /partition/provider and make sure it is installed in your environment")
    N_points = len(inFile)
    x = np.reshape(inFile.x, (N_points,1))
    y = np.reshape(inFile.y, (N_points,1))
    z = np.reshape(inFile.z, (N_points,1))
    xyz = np.hstack((x,y,z))

    r = np.reshape(inFile.red, (N_points, 1))
    g = np.reshape(inFile.green, (N_points, 1))
    b = np.reshape(inFile.blue, (N_points, 1))
    # i = np.reshape(inFile.Reflectance, (N_points, 1))

    rgb = np.hstack((r, g, b))

    if label==True:
        l = np.reshape(inFile.classification, (N_points, 1))
        return xyz, rgb, l
    else:
        return xyz, rgb