import numpy as np
import open3d as o3d
#set of tools to generate meshes

def lod_export(mesh, lodList, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

def poissonMesh(pcd, d=8, w=0, s=1.1, l=False):
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    return p_mesh_crop

def ballPivotMesh(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    return bpa_mesh


def cleanMesh(mesh):
    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles() #Remove triangles from 3 collinear points
    dec_mesh.remove_duplicated_triangles() #Remove Duplicates
    dec_mesh.remove_duplicated_vertices()  
    dec_mesh.remove_non_manifold_edges()
    return dec_mesh

def applyNormals(pcd):
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    
