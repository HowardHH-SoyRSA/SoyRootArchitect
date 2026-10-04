"""Closed transverse contours from native triangle intersections."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import resample_polyline
from .primary import _plane_basis
from .surface_patches import _segment_radius_profile


def recovered_wall_sections(points, labels, path, owner, context, spacing, excluded):
    """Measure a recovered shaft without treating a one-sided label as a tube.

    Each accepted contour closes through original triangle edges, contains
    existing owner support, and stays local to the prior shaft. Missing or
    competing contours are withheld. Only native vertices of those contours
    can become wall evidence; offshoot components are never flood-filled.
    """
    faces = context.triangles
    wall = np.zeros(len(points), bool)
    line = resample_polyline(path, 2 * spacing)
    centers = line.copy()
    accepted = np.zeros(len(line), bool)
    radii = np.full(len(points), spacing)
    if faces is None or not len(faces) or np.count_nonzero(labels == owner) < 20:
        return wall, radii, centers, accepted
    owned = points[labels == owner]
    ps, pr = _segment_radius_profile(owned, path, spacing)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0),axis=1))]
    radius = np.interp(arc, ps, pr)
    width = np.maximum(3*spacing, 2*radius)
    tangents = np.column_stack([np.interp(np.minimum(arc[-1],arc+width),arc,line[:,i])
        -np.interp(np.maximum(0,arc-width),arc,line[:,i]) for i in range(3)])
    tangents /= np.maximum(np.linalg.norm(tangents,axis=1)[:,None],1e-12)
    incident = coo_matrix((np.ones(faces.size), (faces.ravel(), np.repeat(np.arange(len(faces)),3))),
                         shape=(len(points),len(faces))).tocsr()
    unsafe = np.zeros(len(points), bool)
    unsafe[np.unique(context.edges[context.edge_incidence != 2])] = True
    eligible = ((labels == -1) | (labels == owner)) & ~excluded & ~unsafe
    for i, (station, tangent) in enumerate(zip(line,tangents)):
        station = station + tangent * spacing * 1e-6
        reach = max(4*radius[i], 8*spacing)
        nearby = context.point_tree.query_ball_point(station, reach)
        local_faces = np.unique(incident[nearby].indices)
        tri = faces[local_faces]
        tri = tri[eligible[tri].all(axis=1)]
        heights = (points[tri]-station) @ tangent
        crosses = (heights.min(axis=1)<0) & (heights.max(axis=1)>0)
        tri, heights = tri[crosses], heights[crosses]
        if len(tri)<8:
            continue
        edge = tri[:,[[0,1],[1,2],[2,0]]]
        edge_height = heights[:,[[0,1],[1,2],[2,0]]]
        cut = edge_height[:,:,0]*edge_height[:,:,1]<0
        if np.any(cut.sum(axis=1)!=2):
            continue
        intersections = np.sort(edge[cut],axis=1)
        native_edges, node = np.unique(intersections,axis=0,return_inverse=True)
        segments = node.reshape(-1,2)
        graph = coo_matrix((np.ones(2*len(segments)), (np.r_[segments[:,0],segments[:,1]],
                          np.r_[segments[:,1],segments[:,0]])),shape=(len(native_edges),len(native_edges))).tocsr()
        count, part = connected_components(graph,directed=False)
        a,b=points[native_edges[:,0]],points[native_edges[:,1]]
        ha,hb=(a-station)@tangent,(b-station)@tangent
        xyz=a+(ha/(ha-hb))[:,None]*(b-a)
        basis=_plane_basis(tangent)
        uv=(xyz-station)@basis.T
        candidates=[]
        for component in range(count):
            nodes=np.flatnonzero(part==component)
            if (len(nodes)<8 or np.any(np.diff(graph.indptr)[nodes]!=2)
                    or np.any(graph[nodes].data != 1)):
                continue
            vertices=np.unique(native_edges[nodes])
            if np.count_nonzero(labels[vertices]==owner)<3:
                continue
            sequence=[int(nodes[0])];previous=-1;current=sequence[0]
            while True:
                neighbors=graph.indices[graph.indptr[current]:graph.indptr[current+1]]
                following=int(neighbors[0] if neighbors[0]!=previous else neighbors[1])
                if following==sequence[0]:
                    break
                if following in sequence:
                    sequence=[];break
                sequence.append(following);previous,current=current,following
            if len(sequence)!=len(nodes):
                continue
            polygon=uv[sequence];next_polygon=np.roll(polygon,-1,axis=0)
            cross=polygon[:,0]*next_polygon[:,1]-next_polygon[:,0]*polygon[:,1]
            area=.5*cross.sum()
            if abs(area)<spacing**2:
                continue
            centroid=((polygon+next_polygon)*cross[:,None]).sum(axis=0)/(6*area)
            radial=np.linalg.norm(polygon-centroid,axis=1)
            perimeter=np.linalg.norm(next_polygon-polygon,axis=1).sum()
            # A long merged side branch is not a transverse shaft section.
            if (perimeter**2/(4*np.pi*abs(area))>2.5 or radial.max()>3*np.median(radial)
                    or np.linalg.norm(centroid)>max(3*radius[i],4*spacing)
                    or np.max(np.linalg.norm(polygon,axis=1))>reach):
                continue
            candidates.append((vertices,station+centroid@basis,float(np.median(radial))))
        if len(candidates)!=1:
            continue
        vertices,center,measured=candidates[0]
        vertices=vertices[np.abs((points[vertices]-station)@tangent)<=2*spacing]
        wall[vertices]=True
        radii[vertices]=measured
        centers[i]=center
        accepted[i]=True
    return wall,radii,centers,accepted
