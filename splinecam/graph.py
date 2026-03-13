import torch
import numpy as np

import networkx as nx
import igraph as ig
import graph_tool as gt
from graph_tool import topology

from splinecam.utils import verify_collinear, get_region_means, get_Abw, DEFAULT_DEVICE

import tqdm

@torch.jit.script
def make_line_2D(vert1,vert2):
    '''
    Make 2D line equations (Ax + By + C = 0) from pairs of vertices.
    
    Args:
        vert1 (torch.Tensor): First set of vertices, shape (batchsize, 2).
        vert2 (torch.Tensor): Second set of vertices, shape (batchsize, 2).
        
    Returns:
        torch.Tensor: Line parameters [A, B, C] of shape (batchsize, 3).
    '''
    
    x1x2 = vert1[:,0]-vert2[:,0]
    y1y2 = vert1[:,1]-vert2[:,1]
    
    b = vert1[:,1]*x1x2-vert1[:,0]*y1y2
    # Return [A, B, C] where A = y1-y2, B = -(x1-x2), C = b
    return torch.hstack((y1y2[...,None],-x1x2[...,None],b[...,None]))

@torch.jit.script
def find_intersection_2D(line1, line2, eps : float = 1e-7, verify : bool = False):
    '''
    Find intersection points of two batches of 2D lines.
    
    Args:
        line1 (torch.Tensor): First set of lines [A, B, C], shape (batchsize, 3).
        line2 (torch.Tensor): Second set of lines [A, B, C], shape (batchsize, 3).
        eps (float): Tolerance for verification.
        verify (bool): If True, verifies the intersection points satisfy the line equations.
        
    Returns:
        tuple: (intersection points (batchsize, 2), verification flag (bool or torch.Tensor))
    '''
    
    # Concatenate the line parameter matrices horizontally to form A and b from Ax + b = 0
    Ab = torch.cat(
        (torch.reshape(line1,shape=(line1.shape[0],1,line1.shape[1])),
         torch.reshape(line2,shape=(line2.shape[0],1,line2.shape[1]))),
        dim=1
    )
    
    # Solve system of linear equations Ax + b = 0 -> Ax = -b
    v = torch.linalg.solve(Ab[...,:-1], -Ab[...,-1])
    
    flag = False
    if verify:
        # Verify if the computed point v satisfies Ax + b = 0 (i.e., Ax = -b)
        flag = torch.allclose(
            torch.bmm(Ab[...,:-1],v[...,None]),
            -Ab[...,-1][...,None],
            atol=eps,
            rtol=0.,
            equal_nan=False
        )
    
    return v,flag

@torch.jit.script
def get_intersection_pattern(poly,hyps):
    '''
    Get the binary intersection pattern (sign of pre-activations) of a polygon with hyperplanes.
    
    Args:
        poly (torch.Tensor): Polygon vertices, shape (num_vertices, N).
        hyps (torch.Tensor): Parameters of (N-1)-D hyperplanes equations, shape (num_hyps, N+1).
        
    Returns:
        torch.Tensor: Binary pattern tensor, shape (num_vertices, num_hyps).
    '''
    # Compute the pre-activations for each vertex against each hyperplane
    pre_act = (hyps[...,:-1] @ poly.T + hyps[...,-1:]).T
    q = (pre_act>0)*1
    return q


@torch.jit.script
def edge_hyp_intersections(qT,poly,hyps):
    '''
    Find intersections between the polygon edges and the hyperplanes based on pattern changes.
    
    Intersection cases for q \in {1,0}
    1. intersects two edges of polytope: two change of symbols for two different set of edges; two changes in q in a row
    4. intersects one vertex: only one zero and no change of symbol on two sides of zero; two consecutive changes in q
    
    Args:
        qT (torch.Tensor): Transposed pattern tensor, shape (num_hyps, num_vertices).
        poly (torch.Tensor): Polygon vertices.
        hyps (torch.Tensor): Hyperplanes.
        
    Returns:
        torch.Tensor: Indices of intersecting hyperplanes and the corresponding adjacent vertices
                      (hyp_idx, vert_idx_1, vert_idx_2).
    '''
    
    # find where intersection patterns change; detects edge crossing
    ## add vertex intersection check (cases 2-4)
    mask = qT[...,:-1] != qT[...,1:] ## happens outside as well
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    
    # index for hyp and adjacent point (edge) pairs
    hyp_v1_v2_idx = torch.hstack([hyp_vert_idx,hyp_vert_idx[:,-1:]+1])
    
    return hyp_v1_v2_idx

@torch.jit.script
def vertex_order_along_line_batched(endpoint1,endpoint2,v):
    '''
    batched implementation of vertex_order_along_line
    '''
    # Concatenate start point, end point, and intermediate vertices
    v = torch.cat([
        endpoint1,endpoint2,v
    ],dim=1)
    
    dim_to_sort = torch.argmax(v.std(0))
    
    idx = v[:,:,dim_to_sort].argsort()
    
    endpoint_match = idx[:,0] == 0
    
    if torch.all(endpoint_match):
        pass
    else:
        flip_idx = torch.where(
            torch.logical_not(endpoint_match)
        )[0]
        
        idx[flip_idx,:] = torch.flip(idx[flip_idx,:],dims=(1,))
        
    return idx[:,1:-1]-2

@torch.jit.script
def vertex_order_along_line(endpoint1,endpoint2,v):
    '''
    Returns the ordered index of the vertex sequence `v` from `endpoint1` to `endpoint2`.
    
    Args:
        endpoint1 (torch.Tensor): Start point of the line, shape (1, 2).
        endpoint2 (torch.Tensor): End point of the line, shape (1, 2).
        v (torch.Tensor): Intermediate vertices to order, shape (N, 2).
        
    Returns:
        torch.Tensor: Ordered indices of the vertices.
    '''
    # Concatenate start point, end point, and intermediate vertices
    v = torch.cat([
        endpoint1,endpoint2,v
    ],dim=0)
    
    # Sort along the axis with greatest variance to handle axis aligned lines
    dim_to_sort = torch.argmax(v.std(0)) 
    
    # Sort vertices
    idx = v[:,dim_to_sort].argsort()
    
    # Verify if sorting matches forward or reverse sequence
    endpoint_match1 = idx[0] == 0
    endpoint_match2 = idx[-1] == 1
    endpoint_match_rev1 = idx[0] == 1
    endpoint_match_rev2 = idx[-1] == 0
    
    if (endpoint_match1 and endpoint_match2):
        pass # Already in correctly matching order
    
    elif endpoint_match_rev1 and endpoint_match_rev2:
        # Flip back to forward sequence if it was reversed
        idx = torch.flip(idx,dims=(0,))
        
    else:
        print('sorting dimension ',dim_to_sort)
        print('sorted vertices', v[idx])
        print('sorted idx ',idx)
        raise ValueError('sorting issue')
        
    # Return mapping adjusted for the removal of start/end indices
    return idx[1:-1]-2


# @torch.jit.script
def order_vertices_poly(v,hyp_v1_v2_idx,poly,node_names):
    '''
    Orders intersection vertices that lie along the edges of a polygon.
    '''
    
    # Avoid mutating original tensors, was added due to some unexpected behavior
    hyp_v1_v2_idx = hyp_v1_v2_idx.clone()
    v = v.clone()
    poly = poly.clone()
    node_names = node_names.clone()
    
    v_new = []
    hyp_v1_v2_idx_new = []
    node_names_new = []
    
    # Process each edge of the polygon independently
    for ii in torch.unique(hyp_v1_v2_idx[:,1]):
        
        # Filter vertices located on the current edge (sharing the same start node)
        mask = hyp_v1_v2_idx[:,1] == ii
        adj = hyp_v1_v2_idx[mask].clone()
        verts = v[mask].clone()
        nodes = node_names[mask].clone()
        
        # order along line, ordered from poly[ii] to poly[ii+1] 
        idx = vertex_order_along_line(poly[ii][None,...],
                                      poly[ii+1][None,...],
                                      verts)
        
        # Append ordered elements
        v_new.append(verts[idx])
        hyp_v1_v2_idx_new.append(adj[idx])
        node_names_new.append(nodes[idx])
        
    return v_new,hyp_v1_v2_idx_new,node_names_new


def add_line_to_graph(G,
                      node_names,start,end,
                      v,
                      line_name='',
                      layer_name=-1
                     ):
    '''
    Adds a new line consisting of multiple intersection nodes to the graph.
    
    Args:
        G (nx.Graph): Target graph to add new edges to.
        node_names (Tensor/List): Identifiers for the new sequence of nodes.
        start (int/Tensor): Starting node (already currently in the graph).
        end (int/Tensor): Ending node (already currently in the graph).
        v (Tensor): Vertices (coordinates) to assign to each newly added node.
        line_name (str/int): Label or index marking the source hyperplane.
        layer_name (int): Indicator for the neural network layer.
    '''
    
    # Cast tensors to basic types for standard graph libraries if needed
    try:
        node_names = node_names.numpy()
        start = np.int64(start.numpy().squeeze())
        end = np.int64(end.numpy().squeeze())
        line_name = np.int64(line_name.numpy().squeeze())
    except:
        pass
    
    v = v.cpu()
    
    # Insert new nodes into the graph containing their respective vertex coordinates
    [ G.add_node(o,v=vt) for o,vt in zip(node_names, v) ]
    
    # Connect the edges
    G.add_edge(start,node_names[0],layer=layer_name,
               hyp=line_name
              )
        
    [
        G.add_edge(src,dst,
                   layer=layer_name,hyp=line_name,
                  ) for src,dst in zip(node_names[:-1],node_names[1:])
    ]
        
    G.add_edge(node_names[-1],end,
               layer=layer_name,hyp=line_name)
    
    return

def set_bidirectional(G):
    '''
    Utility to make an existing graph-tool graph essentially bidirectional 
    by duplicating each edge with its reverse component.
    
    Args:
        G (graph_tool.Graph): The target graph object to render bidirectional.
    '''
    
    G.set_directed(True)

    # For each existing edge, append a new edge mapping the reverse direction
    for e in G.get_edges():
        G.add_edge(e[1],e[0])
        
def _find_cycles(V,start_edge):
    '''
    Given a bidirectional graph-tool graph and a starting boundary edge, find cycles originating from that edge.
    
    Args:
        V (graph_tool.Graph): A bidirectional graph of hyperplanes.
        start_edge (graph_tool.Edge): The boundary edge from which to begin the cycle search.
        
    Returns:
        list: A list of cycles, where each cycle is a list of vertex IDs.
    '''
    
    edge_list_remove = [[v for v in start_edge]]
    
    ## if edge is a boundary edge
    if not V.ep['layer'][start_edge] == -1:
        raise ValueError('start_edge must be a boundary edge')
    
    V.remove_edge(start_edge)
    
    out_cycles = []  
    
    for each_edge in edge_list_remove:
        
        remove_q = []
        
        vertices = []
        vertex_id = []
        for v in each_edge:
            vertices.append(v)
            vertex_id.append(V.vertex_index[v])
        
        ## if no way in and no way out
        
        if not (V.get_in_degrees(
            [vertex_id[1]]
        )[0]>1) and (V.get_out_degrees(
            [vertex_id[0]]
        )[0]>1):
            
            continue
                                                  
        if V.edge(*vertices) is None and V.edge(vertices[1],vertices[0]) is None:
            continue
      
        remove_q.append(V.edge(vertices[1],vertices[0])) # remove opposite path as well
        vs,es = topology.shortest_path(V,
                 source=vertices[0],
                 target=vertices[1],
#                  weights=gT.ep['len'] ## for dijkstra; bfs faster
                )
                            
        out_cycles.append([V.vertex_index[each] for each in vs])
   
        for e in es:
            v = [v for v in e]
            if V.ep['layer'][e] == -1:
                remove_q.append(e)
                remove_q.append(V.edge(v[1],v[0]))
            else:
                remove_q.append(e)
                edge_list_remove.append(v)
        
        for each in remove_q:
            try:
                V.remove_edge(each)
            except:
                pass
                
#         [V.remove_edge(each) for each in remove_q] #only remove new edges
        
    return out_cycles
        
        
def find_cycles_in_graph(G,return_coordinates=False):
    '''
    Given a graph-tool graph representing a polytope partition, find all simple cycles present within it.
    
    Args:
        G (graph_tool.Graph): The input partition graph.
        return_coordinates (bool): If True, returns a sequence of vertex coordinates instead of node IDs.
        
    Returns:
        list: A list of cycles containing either node indices or coordinate tensors.
    '''
    
    # deep copy
    V = gt.Graph(G)

    set_bidirectional(V)
    
    # tradeoff time complexity for space complexity
    V.set_fast_edge_removal()
    
    ## find boundary edge to start from
    for e in V.edges():
        if V.ep['layer'][e] == -1:
            break
    
    cycles = _find_cycles(V,e)
    
    cycles = [each for each in cycles if len(each)>1]
    
    if return_coordinates:
        cycles = cycle_nodes2vertices(V,cycles)
    
    return cycles

def cycle_nodes2vertices(V,cycles,dcast=np.asarray):
    '''
    Convert cycles represented by node sequences into their corresponding vertex coordinates.
    
    Args:
        V (graph_tool.Graph): The graph possessing the 'v' vertex attribute containing coordinates.
        cycles (list of lists): The list of cycles, where each inner list contains node indices.
        dcast (callable): Typecasting function mapped over each coordinate.
        
    Returns:
        list: cycles where node sequence indices are replaced by actual spatial vertex configurations.
    '''
    
    cycles = [[dcast(
            V.vp['v'][V.vertex(v)]
        )  for v in each_cycle] for each_cycle in cycles]
    
    return cycles

def create_poly_hyp_graph(poly, hyps, q=None, hyp_endpoints=None, dtype=torch.float64, verify=True):
    '''
    Constructs an undirected intersection graph from a polygon's boundary and hyperplanes.
    
    Args:
        poly (torch.Tensor): Ordered vertices forming the polygon, shape (V, 2).
        hyps (torch.Tensor): Parameters for the slicing hyperplanes, shape (H, 3).
        q (torch.Tensor, optional): Precomputed intersection pattern matrix for speedup.
        hyp_endpoints (torch.Tensor, optional): Precomputed geometric endpoints for each hyperplane.
        dtype: Data type for geometry representation.
        verify (bool): Flag toggling geometric consistency checks.
        
    Returns:
        nx.Graph: A NetworkX graph
    '''

    G = nx.Graph()
    
#     hyps = layer.get_weights().type(dtype)
#     poly = poly.type(dtype)
    
    # index for redundant vertex because last vertex is same as first vertex
    redundant_vert_id = len(poly)-1

    poly_node_idx = np.asarray(list(range(len(poly)-1))+[0])
    # poly_node_idx = torch.from_numpy(poly_node_idx).type(torch.int)
    
    poly_hyp_idx = np.asarray(range(len(poly)-1))
    # poly_hyp_idx = torch.from_numpy(poly_hyp_idx).type(torch.int)

    # add nodes as vertices to graph
    [G.add_node(o,v=v.cpu()) for o,v in zip(poly_node_idx[:-1], poly[:-1])]

#     V = G.copy()
#     # add edges (ONLY add edges that dont intersect)
#     [V.add_edge(src,dst,layer=-1,hyp=hyp) for src,dst,hyp in zip(
#         poly_node_idx[:-1],poly_node_idx[1:],poly_hyp_idx
#     )]

    # pos = dict(zip(poly_node_idx,[V.nodes[each]['v'] for each in poly_node_idx]))
    # nx.draw(V,pos=pos)


    new_node_start = poly_node_idx[-2]+1
    node_counter = 0

    ### find hyp and edge intersections, add to graph
    
    # create lines and check intersections
    
    if q is None: 
        q = get_intersection_pattern(poly,hyps)

    no_inter_idx = torch.where(torch.prod(q[:-1] == q[1:],axis=1))[0].cpu()
    
    ## if multiple edges are not intersected
    if len(no_inter_idx)>1:
    
        [
            G.add_edge(src,dst,layer=-1,hyp=hyp) for src,dst,hyp in zip(
            poly_node_idx[no_inter_idx],poly_node_idx[no_inter_idx+1],poly_hyp_idx[no_inter_idx])
        ]
    
    ## if one edge is not intersected
    elif len(no_inter_idx)==1:
        
        G.add_edge(poly_node_idx[no_inter_idx],
                   poly_node_idx[no_inter_idx+1],
                   layer=-1,
                   hyp=poly_hyp_idx[no_inter_idx])
    
    ## if all edges are intersected
    else:
        pass
    
    # get intersecting hypidx and associated vertex idx
    hyp_v1_v2_idx = edge_hyp_intersections(q.T,poly,hyps)
    
    # make polytope lines
    poly_lines = make_line_2D(poly[hyp_v1_v2_idx[:,1]],poly[hyp_v1_v2_idx[:,2]])

    # find intersections
#     if hyp_endpoints is None:
        
    poly_int_hyps = hyps[hyp_v1_v2_idx[:,0]]
    v,flag = find_intersection_2D(poly_lines,
                                  poly_int_hyps,
                                  verify=verify)

    v = v.type(poly_lines.type())
    
    if verify:
        
        assert flag
    
        flag = verify_collinear(v,
                                poly[hyp_v1_v2_idx[:,1]],
                                poly[hyp_v1_v2_idx[:,2]]
                                )

        assert flag

    hyp_endpoints = v.reshape(-1,2,v.shape[-1]) ## 

#     else:
        
#         v = hyp_endpoints.reshape(-1,hyp_endpoints.shape[-1])
#         if v.shape[0] != hyp_v1_v2_idx.shape[0]:
#             print(v)
#             print(hyp_v1_v2_idx)

    ### add to graph

    # indices for new nodes
    new_node_idx = torch.from_numpy(
        np.asarray(range(new_node_start,new_node_start+v.shape[0]))
    ).type(torch.int)

    # create list of ordered vertices
    v_collect, hyp_v1_v2_idx_collect, node_names_collect = order_vertices_poly(v,hyp_v1_v2_idx,poly,new_node_idx)
    
    for v_set,hyp_v1_v2_idx_set,node_names in zip(v_collect,hyp_v1_v2_idx_collect,node_names_collect):    

        add_line_to_graph(
            G=G,
            node_names = node_names,
            start = poly_node_idx[hyp_v1_v2_idx_set[0,1]],
            end = poly_node_idx[hyp_v1_v2_idx_set[0,2]],
            v = v_set,
            line_name = poly_hyp_idx[hyp_v1_v2_idx_set[0,1]],
            layer_name = -1 ## still previous layer
        )
    
    
#     pos = dict([(each,G.nodes[each]['v'].numpy()) for each in G.nodes])
#     nx.draw(G,pos=pos,node_size=50)
    
    uniq_hyp_idx = hyp_v1_v2_idx[::2,0] ## all hyps that intersect
    
#     hyp_endpoints = v.reshape(-1,2,v.shape[-1]) ## 
    hyp_endpoint_nodes = new_node_idx.reshape(-1,2)

    
    # if combination idx empty, just connect the endpoints
    if uniq_hyp_idx.shape[0] <= 1: # < because of no intersection case ##TODO: check why no intersection here for deeper layers
        
        [
            G.add_edge(
                nodes[0],nodes[1],layer=0,hyp=name
            ) for nodes,name in zip(hyp_endpoint_nodes.numpy(),uniq_hyp_idx)
        ]
        
        return G
        
    
    # get combination idx of unique hyperplanes that intersect
    comb_idx,no_inter_idx = create_hyp_combinations(hyps=hyps,
                                       hyp_idx=uniq_hyp_idx,
                                       endpoints=hyp_endpoints)
    
    
    ## if combination idx empty, just connect the endpoints
        
    if no_inter_idx.shape[0] != 0:
        
        if no_inter_idx.shape[0] == 1:
        
            G.add_edge(
                hyp_endpoint_nodes.numpy()[no_inter_idx.cpu()][0],
                hyp_endpoint_nodes.numpy()[no_inter_idx.cpu()][1],
                layer=0,
                hyp=uniq_hyp_idx[no_inter_idx.cpu()]
            )
            
        
        else:
            
            [
                G.add_edge(
                    nodes[0],nodes[1],layer=0,hyp=name
                ) for nodes,name in zip(hyp_endpoint_nodes.numpy()[no_inter_idx.cpu()],
                                        uniq_hyp_idx[no_inter_idx.cpu()])
            ]
    

    if comb_idx.shape[0] == 0:
        return G
        
    v,flag = find_intersection_2D(hyps[comb_idx[:,0]],
                                hyps[comb_idx[:,1]],
                                verify=verify)
    if verify:
        assert flag

    new_node_start = new_node_idx[-1]+1
    new_node_idx = torch.arange(new_node_start,new_node_start+v.shape[0]).type(torch.int)

    for ii,each_hyp in tqdm.tqdm(enumerate(uniq_hyp_idx), desc='iterating hyps', total=len(uniq_hyp_idx)):


        mask = torch.logical_or(comb_idx[:,0] == each_hyp, comb_idx[:,1] == each_hyp)

        if not(torch.sum(mask)): #hyp intersects at vertex, hence
            #came up as uniq hyp but didnt come up as combination
            continue

        verts = v[mask].clone()
        hyp_adj = comb_idx[mask].clone()
        nodes = new_node_idx[mask].clone()

        idx = vertex_order_along_line(
                endpoint1=hyp_endpoints[ii,0][None,...],
                endpoint2=hyp_endpoints[ii,1][None,...],
                v=verts
            )

        verts = verts[idx]  
        hyp_adj = hyp_adj[idx]
        nodes = nodes[idx]

        add_line_to_graph(
            G=G,
            node_names = nodes,
            start = hyp_endpoint_nodes[ii,0],
            end = hyp_endpoint_nodes[ii,1],
            v = verts,
            line_name = each_hyp,
            layer_name = 0 ## coming layer
        )
        
    return G

@torch.jit.script
def hyp2input(hyps,Abw):
    '''
    Project hyperplane equations back to the 2D input space.
    '''
    
    hyps = hyps[...,None,:]
    
    hyps_inp = torch.bmm(
        hyps[...,:-1],Abw[...,:-1])
    
    bias_inp = torch.bmm(
        hyps[...,:-1],Abw[...,-1:]) +  hyps[...,-1:]
    
    return torch.cat([hyps_inp,bias_inp],dim=-1)

# @torch.jit.script TODO: Make jittable
def cycles_list2vec(regions, repeat_first : bool = True):
    '''
    Flattens a list of lists into a contiguous tensor representation.
    
    Args:
        regions (list of list of torch.Tensor): Each element is a list of vertices bounding a region.
        repeat_first (bool): If True, implicitly closes the polygon by duplicating the first vertex at the end.
        
    Returns:
        tuple: (out_cycles, cyc_idx, ends)
            - out_cycles: Flattened tensor of all vertices. (num_vertices, num_dimensions)
            - cyc_idx: Vector storing the matching region index from `regions` for each vertex. (num_vertices,)
            - ends: Vector storing the idx of the last vertex in `out_cycles` for each region. (num_regions,)
    '''
    regions = regions.copy()
    
    if repeat_first:
        for i in range(len(regions)):
            regions[i] = torch.vstack([
                regions[i],regions[i][:1]
            ])
        
        
    out_cycles = torch.vstack(regions)
    cyc_idx = torch.zeros(out_cycles.shape[0], dtype=torch.int64)
    
    start = 0
    ends = torch.zeros(len(regions),dtype=torch.int64)
    for i in range(len(regions)):
        
        n = regions[i].shape[0]
        cyc_idx[start:start+n] = i
        start += n
        ends[i] = start
        
    return out_cycles, cyc_idx, ends


@torch.jit.script
def get_edge_hyp_intersections(vec_cyc,hyp_v1_v2_idx,hyps_input):
    '''
    Compute intersection between vectorized cycles and hyperplanes.
    
    Args:
        vec_cyc (torch.Tensor): Flattened vertices of cycles.
        hyp_v1_v2_idx (torch.Tensor): Indices distinguishing which hyperplane (hyp_v1_v2_idx[:,0]) intersects which edge (vec_cyc[hyp_v1_v2_idx[:,1]] to vec_cyc[hyp_v1_v2_idx[:,2]]).
        hyps_input (torch.Tensor): hyperplane equations in input space
        
    Returns:
        torch.Tensor: intersection points
    '''
    
    vec_cyc = vec_cyc.type(torch.float64)
    
    
    poly_lines = make_line_2D(
    vec_cyc[hyp_v1_v2_idx[:,1]],
    vec_cyc[hyp_v1_v2_idx[:,2]]
    )
    
    v,flag = find_intersection_2D(poly_lines,
                              hyps_input,
                              verify=False)
    
#     if not flag:
#         print('intersection flag false')
    
#     flag = verify_collinear(v,
#                         vec_cyc[hyp_v1_v2_idx[:,1]],
#                         vec_cyc[hyp_v1_v2_idx[:,2]]
#                         )
    
#     if not flag:
#         print('collinear flag false')
    
    return v


@torch.jit.script
def create_hyp_combinations(hyps,hyp_idx,endpoints):
    '''
    Identifies intersecting pairs of hyperplanes that are actively crossing to create new vertices.
    
    Args:
        hyps (torch.Tensor): Bank of all hyperplanes.
        hyp_idx (torch.Tensor): Masked subset identifier for active hyperplanes.
        endpoints (torch.Tensor): Boundary geometric points acting as validity constraints.
        
    Returns:
        tuple: (comb_idx, no_inter_idx)
            - comb_idx: Indices representing valid hyperplane pairs.
            - no_inter_idx: Singular hyperplanes that safely cut without colliding into others.
    '''
    ## make sure the number of hyps and endpoints are the same
    assert len(hyp_idx) == endpoints.shape[0]
    
    ## check intersection for endpoints and hyps
    q = get_intersection_pattern(endpoints.reshape(-1,endpoints.shape[-1]),hyps[hyp_idx])
    q = q.reshape(-1,2,hyp_idx.shape[0])
    
    ## only consider endpoints which change pattern 
#     q = np.logical_xor.reduce(q,axis=1)
    q = torch.logical_xor(
                            q[:,0,:].reshape(-1),
                            q[:,1,:].reshape(-1)
                        ).view(q.shape[0],q.shape[-1])
    
    # remove upper triangular and diagonal
    mask = torch.tril(torch.ones_like(q)) 
    q *= torch.logical_not(
        torch.eye(q.shape[0]).to(q.device)
    ) 
    no_inter_idx = torch.where(q.sum(1) == 0)[0]
    q *= mask
    
    # get combination
    loc = torch.where(q)
    comb_idx = torch.stack([
        hyp_idx[loc[0]],hyp_idx[loc[1]]
    ]).T
    
    return comb_idx,no_inter_idx


@torch.no_grad()
def to_next_layer_partition(cycles, Abw, current_layer, NN, dtype=torch.float64, device=DEFAULT_DEVICE):
    '''
    Partitions the current partition defined by cycles, using current layer's hyperplanes.
    
    Args:
        cycles (list): List of lists. Each inner list is a list of vertices defining a 2D polygon.
        Abw (torch.Tensor): Affine parameters for each cycle.
        current_layer (int): Index specifying the target layer for intersection.
        NN: Neural Network instance
        dtype: Numerical precision higher ensures better intersection accuracy.
        device: CPU or GPU targeting matrix routines.
        
    Returns:
        tuple: (res_regions, new_cyc_idx) 
    '''
    
    ## convert cycles to vectorized form
    # vec_cyc: Flattened tensor of all vertices. (num_vertices, num_dimensions)
    # cyc_idx: Vector storing the matching region index from `cycles` for each vertex. (num_vertices,)
    # ends: Vector storing the idx of the last vertex in `vec_cyc` for each region. (num_regions,)
    vec_cyc,cyc_idx,ends = cycles_list2vec(cycles)
    
    ### STEP 1: Find which edges intersect with which hyperplanes

    # forward pass through network
    cycles_next = NN.layers[:current_layer].forward(vec_cyc.to(device))
    
    # get intersection pattern for each vertex (num_vertices, num_hyps)
    q = NN.layers[current_layer].get_intersection_pattern(cycles_next)
    
    # find changes in intersection pattern between two consecutive vertices in vec_cyc
    mask = q.T[...,:-1] != q.T[...,1:]
    mask = mask.cpu()
    
    # set false for vertices between cycles which are not connected
    mask[:,(ends-1)[:-1]] = False
    
    if mask.sum() == 0: ## no change in intersection pattern
        return cycles, torch.arange(len(cycles))
    
    ## get indices for hyperplanes that intersect, and index of the starting vertex of the intersecting edge
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    hyp_vert_cyc_idx = torch.hstack([hyp_vert_idx,cyc_idx[hyp_vert_idx[:,1:]]])
    
    assert torch.all(hyp_vert_cyc_idx[::2,2] == hyp_vert_cyc_idx[1::2,2])
    
    ### STEP 2: Obtain parameters of hyperplanes that intersect
    inter_hyps_idx = torch.unique(hyp_vert_cyc_idx[:,0])
    hyps = NN.layers[current_layer].get_weights(row_idx=inter_hyps_idx)
    hyp_idx_map = torch.ones(q.shape[1],dtype=torch.int64)*(hyps.shape[0]+100) ## initialize with idx out of range
    hyp_idx_map[inter_hyps_idx] = torch.arange(hyps.shape[0], dtype=torch.int64)
    
    ### STEP 3 : Project hyperplanes to 2D input space
    hyps_input = hyp2input(
        hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]].to(device), ## hyps that intersect
        Abw[hyp_vert_cyc_idx[::2,2]].to(device) ## corresponding region Abw
    )[:,0,:]
    
    
    ## order indices into tensor with hyp indices, v1, v2 indices
    hyp_v1_v2_idx= torch.hstack([hyp_vert_idx,hyp_vert_idx[:,-1:]+1])

    ### STEP 4 : Get intersection points betweens hyperplanes and cycle edges
    v = get_edge_hyp_intersections(
        vec_cyc = vec_cyc.to(device),
        hyps_input = torch.repeat_interleave(hyps_input,2,dim=0).to(device),
        hyp_v1_v2_idx = hyp_v1_v2_idx
    )
    
    ## matrix containing the line segments of the hyperplanes (hyps_input.shape[0],2,2)
    hyp_endpoints = v.reshape(-1,2,v.shape[-1])
    
    ## find which unique cycles are intersected
    uniq_cycle_idx = torch.unique(hyp_vert_cyc_idx[:,-1])
    
    res_regions = []
    new_cyc_idx = []
    
    ### STEP 5 : For each unique cycle, find new regions formed
    for target_cycle_idx in tqdm.tqdm(uniq_cycle_idx):
        
        ## get vertices and hyperplanes that intersect with the current cycle
        vert_mask = cyc_idx==target_cycle_idx
        hyp_mask = hyp_vert_cyc_idx[::2,-1] == target_cycle_idx

        ## create graph by computing intersection between cycle edges and hyperplanes additionally provided
        ## as line segments.
        G = create_poly_hyp_graph(
            poly = vec_cyc[vert_mask].to(device),
            hyps = hyps_input[hyp_mask].to(device),
            hyp_endpoints = hyp_endpoints[hyp_mask].to(device),
            dtype = dtype
        )
        
        G = ig.Graph.from_networkx(G)

        G = G.to_graph_tool(
            vertex_attributes={'v':'vector<float>'},
            edge_attributes={'layer':'int','hyp':'int'}
        )
        
        if current_layer == 1:
            print('Finding layer 1 regions')
        
        ## find cycles in the graph
        cycles_new = find_cycles_in_graph(G,return_coordinates=False)

        cycles_new = cycle_nodes2vertices(
            G,
            cycles_new,
            dcast=lambda x: torch.from_numpy(
                np.asarray(x),
            ).type(dtype),
        )
        cycles_new = [torch.vstack(each) for each in cycles_new]
        new_cyc_idx += [target_cycle_idx for i in range(len(cycles_new))]
        
        res_regions += cycles_new
    
    
    ## add cycles that were not intersected
    non_int_cyc_idx = [each_idx for each_idx in torch.arange(len(cycles)) if each_idx not in uniq_cycle_idx]
    
    res_regions += [cycles[each_idx] for each_idx in non_int_cyc_idx]
    new_cyc_idx += non_int_cyc_idx
    
    return res_regions, new_cyc_idx

def _batched_gpu_op(method, data, batch_size, out_size, dtype=torch.float32, workers=2, device=DEFAULT_DEVICE, out_device='cpu'):
    '''
    Executes a callable map operation in minibatches using multi-processed PyTorch workers.
    '''
    
    dataloadr = torch.utils.data.DataLoader(data,
                                      pin_memory=False,
                                      batch_size=batch_size,
                                      num_workers=workers,
                                      shuffle=False,
                                      drop_last=False)
    
    ##malloc
    out = torch.zeros(out_size, device=out_device, dtype=dtype)
    
    start = 0
    for in_batch in dataloadr:
        
        end  = start+in_batch.shape[0]
        out_batch = method(in_batch.to(device))
        out[start:end] = out_batch.to(out_device)
        start = end

    return out

class util_dataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        
        self._len = self.data1.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]


def _batched_gpu_op_2(method, data1, data2, batch_size, out_size, dtype=torch.float32, workers=2, device=DEFAULT_DEVICE):
    '''
    Executes a callable map operation in minibatches using multi-processed PyTorch workers.
    '''
    
    assert data1.shape[0] == data2.shape[0]
    
    dataloadr = torch.utils.data.DataLoader(util_dataset(data1,data2),
                                      pin_memory=True,
                                      batch_size=batch_size,
                                      num_workers=workers,
                                      shuffle=False,
                                      drop_last=False)
    
    ##malloc
    out = torch.zeros(out_size, device='cpu', dtype=dtype)
    
    start = 0
    for in_batch1,in_batch2 in dataloadr:
        
        end  = start+in_batch1.shape[0]
        out_batch = method(in_batch1.to(device),in_batch2.to(device))
        out[start:end] = out_batch.cpu()
        start = end

    return out


@torch.no_grad()
def to_next_layer_partition_batched(cycles, Abw, current_layer, NN,
                                    dtype=torch.float64, device=DEFAULT_DEVICE,
                                    batch_size=-1, fwd_batch_size=-1, workers=2):
    '''
    Batched and parallelized implementation of `to_next_layer_partition`.
    '''
    
    if batch_size == -1: ## revert to non-batched
        res_regions, new_cyc_idx = to_next_layer_partition(
            cycles, Abw, current_layer, NN, dtype, device
        )
        return res_regions, new_cyc_idx
    
    vec_cyc,cyc_idx,ends = cycles_list2vec(cycles)
    
#     cycles_next = NN.layers[:current_layer].forward(vec_cyc.to(device))
#     q = NN.layers[current_layer].get_intersection_pattern(cycles_next)
    
    fused_op = lambda x: NN.layers[current_layer].get_intersection_pattern(
        NN.layers[:current_layer].forward(x))
    
        
    q = _batched_gpu_op(fused_op, 
                        vec_cyc,
                        workers = workers,
                        out_size=(
                            vec_cyc.shape[0],
                            torch.prod(NN.layers[current_layer].output_shape),
                        ),
                        batch_size = fwd_batch_size, out_device='cpu')                                  
    
    
    n_hyps  = torch.prod(NN.layers[current_layer].output_shape)
    
    ## edge intersections. remove between cycles
    mask = q.T[...,:-1] != q.T[...,1:]
    mask = mask.cpu()
    mask[:,(ends-1)[:-1]] = False
    
    if mask.sum() == 0:
        return cycles, torch.arange(len(cycles))
    
#     del  q
#     del cycles_next
    
    ## get indices for hyps-vertex-cycle triads
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    hyp_vert_cyc_idx = torch.hstack([hyp_vert_idx,cyc_idx[hyp_vert_idx[:,1:]]])
    
    ## assert all cycles occur twice in order
    assert torch.all(hyp_vert_cyc_idx[::2,2] == hyp_vert_cyc_idx[1::2,2])
    
    ## query hyps, only get rows which intersect, create idx map
    inter_hyps_idx = torch.unique(hyp_vert_cyc_idx[:,0])
    hyps = NN.layers[current_layer].get_weights(row_idx=inter_hyps_idx).cpu()
    hyp_idx_map = torch.ones(n_hyps,dtype=torch.int64)*(hyps.shape[0]+100) ## initialize with idx out of range
    hyp_idx_map[inter_hyps_idx] = torch.arange(hyps.shape[0], dtype=torch.int64)
    
    ## bring hyps to corresponding cycle inputs
    
    hyps_input = _batched_gpu_op_2(
        method = hyp2input,
        data1 = hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]],
        data2 = Abw[hyp_vert_cyc_idx[::2,2]],
        batch_size = batch_size,
        out_size = (hyp_vert_cyc_idx[::2,0].shape[0],1,3),
        dtype = dtype,
        workers = workers
    )[:,0,:]
        
    
#     hyps_input = hyp2input(
#         hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]].to(device), ## hyps that intersect
#         Abw[hyp_vert_cyc_idx[::2,2]].to(device) ## corresponding region Abw
#     )[:,0,:]
    
    
    ## get intersection with all cycle edges
#     hyp_v1_v2_idx= torch.hstack([hyp_vert_idx,hyp_vert_idx[:,-1:]+1])
#     v = get_edge_hyp_intersections(
#         vec_cyc = vec_cyc.to(device),
#         hyps_input = torch.repeat_interleave(hyps_input,2,dim=0).to(device),
#         hyp_v1_v2_idx = hyp_v1_v2_idx
#     )
    
#     hyp_endpoints = v.reshape(-1,2,v.shape[-1])
    
    ## iterate over each region and obtain new regions
    uniq_cycle_idx = torch.unique(hyp_vert_cyc_idx[:,-1])
    
    res_regions = []
    new_cyc_idx = []
    
    ## for each intersected cycle, find new regions
    for target_cycle_idx in tqdm.tqdm(uniq_cycle_idx, desc='Iterating regions'):
        
        vert_mask = cyc_idx==target_cycle_idx
        hyp_mask = hyp_vert_cyc_idx[::2,-1] == target_cycle_idx
        
        G = create_poly_hyp_graph(
            poly = vec_cyc[vert_mask].to(device),
            hyps = hyps_input[hyp_mask].to(device),
#             hyp_endpoints = hyp_endpoints[hyp_mask].to(device),
            dtype = dtype
        )
        
#         import networkx as nx
#         pos = dict([(each,G.nodes[each]['v']) for each in G.nodes])
#         nx.draw(G,pos=pos)
        
        G = ig.Graph.from_networkx(G)

        G = G.to_graph_tool(
            vertex_attributes={'v':'vector<float>'},
            edge_attributes={'layer':'int','hyp':'int'}
        )
        
        if current_layer == 1:
            print('Finding regions from first layer graph')
        
        cycles_new = find_cycles_in_graph(G,return_coordinates=False)

        cycles_new = cycle_nodes2vertices(
            G,
            cycles_new,
            dcast=lambda x: torch.from_numpy(
                np.asarray(x),
            ).type(dtype),
        )
        cycles_new = [torch.vstack(each) for each in cycles_new]
        
        new_cyc_idx += [target_cycle_idx for i in range(len(cycles_new))]
        
        res_regions += cycles_new
    
    
    ## add cycles that were not intersected
    non_int_cyc_idx = [each_idx for each_idx in torch.arange(len(cycles)) if each_idx not in uniq_cycle_idx]
    
    res_regions += [cycles[each_idx] for each_idx in non_int_cyc_idx]
    new_cyc_idx += non_int_cyc_idx
    
    return res_regions, new_cyc_idx

def networkx2graphtool(G):
    '''
    Converts a NetworkX graph instance to a Graph-Tool instance.
    '''
    
    G = ig.Graph.from_networkx(G)

    G = G.to_graph_tool(
        vertex_attributes={'v':'vector<float>'},
        edge_attributes={'layer':'int','hyp':'int'}
    )
    
    return G