import numpy as np
import networkx as nx

def Bn_construction_helper(n:int):
    # the latitudes where the tetrahedrons are located
    Bn_latitudes = {1:[1.8234765832], 2:[1.5224643081, 1.907334186489793], 3:[1.5122531553, 1.5791609841, 1.9105342920897932], 4:[1.5119462416, 1.5693081599, 1.580856519789793, 1.9106309000897932], 5:[1.5119388208, 1.569063224, 1.5713521159, 1.5808975086897932, 1.910633236289793]}

    data = [np.array([0, 0, 0, 1])]
    # coordinate of regular tetrahedron on the equator
    v1 = np.array([np.sqrt(8/9), 0., -1/3, 0.])
    v2 = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3, 0.])
    v3 = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3, 0.])
    v4 = np.array([0., 0., 1, 0.])
    v = [v1, v2, v3, v4]
    phi = Bn_latitudes[n]
    for i in range(n):
        for j in range(4):
            x1 = np.sin(phi[i]) * v[j][0]
            x2 = np.sin(phi[i]) * v[j][1]
            x3 = np.sin(phi[i]) * v[j][2]
            x4 = np.cos(phi[i])
            data.append(np.array([x1, x2, x3, x4]))
    return data


def get_two_dim_Bn(n:int):
    Bn = {
        2: [np.array([0, 0, 1]), np.array([0.99480443, 0.        , 0.10180448]), np.array([-0.49740221,  0.86152591,  0.10180448]), np.array([-0.49740221, -0.86152591,  0.10180448]), np.array([ 0.87481637,  0.        , -0.48445465]), np.array([-0.43740819,  0.7576132 , -0.48445465]), np.array([-0.43740819, -0.7576132 , -0.48445465])],
        3: [np.array([0, 0, 1]), np.array([0.99018408, 0.        , 0.13976943]), np.array([-0.49509204,  0.85752456,  0.13976943]), np.array([-0.49509204, -0.85752456,  0.13976943]), np.array([ 0.99960625,  0.        , -0.02805968]), np.array([-0.49980312,  0.86568441, -0.02805968]), np.array([-0.49980312, -0.86568441, -0.02805968]), np.array([ 0.86670619,  0.        , -0.49881898]), np.array([-0.4333531 ,  0.75058958, -0.49881898]), np.array([-0.4333531 , -0.75058958, -0.49881898])],
        4: [np.array([0, 0, 1]), np.array([0.98977549, 0.        , 0.14263411]), np.array([-0.49488774,  0.85717071,  0.14263411]), np.array([-0.49488774, -0.85717071,  0.14263411]), np.array([0.99997152, 0.        , 0.00754764]), np.array([-0.49998576,  0.86600074,  0.00754764]), np.array([-0.49998576, -0.86600074,  0.00754764]), np.array([ 0.99928869,  0.        , -0.03771093]), np.array([-0.49964434,  0.86540939, -0.03771093]), np.array([-0.49964434, -0.86540939, -0.03771093]), np.array([ 0.86607469,  0.        , -0.49991462]), np.array([-0.43303734,  0.75004268, -0.49991462]), np.array([-0.43303734, -0.75004268, -0.49991462])],
        5: [np.array([0, 0, 1]), np.array([0.98974565, 0.        , 0.14284099]), np.array([-0.49487282,  0.85714488,  0.14284099]), np.array([-0.49487282, -0.85714488,  0.14284099]), np.array([0.99994894, 0.        , 0.01010566]), np.array([-0.49997447,  0.86598118,  0.01010566]), np.array([-0.49997447, -0.86598118,  0.01010566]), np.array([ 0.99999804,  0.        , -0.00197826]), np.array([-0.49999902,  0.86602371, -0.00197826]), np.array([-0.49999902, -0.86602371, -0.00197826]), np.array([ 0.99926217,  0.        , -0.03840714]), np.array([-0.49963109,  0.86538643, -0.03840714]), np.array([-0.49963109, -0.86538643, -0.03840714]), np.array([ 0.86602897,  0.        , -0.49999382]), np.array([-0.43301449,  0.75000309, -0.49999382]), np.array([-0.43301449, -0.75000309, -0.49999382])],
        6:[np.array([0, 0,1]), np.array([0.9897432445692924, 0.0,0.14285765583072446]), np.array([-0.49487162228464593, 0.8571427930210419,0.14285765583072446]), np.array([-0.49487162228464654, -0.8571427930210415,0.14285765583072446]), np.array([0.9999467894937784, 0.0,0.010315918819247897]), np.array([-0.4999733947468889, 0.8659793221343025,0.010315918819247897]), np.array([-0.4999733947468895, -0.8659793221343021,0.010315918819247897]), np.array([0.9999998133980929, 0.0,-0.0006109040671047638]), np.array([-0.49999990669904615, 0.8660252421824467,-0.0006109040671047638]), np.array([-0.49999990669904676, -0.8660252421824463,-0.0006109040671047638]), np.array([0.9999960957358961, 0.0,-0.0027943716582357297]), np.array([-0.49999804786794777, 0.8660220225925417,-0.0027943716582357297]), np.array([-0.4999980478679484, -0.8660220225925412,-0.0027943716582357297]), np.array([0.9992600146271758, 0.0,-0.03846327036663086]), np.array([-0.49963000731358764, 0.865384557653144,-0.03846327036663086]), np.array([-0.49963000731358825, -0.8653845576531436,-0.03846327036663086]), np.array([0.8660252904361175, 0.0,-0.5000001963249996]), np.array([-0.43301264521805855, 0.7499999018374744,-0.5000001963249996]), np.array([-0.43301264521805904, -0.7499999018374741,-0.5000001963249996])]
    }
    return Bn[n]

def get_Bn_construction(n:int):
    Bn = {
        1: [np.array([0, 0, 0, 1]), np.array([ 0.91287093,  0.        , -0.32274861, -0.25      ]), np.array([-0.45643546,  0.79056941, -0.32274861, -0.25      ]), np.array([-0.45643546, -0.79056941, -0.32274861, -0.25      ]), np.array([ 0.        ,  0.        ,  0.96824584, -0.25      ])], 
        2:[np.array([0, 0, 0, 1]), np.array([ 0.94170806,  0.        , -0.33294408,  0.0483132 ]), np.array([-0.47085403,  0.81554311, -0.33294408,  0.0483132 ]), np.array([-0.47085403, -0.81554311, -0.33294408,  0.0483132 ]), np.array([0.        , 0.        , 0.99883224, 0.0483132 ]), np.array([ 0.88992084,  0.        , -0.31463453, -0.33022115]), np.array([-0.44496042,  0.77069406, -0.31463453, -0.33022115]), np.array([-0.44496042, -0.77069406, -0.31463453, -0.33022115]), np.array([ 0.        ,  0.        ,  0.94390359, -0.33022115])], 
        3:[np.array([0, 0, 0, 1]), np.array([ 0.94119386,  0.        , -0.33276228,  0.05850974]), np.array([-0.47059693,  0.81509779, -0.33276228,  0.05850974]), np.array([-0.47059693, -0.81509779, -0.33276228,  0.05850974]), np.array([0.        , 0.        , 0.99828684, 0.05850974]), np.array([ 0.94277606,  0.        , -0.33332167, -0.00836456]), np.array([-0.47138803,  0.81646802, -0.33332167, -0.00836456]), np.array([-0.47138803, -0.81646802, -0.33332167, -0.00836456]), np.array([ 0.        ,  0.        ,  0.99996502, -0.00836456]), np.array([ 0.88891998,  0.        , -0.31428067, -0.33324005]), np.array([-0.44445999,  0.76982728, -0.31428067, -0.33324005]), np.array([-0.44445999, -0.76982728, -0.31428067, -0.33324005]), np.array([ 0.        ,  0.        ,  0.94284202, -0.33324005])], 
        4:[np.array([0, 0, 0, 1]), np.array([ 0.94117688,  0.        , -0.33275628,  0.05881612]), np.array([-0.47058844,  0.81508309, -0.33275628,  0.05881612]), np.array([-0.47058844, -0.81508309, -0.33275628,  0.05881612]), np.array([0.        , 0.        , 0.99826883, 0.05881612]), np.array([ 0.942808  ,  0.        , -0.33333296,  0.00148817]), np.array([-0.471404  ,  0.81649568, -0.33333296,  0.00148817]), np.array([-0.471404  , -0.81649568, -0.33333296,  0.00148817]), np.array([0.        , 0.        , 0.99999889, 0.00148817]), np.array([ 0.94276133,  0.        , -0.33331647, -0.01006002]), np.array([-0.47138067,  0.81645526, -0.33331647, -0.01006002]), np.array([-0.47138067, -0.81645526, -0.33331647, -0.01006002]), np.array([ 0.        ,  0.        ,  0.9999494 , -0.01006002]), np.array([ 0.88888962,  0.        , -0.31426994, -0.33333113]), np.array([-0.44444481,  0.76980099, -0.31426994, -0.33333113]), np.array([-0.44444481, -0.76980099, -0.31426994, -0.33333113]), np.array([ 0.        ,  0.        ,  0.94280982, -0.33333113])],
        5: [np.array([0, 0, 0, 1]), np.array([ 0.94117647,  0.        , -0.33275613,  0.05882353]), np.array([-0.47058824,  0.81508273, -0.33275613,  0.05882353]), np.array([-0.47058824, -0.81508273, -0.33275613,  0.05882353]), np.array([0.        , 0.        , 0.9982684 , 0.05882353]), np.array([ 0.94280763,  0.        , -0.33333283,  0.0017331 ]), np.array([-0.47140381,  0.81649535, -0.33333283,  0.0017331 ]), np.array([-0.47140381, -0.81649535, -0.33333283,  0.0017331 ]), np.array([0.       , 0.       , 0.9999985, 0.0017331]), np.array([ 9.42808896e-01,  0.00000000e+00, -3.33333282e-01, -5.55789076e-04]), np.array([-4.71404448e-01,  8.16496455e-01, -3.33333282e-01, -5.55789076e-04]), np.array([-4.71404448e-01, -8.16496455e-01, -3.33333282e-01, -5.55789076e-04]), np.array([ 0.00000000e+00,  0.00000000e+00,  9.99999846e-01, -5.55789076e-04]), np.array([ 0.94276094,  0.        , -0.33331633, -0.01010101]), np.array([-0.47138047,  0.81645493, -0.33331633, -0.01010101]), np.array([-0.47138047, -0.81645493, -0.33331633, -0.01010101]), np.array([ 0.        ,  0.        ,  0.99994898, -0.01010101]), np.array([ 0.88888889,  0.        , -0.31426968, -0.33333333]), np.array([-0.44444444,  0.76980036, -0.31426968, -0.33333333]), np.array([-0.44444444, -0.76980036, -0.31426968, -0.33333333]), np.array([ 0.        ,  0.        ,  0.94280904, -0.33333333])]}
    return Bn[n]


def get_two_dim_En(n):
    En = {
        2: [np.array([0, 0, 1]), np.array([0.9735071 , 0.        , 0.22865677]), np.array([0.30083024, 0.92586027, 0.22865677]), np.array([-0.78758379,  0.57221312,  0.22865677]), np.array([-0.78758379, -0.57221312,  0.22865677]), np.array([ 0.30083024, -0.92586027,  0.22865677]), np.array([ 0.69970233,  0.        , -0.7144345 ]), np.array([ 0.21621991,  0.66545646, -0.7144345 ]), np.array([-0.56607108,  0.41127471, -0.7144345 ]), np.array([-0.56607108, -0.41127471, -0.7144345 ]), np.array([ 0.21621991, -0.66545646, -0.7144345 ])],
        3: [np.array([0, 0, 1]), np.array([0.91097863, 0.        , 0.41245355]), np.array([0.28150788, 0.86639217, 0.41245355]), np.array([-0.7369972 ,  0.53545981,  0.41245355]), np.array([-0.7369972 , -0.53545981,  0.41245355]), np.array([ 0.28150788, -0.86639217,  0.41245355]), np.array([ 0.99248491,  0.        , -0.12236705]), np.array([ 0.30669471,  0.94390924, -0.12236705]), np.array([-0.80293716,  0.583368  , -0.12236705]), np.array([-0.80293716, -0.583368  , -0.12236705]), np.array([ 0.30669471, -0.94390924, -0.12236705]), np.array([ 0.62336709,  0.        , -0.78192932]), np.array([ 0.19263103,  0.59285734, -0.78192932]), np.array([-0.50431457,  0.36640598, -0.78192932]), np.array([-0.50431457, -0.36640598, -0.78192932]), np.array([ 0.19263103, -0.59285734, -0.78192932])]
    }
    return En[n]


def four_n_plus_one(n):
    data = [np.array([0, 0, 0, 1])]
    # coordinate of regular tetrahedron on the equator
    v1 = np.array([np.sqrt(8/9), 0., -1/3, 0.])
    v2 = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3, 0.])
    v3 = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3, 0.])
    v4 = np.array([0., 0., 1, 0.])
    v = [v1, v2, v3, v4]
    phi = np.zeros(n)
    while (phi[n-1] < math.acos(-1/4)):
        phi = np.random.rand(n)*(math.acos(-1/3))
        # print('phi: ', phi)

    for i in range(n):
        #rotation
        # theta = random.uniform(0, 2*math.pi)

        for j in range(4):
            x1 = np.sin(phi[i]) * v[j][0]
            x2 = np.sin(phi[i]) * v[j][1]
            x3 = np.sin(phi[i]) * v[j][2]
            x4 = np.cos(phi[i])
            data.append(np.array([x1, x2, x3, x4]))
    del data[-1]
    return data

class ASDpolygon:
    def __init__(self, data, mFace, n_mFace, n_tri, e_diam, s_diam, diam_graph, n_edge=None, n_tetrahedra=None, n_ridge= None):
        self.data = data
        self.n_pts = diam_graph.number_of_nodes()
        self.mFace = mFace
        self.n_mFace = n_mFace
        self.n_tri = n_tri
        self.e_diam = e_diam
        self.s_diam = s_diam
        self.diam_graph = diam_graph
        self.n_edge = n_edge
        self.n_tetrahedra = n_tetrahedra
        self.n_ridge = n_ridge

    def _key(self):
        return (self.n_pts, self.mFace, self.n_mFace, self.n_tri)

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other):
        if isinstance(other, ASDpolygon):
            return abs(self.s_diam -other.s_diam) < 1e-4 and nx.is_isomorphic(self.diam_graph, other.diam_graph)

    def details(self):
        print('Number of points: ', self.n_pts)
        print('Maximum number of edges in a face: ', self.mFace)
        print('Number of faces with maximal number of faces: ', self.n_mFace)
        print('Number of triangles in diameter graph: ', self.n_tri)
        print('Euclidean diameter: ', self.e_diam)
        print('Spherical diameter: ', self.s_diam)
        print('self.diam_graph: ', self.diam_graph)
        print('Number of edges: ', self.n_edge)
        print('Number of tetrahedra: ', self.n_tetrahedra)
        print('Number of ridges: ', self.n_ridge)
