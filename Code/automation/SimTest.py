# -*- coding: utf-8 -*-
import time
import random

class GeoDomain(object):
    def __init__(self):
        """
        Initialise a GeoDomain of either a polygon or a cube.
        Defined by the original coordinates and the sides of the domain
        Parameters
        ----------
        gtype : List of str
            list of geometry types: "2d" or "3d"
        sides : List of float
            Length of sides of the geometry: x, y, (z)
        org_corr : tuple, optional
            Coordinates as a tuple. Length of 2 for 2d and 3 for 3d

        Returns
        -------
        None.

        """
        self.body = Model.Geometry.Children[0].Children[0]
        self.bodyWrapper = self.body.GetGeoBody()
        self.edges = self.bodyWrapper.Edges
        self.edge_lengths = [edge.Length for edge in self.edges]
        self.nodes = 0
        super(GeoDomain, self).__init__()
        
    def meshing(self, size):
        """
        Define mesh size and generate meshed domain
        Parameters
        ----------
        size : 
            Indicating the element size
        Returns
        -------
        None.
        """
        Model.Mesh.ElementSize = Quantity(str(size)+"[m]")
        Model.Mesh.GenerateMesh()
        self.nodes = Model.Mesh.Nodes
        
class PlaneDomain(GeoDomain):
    def __init__(self):
        """
        2D domain
        Returns
        -------
        None.
        """
        super(PlaneDomain, self).__init__()

class SolidDomain(GeoDomain):
    def __init__(self):
        """
        3D domain
        Returns
        -------
        None.
        """
        super(SolidDomain, self).__init__()
        self.faces = self.bodyWrapper.Faces

class SimTest(object):
    def __init__(self, geo_domain):
        """
        

        Parameters
        ----------
        geo_domain : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.geo = geo_domain
        self.body = geo_domain.body
        self.bodyWrapper = geo_domain.bodyWrapper
        self.nodes = None
        self.curr_timing = 0
        self.timings = []
        self.nodelist = []
        super(SimTest, self).__init__()

class Test2d(SimTest):
    def __init__(self, geo_domain, sizes):
        self.sizes = sizes
        super(Test2d, self).__init__(geo_domain)
        
    def random_2D_simple_simulation(self):
        """
        Run a simple 2D simulation by creating a pressure (randomised) and a fixed support
        Returns
        -------
        None.
        """
        self.body.Thickness = Quantity("0.00000000001 [m]")
        p_edge = self.geo.edges[1]
        s_edge = self.geo.edges[3]
        
        sln = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
        p_sln = sln
        p_sln.Entities = [p_edge]
        pressure = Model.Analyses[0].AddPressure()
        pressure.Location = p_sln
        # pressure random from 0 to 10 (float -> string)
        rand_p = str(random.random() * random.randint(1, 10)) 
        pressure.Magnitude.Output.DiscreteValues = [Quantity(rand_p+"[Pa]")]
        
        support = Model.Analyses[0].AddFixedSupport()
        s_sln = sln
        s_sln.Entities = [s_edge]
        support.Location = s_sln
        
        Model.Analyses[0].Solution.AddEquivalentStress()
        Model.Solve()
    
    def time_2d(self):
        for size in self.sizes:
            self.geo.meshing(size)
            start = time.time()
            for i in range(10):
                self.random_2D_simple_simulation()
            end = time.time()
            self.curr_timing = (end - start)/5
            print("Node NO.: %s, Timing: %s" % (self.geo.nodes, self.curr_timing))
            self.timings.append(self.curr_timing)
            self.nodelist.append(self.geo.nodes)
            
class Test3d(SimTest):
    def __init__(self, geo_domain, sizes):
        self.sizes = sizes
        super(Test3d, self).__init__(geo_domain)
        
    def random_3D_simple_simulation(self):
        """
        Run a simple 2D simulation by creating a pressure (randomised) and a fixed support
        Returns
        -------
        None.
        """
        p_face = self.geo.faces[1]
        s_face = self.geo.faces[3]
        
        sln = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
        p_sln = sln
        p_sln.Entities = [p_face]
        pressure = Model.Analyses[0].AddPressure()
        pressure.Location = p_sln
        # pressure random from 0 to 10 (float -> string)
        rand_p = str(random.random() * random.randint(1, 10)) 
        pressure.Magnitude.Output.DiscreteValues = [Quantity(rand_p+"[Pa]")]
        
        support = Model.Analyses[0].AddFixedSupport()
        s_sln = sln
        s_sln.Entities = [s_face]
        support.Location = s_sln
        
        Model.Analyses[0].Solution.AddEquivalentStress()
        Model.Solve()
    
    def time_3d(self):
        for size in self.sizes:
            self.geo.meshing(size)
            start = time.time()
            for i in range(1):
                self.random_3D_simple_simulation()
            end = time.time()
            self.curr_timing = (end - start)/5
            print("Node NO.: %s, Timing: %s" % (self.geo.nodes, self.curr_timing))
            self.timings.append(self.curr_timing)
            self.nodelist.append(self.geo.nodes)

solid = SolidDomain()
sizes = [4e-3]#, 2e-3, 1e-3, 8e-4, 6e-4]
test = Test3d(solid, sizes)
test.time_3d()
