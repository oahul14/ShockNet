import csv
import time
import random
import copy
import os

sln = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
node_sln = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)

def quantify_val_list(val_list, unit):
    return [Quantity(str(i) + "[" + unit + "]") for i in val_list]

def quantify_val(val, unit):
    return Quantity(str(val) + "[" + unit + "]")
    
def get_next_file_number(path):
    fnames = os.listdir(path)
    if not fnames:
        return 1
    split = lambda fname : int(fname.split('_')[0])
    fnumbers = map(split, fnames)
    return max(fnumbers) + 1
    
def expMaxStressImg(path):
    Graphics.Camera.SceneHeight = quantify_val(0.001, 'm')
    Graphics.Scene.LineWeight = 0
    img = Ansys.Mechanical.Graphics.GraphicsImageExportSettings()
    img.Background = GraphicsBackgroundType.White
    img.Resolution = GraphicsResolutionType.HighResolution
    img.Capture = GraphicsCaptureType.ImageOnly
    img.Height = 10
    Graphics.ExportImage(path, GraphicsImageExportFormat.EPS, img)

class DomainObject(object):
    def __init__(self):
        self.analysis_settings = Model.Analyses[0].AnalysisSettings
        self.body = Model.Geometry.Children[0].Children[0]
        self.bodyWrapper = self.body.GetGeoBody()
        self.edges = self.bodyWrapper.Edges
        self.nodes = 0
        self.force = None
        self.names_selection = None
        self.nodal_force = None
        self.node_list = None
        self.support = None
        super(DomainObject, self).__init__()

    def meshing(self, size):
        
        Model.Mesh.ElementSize = Quantity(str(size)+"[m]")
        Model.Mesh.GenerateMesh()
        self.nodes = Model.Mesh.Nodes
    
    def config_geometry(self, material, thickness=1e-8):
        Model.Geometry.Model2DBehavior = Model2DBehavior.PlaneStrain
        self.body.Material = material
        self.body.Thickness = quantify_val(thickness, 'm')
    
    def config_analysis(self, number_of_steps=1, 
                    max_energy_err=999999, 
                    min_time_step=1e-9, 
                    max_time_step=1e-6,
                    safety_factor=0.5,
                    result_number=200):
        self.analysis_settings.NumberOfSteps = number_of_steps
        self.analysis_settings.MaximumEnergyError = max_energy_err
        self.analysis_settings.MinimumTimeStep = quantify_val(min_time_step, "sec")
        self.analysis_settings.MaximumTimeStep = quantify_val(max_time_step, "sec")
        self.analysis_settings.TimeStepSafetyFactor = safety_factor
        self.analysis_settings.SetSaveResultsOnPoints(1, result_number)
        
    def addSupport(self, loc):
        support_sln = sln
        support_sln.Entities = [self.edges[loc]]
        self.support = Model.Analyses[0].AddFixedSupport()
        self.support.Location = support_sln
        
    def addForce(self, loc):
        self.force = Model.Analyses[0].AddForce()
        force_sln = sln
        force_sln.Entities = [self.edges[loc]]
        self.force.Location = force_sln
        
    def randBoundaryForce(self, maxforce=2, 
                                periods=[(0, 2e-4), (1.5e-4, 3e-4)], 
                                shock_interval=1e-8, 
                                factor=2):
        # contains 6 time steps that define the force
        rand_periods = copy.deepcopy(periods)
        rand_periods[0] = float("{:.8f}".format(random.uniform(periods[0][0], periods[0][1])))
        rand_periods[1] = float("{:.8f}".format(random.uniform(periods[1][0], periods[1][1])))
        shock_shape = [0., rand_periods[0], shock_interval, rand_periods[1], shock_interval]
        
        # shock_shape = [0.] + [random.uniform(i[0], i[1]) for i in periods]

        for i in range(1, len(shock_shape)):
            shock_shape[i] += shock_shape[i-1]
        shock_shape.append(factor * shock_shape[-1])
        shock_shape = ["{:.8f}".format(i) for i in shock_shape]
        time_quantities = quantify_val_list(shock_shape, "sec")

        self.analysis_settings.SetStepEndTime(1, time_quantities[-1])

        # define randomised force
        randforce = random.uniform(0, maxforce)
        force_dist = [0, 0, randforce, randforce, 0, 0]
        force_quantities = quantify_val_list(force_dist, "N")

        # define randomised tabular force on boundary
        self.force.DefineBy = LoadDefineBy.Components
        # set X component to 0
        self.force.XComponent.Output.DiscreteValues = [Quantity("0 [N]")] 
        self.force.YComponent.inputs[0].DiscreteValues = time_quantities
        self.force.YComponent.Output.DiscreteValues = force_quantities
        
        return shock_shape, randforce
    
    def simRandShockByBoundary(self, iters, path, start_step=False, 
                                                support_loc=1, 
                                                force_loc=3, 
                                                result_interval=10, 
                                                maxforce=2, 
                                                periods=[(0, 2e-4), (1.5e-4, 3e-4)], 
                                                shock_interval=1e-8, 
                                                factor=4):
        stress = Model.Analyses[0].Solution.AddEquivalentStress()
        stress.By = SetDriverStyle.MaximumOverTime
        stress.AddImage()
        
        self.addSupport(support_loc)
        
        step = 0
        if not start_step:
            step = get_next_file_number(path)
            
        iter_start = time.time()
        for i in range(iters):
            self.addForce(force_loc)
            shock_shape, randforce = self.randBoundaryForce(maxforce=maxforce, \
                                                            periods=periods, \
                                                            shock_interval=shock_interval, \
                                                            factor=factor)
            # solve
            sim_start = time.time()
            Model.Solve()
            sim_end = time.time()
            
            max_stress_para = stress.MaximumOfMaximumOverTime.ToString().Split(' ')[0]
            min_stress_para = stress.MinimumOfMinimumOverTime.ToString().Split(' ')[0]
            # define file name
            shock_shape_paras = [str(i) for i in shock_shape]
            force_para = str(randforce)
            time_para = "{:.4f}".format(sim_end - sim_start)
            # max_stress_para = "{:.8f}".format(max_stress)
            name_paras = [str(step), time_para, max_stress_para, min_stress_para, force_para] \
                        + shock_shape_paras
            img_name = '_'.join(name_paras).replace('.', 'd') + ".eps"
            
            stress.EvaluateAllResults()
            expMaxStressImg(path + "\\" + img_name)
            
            self.force.Delete()
            stress.ClearGeneratedData()
            
            step += 1
        
        stress.Delete()
        self.support.Delete()
        iter_end = time.time()
        
        print("Finishied iter number: ", iters)
        print("Total time usage: ", iter_end - iter_start)
        print("Average time usage per iteration: ", (iter_end - iter_start)/iters)
        

        
        
    def addNodalForce(self, lend, rend):
        self.names_selection = Model.AddNamedSelection()
        pos = random.randint(0, 2)
        node_list = []
        if pos == 0:
            src_len = random.randint(1, rend - lend + 1)
            node_list = [i for i in range(lend, lend + src_len)]
        elif pos == 2:
            src_len = random.randint(1, rend - lend + 1)
            node_list = [i for i in range(rend - src_len + 1, rend)]
        else:
            half = (rend + lend) / 2
            ltmp, rtmp = random.randint(lend, half), random.randint(half, rend + 1)
            node_list = [i for i in range(ltmp, rtmp)]
        
        self.node_list = node_list
        
        names_sln = node_sln
        names_sln.Ids = node_list
        ExtAPI.SelectionManager.NewSelection(names_sln)
        self.names_selection.Location = names_sln
        
        self.nodal_force = Model.Analyses[0].AddNodalForce()
        self.nodal_force.Location = self.names_selection
        
        
    def randNodalForce(self, maxforce=2, 
                                lend=53, rend=76,
                                periods=[(0, 2e-4), (1.5e-4, 3e-4)], 
                                shock_interval=1e-8, 
                                factor=2):
        # contains 6 time steps that define the force
        rand_periods = copy.deepcopy(periods)
        rand_periods[0] = float("{:.8f}".format(random.uniform(periods[0][0], periods[0][1])))
        rand_periods[1] = float("{:.8f}".format(random.uniform(periods[1][0], periods[1][1])))
        shock_shape = [0., rand_periods[0], shock_interval, rand_periods[1], shock_interval]
        
        # shock_shape = [0.] + [random.uniform(i[0], i[1]) for i in periods]
        
        for i in range(1, len(shock_shape)):
            shock_shape[i] += shock_shape[i-1]
        shock_shape.append(factor * shock_shape[-1])
        shock_shape = ["{:.8f}".format(i) for i in shock_shape]
        time_quantities = quantify_val_list(shock_shape, "sec")
        
        self.analysis_settings.SetStepEndTime(1, time_quantities[-1])
        
        # define randomised force
        randforce = random.uniform(0, maxforce)
        force_dist = [0, 0, randforce, randforce, 0, 0]
        force_quantities = quantify_val_list(force_dist, "N")
        
        # set X component to 0
        self.nodal_force.XComponent.Output.DiscreteValues = [Quantity("0 [N]")] 
        self.nodal_force.YComponent.Inputs[0].DiscreteValues = time_quantities
        self.nodal_force.YComponent.Output.DiscreteValues = force_quantities
        
        return shock_shape, randforce
    
    def simRandShockByNodes(self, iters, path, start_step=False, 
                                                lend=53,
                                                rend=76,
                                                support_loc=1, 
                                                result_interval=10, 
                                                maxforce=2, 
                                                periods=[(0, 2e-4), (1.5e-4, 3e-4)], 
                                                shock_interval=1e-8, 
                                                factor=4):
        stress = Model.Analyses[0].Solution.AddEquivalentStress()
        stress.By = SetDriverStyle.MaximumOverTime
        stress.AddImage()
        
        self.addSupport(support_loc)
        
        step = 0
        if not start_step:
            step = get_next_file_number(path)
            
        iter_start = time.time()
        for i in range(iters):
            self.addNodalForce(lend, rend)
            shock_shape, randforce = self.randNodalForce(maxforce=maxforce, \
                                                            periods=periods, \
                                                            shock_interval=shock_interval, \
                                                            factor=factor)
            # solve
            sim_start = time.time()
            Model.Solve()
            sim_end = time.time()
            
            max_stress_para = stress.MaximumOfMaximumOverTime.ToString().Split(' ')[0]
            min_stress_para = stress.MinimumOfMinimumOverTime.ToString().Split(' ')[0]
            # define file name
            shock_shape_paras = [str(i) for i in shock_shape]
            force_para = str(randforce)
            node_para = '-'.join([str(lend), str(self.node_list[0]), str(self.node_list[-1]), str(rend)])
            # max_stress_para = "{:.8f}".format(max_stress)
            name_paras = [str(step), node_para, max_stress_para, min_stress_para, force_para] \
                        + shock_shape_paras
            img_name = '_'.join(name_paras).replace('.', 'd') + ".eps"
            
            stress.EvaluateAllResults()
            expMaxStressImg(path + "\\" + img_name)
            
            self.nodal_force.Delete()
            self.names_selection.Delete()
            stress.ClearGeneratedData()
            
            step += 1
        
        stress.Delete()
        self.support.Delete()
        iter_end = time.time()
        
        print("Finishied iter number: ", iters)
        print("Total time usage: ", iter_end - iter_start)
        print("Average time usage per iteration: ", (iter_end - iter_start)/iters)

def run_simulation(data_type, material, n):
    dm = DomainObject()
    dm.config_geometry(material)
    dm.config_analysis()
    path = os.path.join(os.path.dirname("/your-path-to/NumericalResults"), data_type + material)
    if not os.path.isdir(path):
        os.mkdir(path)
    if data_type == "nodal":
        dm.simRandShockByNodes(n, path)
    elif data_type == "boundary":
        dm.simRandShockByBoundary(n, path)
    else:
        print("Only nodal and boundary simulations are implemented")
        raise NotImplementedError

if __name__ == "__main__":
    n = 3000
    materials = ["Dolomite", "Limestone"]
    data_type = "nodal"
    for mat in materials:
        run_simulation(data_type, mat, n)


