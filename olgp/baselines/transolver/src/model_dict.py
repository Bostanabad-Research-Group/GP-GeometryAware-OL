from olgp.baselines.transolver.src.Transolver_Irregular_Mesh import Model as Transolver_Irregular_Mesh

def get_model(args):
    model_dict = {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh, # for PDEs in 1D space or in unstructured meshes
    }
    return model_dict[args]
