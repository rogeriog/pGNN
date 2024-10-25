import numpy as np
from keras.models import Model
import warnings
from pickle import load, dump


import pandas as pd
import os
from typing import Tuple, Any

warnings.filterwarnings("ignore")
from megnet.utils.models import load_model, AVAILABLE_MODELS
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

### FUNCTIONS TO SETUP, EVALUATE AND TRAIN MEGNET MODELS

def model_setup(ntarget: int = None,
                **kwargs) -> Any:
    """
    This function takes in a number of optional parameters for creating a MEGNet model, such as number of neurons 
    in different layers, and the number of features for bonds.
    It returns an instance of a MEGNet model which is set up with the given parameters.
    """
    ## default architecture:
    n1=kwargs.get('n1', 64) 
    n2=kwargs.get('n2', 32) 
    n3=kwargs.get('n3', 16)
    nfeat_bond = kwargs.get('nfeat_bond', 100)
    r_cutoff = kwargs.get('r_cutoff', 5)
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = kwargs.get('gaussian_width', 0.5)
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                        ntarget=ntarget, **kwargs)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary.splitlines()[-4])
    return model

def load_model_scaler(id: str = '',
                      n_targets: int = 1 ,
                      neuron_layers: Tuple[int] = (64,32,16),
                      **kwargs) -> Tuple[Any, Any]:
    """
    This function takes in an id, number of targets, a mode, and other optional parameters for loading a previously trained MEGNet model and its corresponding scaler.
    It returns a tuple of the loaded model and scaler.
    """
    n1,n2,n3=neuron_layers
    model = model_setup(ntarget=n_targets, n1=n1, n2=n2, n3=n3,
                        **kwargs)
    modelpath_id = kwargs.get("modeldir", "./")+id
    model_file=kwargs.get('model_file',f"{modelpath_id}_weights.h5")
    scaler_file=kwargs.get('scaler_file',f'{modelpath_id}_scaler.pkl')
    model.load_weights(model_file)
    try: ## if scaler not found, it will be None
        scaler = load(open(scaler_file, 'rb'))
    except:
        scaler = None
    return (model, scaler)

def megnet_evaluate_structures(model, structures,
                               targets=None,
                               scaler=None, **kwargs):

    labels = kwargs.get('labels', ['']*len(structures))

    noTargets=False
    if targets is None:
        target_values = np.array([1]*len(structures))
        noTargets=True
    else:
        if isinstance(targets, pd.DataFrame):
            target_values=targets.values
        else:
            target_values=targets
    # have to exclude structures that dont form compatible graphs and their corresponding targets.
    structures_valid = []
    targets_valid = []
    labels_valid = []
    structures_invalid = []
    for s, p, l in zip(structures, target_values, labels):
        try:
            graph = model.graph_converter.convert(s)
            structures_valid.append(s)
            if scaler is not None:
                targets_valid.append(np.nan_to_num(
                    scaler.transform(p.reshape(1, -1))))
            else:
                targets_valid.append(p)
            labels_valid.append(l)
        except:
            structures_invalid.append(s)
    # structures_valid = np.array(structures_valid)

    y = np.array(targets_valid)
    y = y.squeeze()
    labels = np.array(labels_valid)
    print(f"Following invalid structures: {structures_invalid}.")
    # print(type(structures_valid),structures_valid)
    ypred = model.predict_structures(list(structures_valid))
    if noTargets:
        return (structures_valid,ypred)
    if not noTargets:
        return (structures_valid,ypred, y, labels)
    # y_pred=y_pred.flatten()

def train_MEGNet_on_the_fly(structures, targets, **kwargs):
        # apply a scaler to the targets
        from sklearn.preprocessing import MinMaxScaler
        targets = np.array(targets)
        targets = targets.reshape(-1,1)
        scaler = MinMaxScaler()
        targets = scaler.fit_transform(targets)
        adjacent_model_path = kwargs.get('adjacent_model_path', '.')
        # create folder if it does not exist
        if not os.path.exists(adjacent_model_path):
            os.makedirs(adjacent_model_path)
        # save scaler to pickle
        dump(scaler, open(os.path.join(adjacent_model_path, 'MEGNetModel__adjacent_scaler.pkl'), 'wb'))
        print('Scaler of the targets for adjacent model saved to MEGNetModel__adjacent_scaler.pkl')
        # train a MEGNet model on the fly to predict a new set of features
        max_epochs=kwargs.get('max_epochs',100)
        patience = kwargs.get('patience',10)
        n1=kwargs.get('n1', 64) 
        n2=kwargs.get('n2', 32) 
        n3=kwargs.get('n3', 16)
        nfeat_bond = kwargs.get('nfeat_bond', 100)
        r_cutoff = kwargs.get('r_cutoff', 5)
        gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
        gaussian_width = kwargs.get('gaussian_width', 0.5)
        graph_converter = CrystalGraph(cutoff=r_cutoff)
        early_stopping = EarlyStopping(monitor='val_mae',patience=patience, restore_best_weights=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # we use just one k fold to get the validation set as criteria for convergence
        train_index, val_index = list(kf.split(structures))[0]
        train_structures, val_structures = [structures.iloc[i] for i in train_index], [structures.iloc[i] for i in val_index]
        train_targets, val_targets = [targets[i] for i in train_index], [targets[i] for i in val_index]

        model = MEGNetModel(metrics=['mae'], graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                            )
        model.train(train_structures, train_targets, validation_structures=val_structures, 
                    validation_targets=val_targets, epochs=max_epochs, save_checkpoint=False, callbacks=[early_stopping])
        model.save(os.path.join(adjacent_model_path, 'MEGNetModel__adjacent.h5'))
        print(f'MEGNet model on the fly saved to {os.path.join(adjacent_model_path, "MEGNetModel__adjacent.h5")}')

### FUNCTIONS TO OBTAIN FEATURE DATAFRAMES FROM MEGNET MODELS
from keras.models import Model
import pandas as pd
import numpy as np

def get_MVL_MEGNetFeatures(structures, layer_name='layer32'):
    """
    Extracts features from a specified intermediate layer of the MEGNet model.
    
    Parameters:
    - structures: List of structures for which to extract features.
    - layer_name: Name of the layer to extract features from. Options are 'layer16' or 'layer32'.
    
    Returns:
    - DataFrame containing the extracted features for each structure.
    """
    # Map layer names to indices. Assuming layer16 corresponds to the -2 layer and layer32 to the -3 layer.
    layer_mapping = {
        'layer16': -2,  # Second-to-last layer (16 neurons)
        'layer32': -3   # Third-to-last layer (32 neurons)
    }
    
    if layer_name not in layer_mapping:
        raise ValueError("Invalid layer_name. Choose 'layer16' or 'layer32'.")
    
    layer_idx = layer_mapping[layer_name]
    
    MVL_MEGNetFeats = []  # To store features from all models
    for model_name in ['Efermi_MP_2019', 'Bandgap_MP_2018', 'logK_MP_2019', 'logG_MP_2019']:
        model = load_model(model_name)  # Load the model
        # Create a model that outputs the specified intermediate layer
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.layers[layer_idx].output)
        
        MEGNetModel_structs = []  # To store features for the current model
        indexes = structures.index.to_list()
        for original_index, s in zip(indexes, structures):
            try:
                # Convert structure to graph input
                graph = model.graph_converter.convert(s)
                inp = model.graph_converter.graph_to_input(graph)
                # Predict using the intermediate layer model
                pred = intermediate_layer_model.predict(inp, verbose=False)
                # Create DataFrame for the prediction
                s_model_results = pd.DataFrame([pred[0][0]], 
                                               columns=[f"{model_name}_{idx+1}" for idx in range(len(pred[0][0]))],
                                               index=[original_index])
                MEGNetModel_structs.append(s_model_results)
            except Exception as e:
                print(e)
                print("Probably an invalid structure was passed to the model, continuing..")
                # Create a DataFrame with NaN values in case of failure
                nans = [[np.nan] * pred.shape[2]]  # Adjust size to match the layer output
                columns = [f"{model_name}_{idx+1}" for idx in range(pred.shape[2])]
                s_model_results = pd.DataFrame(nans, columns=columns, index=[original_index])
                MEGNetModel_structs.append(s_model_results)
                continue
        
        # Concatenate results for the current model
        MEGNetModel_structs = pd.concat(MEGNetModel_structs, axis=0)
        MVL_MEGNetFeats.append(MEGNetModel_structs)
        print(f"Features calculated for model {model_name}.")
    
    # Concatenate all models' features into a final DataFrame
    MVL_MEGNetFeats = pd.concat(MVL_MEGNetFeats, axis=1)
    return MVL_MEGNetFeats


def get_Custom_MEGNetFeatures(structures,
                              model_type: str,
                              n_targets: int = 1,
                              neuron_layers: Tuple[int] = (64, 32, 16), 
                              model=None, 
                              model_file=None, 
                              scaler=None,
                              scaler_file=None,
                              **kwargs):
    """
    Extracts features from a specified MEGNet model, either passed directly or loaded from file.
    
    Parameters:
    - structures: List of structures for which to extract features.
    - model_type: The type of model to use ('MatMinerEncoded_v1', 'OFMEncoded_v1', etc).
    - n_targets: Number of target features.
    - neuron_layers: Tuple representing the number of neurons in each hidden layer.
    - model: The MEGNet model (optional, if not provided, it will be loaded from model_file).
    - model_file: Path to the model file (optional).
    - scaler: Scaler to transform the features back if provided (optional).
    - scaler_file: Path to the scaler file (optional).
    - kwargs: Additional arguments.

    Returns:
    - DataFrame containing the extracted features.
    """
    
    # Set model-specific configurations and file paths
    if model_type == 'MatMinerEncoded_v1':
        # Ensure the model is loaded from custom_models/ directory
        if model_file is None:
            parent_dir = os.path.dirname(__file__)
            model_file = os.path.join(parent_dir, 'custom_models', 'MEGNetModel__MatMinerEncoded_v1.h5')
        if not os.path.isfile(model_file):
            raise ValueError('MEGNetModel__MatMinerEncoded_v1.h5 not found in custom_models directory.')

        file_path_without_ext, _ = os.path.splitext(model_file)
        scaler_file = scaler_file or file_path_without_ext + "_scaler.pkl"
        n_targets = 758
        neuron_layers = (64, 128, 64)
        model_name = "MatMinerEncoded_v1"

    elif model_type == 'OFMEncoded_v1':
        # Ensure the model is loaded from custom_models/ directory
        if model_file is None:
            parent_dir = os.path.dirname(__file__)
            model_file = os.path.join(parent_dir, 'custom_models', 'MEGNetModel__OFMEncoded_v1.h5')
        
        if not os.path.isfile(model_file):
            raise ValueError('MEGNetModel__OFMEncoded_v1.h5 not found in custom_models directory.')

        file_path_without_ext, _ = os.path.splitext(model_file)
        scaler_file = scaler_file or file_path_without_ext + "_scaler.pkl"
        n_targets = 188
        neuron_layers = (64, 128, 64)
        model_name = "OFMEncoded_v1"

    else:
        raise ValueError("model_type not recognized. Must be 'MatMinerEncoded_v1', 'OFMEncoded_v1', 'adjacent', or 'pretrained_models'.")
    
    # If model is not provided, load it from file
    if model is None:
        model, scaler = load_model_scaler(n_targets=n_targets, 
                                          neuron_layers=neuron_layers,
                                          model_file=model_file, scaler_file=scaler_file, 
                                          **kwargs)

    # Start extracting features
    MEGNetFeatsDF = [] 
    indexes = structures.index.to_list()
    structures_valid, ypred = megnet_evaluate_structures(model, structures)

    for original_index, s in zip(indexes, structures):
        if s in list(structures_valid):
            s_idx = list(structures_valid).index(s)
            p = ypred[s_idx]
            if scaler is None:
                feat_data = pd.DataFrame([p], 
                                         columns=[f"MEGNet_{model_name}_{idx + 1}" for idx in range(n_targets)],
                                         index=[original_index])
            else:
                feat_data = pd.DataFrame(scaler.inverse_transform(p.reshape(1, -1)),
                                         columns=[f"MEGNet_{model_name}_{idx + 1}" for idx in range(n_targets)],
                                         index=[original_index])
            struct = pd.DataFrame({'structure': [s]}, index=[original_index])
            modeldata_struct = pd.concat([struct, feat_data], axis=1)
        else:
            feat_data = pd.DataFrame([[np.nan] * n_targets],
                                     columns=[f"MEGNet_{model_name}_{idx + 1}" for idx in range(n_targets)],
                                     index=[original_index])
            struct = pd.DataFrame({'structure': [s]}, index=[original_index])
            modeldata_struct = pd.concat([struct, feat_data], axis=1)
        
        MEGNetFeatsDF.append(modeldata_struct)

    MEGNetFeatsDF = pd.concat(MEGNetFeatsDF, axis=0)  
    return MEGNetFeatsDF
    
def get_Adjacent_MEGNetFeatures(structures,
                               n_targets: int = 1,
                               neuron_layers: Tuple[int] = (64, 32, 16), 
                               model=None, 
                               model_file='MEGNetModel__adjacent.h5', 
                               scaler=None,
                               scaler_file='MEGNetModel__adjacent_scaler.pkl',
                               layer_name='layer32',
                               **kwargs):
    # Check if the model file exists, if not, issue error
    model_path = kwargs.get('model_path', '')
    model_file = os.path.join(model_path, model_file)
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"{model_file} not found. Please train the model first.")
    
    # get base folder if any from model_file
    base_folder = os.path.dirname(model_file)
    # set it on scaler_file
    scaler_file = os.path.join(base_folder, scaler_file)

    model_name = kwargs.pop('model_name', 'Adjacent')
    model, scaler = load_model_scaler(n_targets=n_targets, 
                                      model_file=model_file, scaler_file=scaler_file, 
                                      **kwargs)
    
    # Determine layer index based on layer_name
    layer_mapping = {
        'layer16': -2,  # Second-to-last layer (16 neurons)
        'layer32': -3   # Third-to-last layer (32 neurons)
    }
    
    if layer_name not in layer_mapping:
        raise ValueError("Invalid layer_name. Choose 'layer16' or 'layer32'.")
    
    layer_idx = layer_mapping[layer_name]
    
    # Create a model that outputs the specified intermediate layer
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer_idx].output)
    
    MEGNetAdjacent_structs = []   
    indexes = structures.index.to_list()
    for original_index, s in zip(indexes, structures):
        try:
            graph = model.graph_converter.convert(s)
            inp = model.graph_converter.graph_to_input(graph)
            pred = intermediate_layer_model.predict(inp, verbose=False)
            s_model_results = pd.DataFrame([pred[0][0]], 
                                           columns=[f"{model_name}_{idx + 1}" for idx in 
                                                    range(len(pred[0][0]))],
                                           index=[original_index])
            MEGNetAdjacent_structs.append(s_model_results)
        except Exception as e:
            print(e)
            print("Probably an invalid structure was passed to the model, continuing..")
            # Create a DataFrame with NaN values in case of failure
            nans = [[np.nan] * len(pred[0][0])]
            columns = [f"{model_name}_{idx + 1}" for idx in range(len(pred[0][0]))]
            s_model_results = pd.DataFrame(nans, columns=columns, index=[original_index])
            MEGNetAdjacent_structs.append(s_model_results)
            continue
    
    # Concatenate all structures' features into a final DataFrame
    MEGNetAdjacentFeatsDF = pd.concat(MEGNetAdjacent_structs, axis=0)
    print(f"Features calculated for model {model_name}.")
    return MEGNetAdjacentFeatsDF


__all__ = ['model_setup', 'load_model_scaler', 'megnet_evaluate_structures',
               'get_MVL_MEGNetFeatures', 'get_Custom_MEGNetFeatures', 'get_Adjacent_MEGNetFeatures'
           ]