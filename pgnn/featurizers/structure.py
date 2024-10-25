from .megnet_models import ( get_MVL_MEGNetFeatures, get_Custom_MEGNetFeatures, 
                           train_MEGNet_on_the_fly, get_Adjacent_MEGNetFeatures )

class LatentMMFeaturizer:
     """Latent space featurizer for MatMiner features used in OMEGAfast."""

     def get_features(self, structures):
          # Logic for latent space MatMiner feature extraction
          return get_Custom_MEGNetFeatures(structures, model_type='MatMinerEncoded_v1')

l_MM_v1 = LatentMMFeaturizer()

class LatentOFMFeaturizer:
     """Latent space featurizer for OFM features."""

     def get_features(self, structures):
          # Logic for latent space OFM feature extraction
          return get_Custom_MEGNetFeatures(structures, model_type='OFMEncoded_v1')

l_OFM_v1 = LatentOFMFeaturizer()

class MVLFeaturizer:
    """Pretrained MEGNet models for various properties."""

    def __init__(self, layer_name='layer32'):
        self.layer_name = layer_name

    def get_features(self, structures):
        return get_MVL_MEGNetFeatures(structures, layer_name=self.layer_name)

# Create instances for both 'layer32' and 'layer16'
mvl32 = MVLFeaturizer(layer_name='layer32')
mvl16 = MVLFeaturizer(layer_name='layer16')

class AdjacentMEGNetFeaturizer:
    """Adjacent MEGNet model that trains on the fly."""

    def __init__(self, layer_name='layer32'):
        self.layer_name = layer_name

    def train_adjacent_megnet(self, structures, targets, **kwargs):
        train_MEGNet_on_the_fly(structures, targets, **kwargs)

    def get_features(self, structures, **kwargs):
        return get_Adjacent_MEGNetFeatures(structures, layer_name=self.layer_name, **kwargs)

# Create instances for both 'layer32' and 'layer16'
adj_megnet = AdjacentMEGNetFeaturizer(layer_name='layer32')
adj_megnet_layer16 = AdjacentMEGNetFeaturizer(layer_name='layer16')

__all__ = ( "l_MM_v1", "l_OFM_v1", "mvl32", "mvl16", "adj_megnet", "adj_megnet_layer16")