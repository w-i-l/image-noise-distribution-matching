import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class SiameseModel:
    def __init__(
        self,
        img_shape=(256, 256, 1),
        embedding_dim=192,
        base_filters=32,
        layers_filters_multiplier=[1, 2, 4, 8],
        dropout=0.25,
        l2_reg=0.00005
    ):
        self.img_shape = img_shape
        self.embedding_dim = embedding_dim
        self.base_filters = base_filters
        self.layers_filters_multiplier = layers_filters_multiplier
        self.dropout = dropout
        self.l2_reg = l2_reg
        
    
    def build_model(self) -> Model:
        """
        Build the Siamese triplet model.
        """
        
        base_network = self._build_base_model()
    
        anchor_input = layers.Input(shape=self.img_shape, name='anchor')
        positive_input = layers.Input(shape=self.img_shape, name='positive')
        negative_input = layers.Input(shape=self.img_shape, name='negative')
        
        anchor_embed = base_network(anchor_input)
        positive_embed = base_network(positive_input)
        negative_embed = base_network(negative_input)
        
        outputs = layers.Concatenate()([
            anchor_embed,
            positive_embed,
            negative_embed
        ])
        
        triplet_model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=outputs,
            name='triplet_model'
        )
        
        print(f"Encoder model built with: {base_network.count_params():,} parameters")
        
        return base_network, triplet_model
    
        
    def _build_base_model(self) -> Model:
        """
        Build the base CNN model to extract embeddings from input images.
        """
        
        inputs = layers.Input(shape=self.img_shape, name='input')
    
        # block 1
        block1_multiplier = self.layers_filters_multiplier[0]
        x = layers.Conv2D(
            self.base_filters * block1_multiplier, 
            (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # block 2
        block2_multiplier = self.layers_filters_multiplier[1]
        x = layers.Conv2D(
            self.base_filters * block2_multiplier,
            (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout * 0.5)(x)
        
        # block 3
        block3_multiplier = self.layers_filters_multiplier[2]
        x = layers.Conv2D(
            self.base_filters * block3_multiplier,
            (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(self.dropout * 0.7)(x)
        
        # block 4
        block4_multiplier = self.layers_filters_multiplier[3]
        x = layers.Conv2D(
            self.base_filters * block4_multiplier,
            (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # embeddings
        x = layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)
        
        x = layers.Dense(self.embedding_dim, activation=None)(x)
        embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
        
        base_network = Model(inputs, embeddings, name='base_network')
        return base_network
