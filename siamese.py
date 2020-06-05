import tensorflow as tf 
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Input, Flatten, LSTM, Lambda
from tensorflow.keras import backend as K

class Models():
    def __init__(self, input_shape=(28, 28, 3)):
        self.input_shape = input_shape
        self.left_input = Input(input_shape)
        self.right_input = Input(input_shape) 

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (7, 7), strides=(1, 1), activation='relu', padding='valid', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(126, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(1028, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())

        model.add(Dense(265, activation='relu'))
        model.add(BatchNormalization())
        
        encoder_1 = model(self.left_input)
        encoder_2 = model(self.right_input)
        
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoder_1, encoder_2])
        
        prediction = Dense(2, activation='softmax')(L1_distance)
        
        siamese_net = Model(inputs=[self.left_input, self.right_input], outputs=prediction)
        
        self.model = siamese_net

    def summary(self):
        return self.model.summary()

    def compile(self):
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer = optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, left_input, right_input, y_train, epochs):
        self.model.fit([left_input[:1000], right_input[:1000]], y_train[:1000], validation_data=([left_input[1000:], right_input[1000:]], y_train[1000:]),epochs=epochs, verbose=True, batch_size=32)

    def save(self):
        self.model.save('save/model.h5')

if __name__ == "__main__":
    model = Models()
    model.create_model()
    model.summary()
    model.compile()