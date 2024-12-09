import keras

def convolution_model(input_shape,categories):
   
    input_image=keras.Input(shape=input_shape)

    mobile_net= keras.applications.MobileNet(weights='imagenet',input_tensor=input_image,include_top=False)
    mobile_net_output=mobile_net.output

    mobile_netV2=keras.applications.MobileNetV2(weights='imagenet',input_tensor=input_image,include_top=False)
    mobile_netV2_output=mobile_netV2.output

    mobile_net_flat=keras.layers.GlobalAveragePooling2D()(mobile_net_output)
    mobile_netV2_flat=keras.layers.GlobalAveragePooling2D()(mobile_netV2_output)

    fused_features=keras.layers.Concatenate()([mobile_net_flat,mobile_netV2_flat])
    
    F_dense=keras.layers.Dense(512,activation="relu")(fused_features)
    F_dropout=keras.layers.Dropout(rate=.2)(F_dense)
    F_batchnorm=keras.layers.BatchNormalization()(F_dropout)
    F_final=keras.layers.Dense(len(categories))(F_batchnorm)
    F_softmax=keras.layers.Softmax()(F_final)
    model=keras.Model(inputs=input_image,outputs=F_softmax)
    return model