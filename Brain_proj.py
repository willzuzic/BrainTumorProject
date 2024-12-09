from src.data_preprocessing import load_dataset, one_hot
from src.model import convolution_model
from src.train_model import train_model

categories = ['notumor', 'glioma', 'meningioma', 'pituitary']
train_dir='images/Training'
test_dir='images/Testing'

X_train,Y_train=load_dataset(train_dir,categories)
X_test,Y_test=load_dataset(test_dir,categories)
Y_train=one_hot(Y_train,len(categories))
Y_test=one_hot(Y_test,len(categories))

model=convolution_model((224,224,3),categories)
model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
history=train_model(model,X_train,Y_train,X_test,Y_test)


