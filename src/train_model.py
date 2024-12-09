import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def train_model(model,X_train,Y_train,X_test,Y_test,batch_size=50,epochs=100):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0]).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.shuffle(buffer_size=X_test.shape[0]).batch(batch_size)

    history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

    df_loss_acc = pd.DataFrame(history.history)
    df_loss = df_loss_acc[['loss', 'val_loss']].copy()
    df_loss.columns = ['train', 'validation']  
    df_acc = df_loss_acc[['accuracy', 'val_accuracy']].copy()
    df_acc.columns = ['train', 'validation']  

    df_loss.plot(title='Model Loss', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Loss')
    df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')

    plt.show()

    return history

