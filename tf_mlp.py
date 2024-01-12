import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


label_map = {
    0: "bike",
    1: "cabinet",
    2: "chair",
    3: "coffee maker",
    4: "fan",
    5: "kettle",
    6: "lamp",
    7: "mug",
    8: "sofa",
    9: "stapler",
    10: "table",
    11: "toaster"
}


# physical_devices = tf.config.list_physical_devices('GPU')
# print("num physical devices:", len(physical_devices))


# Make predictions with the model on an unshuffled test dataset
def generate_csv(model, test_data, file_name):
    predictions = model.predict(test_data)
    y_pred = tf.argmax(predictions, axis=-1)

    df = pd.DataFrame({
        "Index": np.arange(10800),  # This creates a list of integers from 0 to the lenght of test_data (number of pics)
        "Label": y_pred
    })
    df.to_csv(file_name, index=False)


# Load data
with np.load("beginner_data.npz") as data:
    X_train = data['train_images']  # Shape (56259, 64, 64, 3)
    y_train = data['train_labels']  # Shape (56259,)
    X_test = data['test_images']    # Shape (10800, 64, 64, 3)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train[:50000], y_train[:50000]))
val_dataset = tf.data.Dataset.from_tensor_slices((X_train[50000:], y_train[50000:]))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test))


# Plot some pictures just for fun (doens't work on wsl with the GPU)
# fig, axs = plt.subplots(12, 4, figsize=(7, 18))

# for label in range(12):
#     for col in range(4):
#         i_train = np.argwhere(y_train == label)[col][0]
#         axs[label, col].imshow(X_train[i_train])
#         axs[label, col].set_title(label_map[label])
# for ax in axs.flatten():
#     ax.axis("off")
# plt.tight_layout()
# plt.show()


# -------------------------------DATA PREPROCESSING----------------------------------------
def preprocess(x, y):
    """
    Changes the data from uint8 to float [0, 1]
    a tf dataset should have x as the image (data) and y as the label
    """
    x = x/255
    return (x, y)

# Scale to [0, 1]
train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
test_dataset = test_dataset.map(lambda x: x/255)


# ---------------------------------DEFINING NEURAL NETWORK MODEL--------------------------------
# This one is a Multi-Layer-Perceptron (mlp) (the one you always see on beginner intro videos)
# A sequential model is basically we just do the defined stuff step by step
mlp_model = keras.models.Sequential([
    # Another "preprocessing" step: We use batch normalization to help us get the values noramlized in std deviation way
    tf.keras.layers.BatchNormalization(),
    # First, we gotta flatten the image (your input is now a 64*64*3 vector, so there are that many circles in layer1)
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    # Next, let's create a neural network layer with 512 things (neurons, the little circles)
    # Dense layer is just you connect all the things
    # Relu function as the activation, it's the thing that goes from 0 when x<0 and x when x>0
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # Next, let's scale it down to just 128 options
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # Finally, let's scale it down one more time to 32
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    # Lastly, we output it to our 12 options
    # Note that we are not specifying a activation layer/function (since those go between the nuron layers, they are the lines)
    # This signals to keras that we are done, this is the last step
    # It will auto apply loss function and gradient descent and bias updates before running through the next one
    tf.keras.layers.Dense(12)
])

resnet_model = keras.Sequential([
    # tf.keras.layers.Conv2D()
])

# print(mlp_model.summary())

# -----------------------------------HYPERPARAMETER SETTINGS--------------------------------
SAVE_PATH = "./"
SEED = 129
tf.random.set_seed(SEED)
np.random.seed(SEED)

# optimizer (gradient decsent algorithm)
# loss (loss function: )
# metrics (list of functions used to measure how successful our machine is)
mlp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Includes the softmax, so no need to specify activation=softmax in last dense layer. logits means log the answer? or use sigmoid?
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

batch_size = 512
epochs = 100
# Batching the datasets
# And also shuffles the training data set
train_dataset = train_dataset.cache().shuffle(buffer_size=train_dataset.cardinality(), seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

callbacks = [
    # This is an early stopping policy, which stops the training after 3 epochs of no improvement in a row of the accuracy
    tf.keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=5),
    # This saves the best model we've seen somewhere in a file
    tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor="val_sparse_categorical_accuracy", save_best_only=True)
]

# ---------------------------------------TRAIN MODEL-------------------------------------
history = mlp_model.fit(
    train_dataset,
    validation_data=val_dataset,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks
)


# --------------------------------------GET RESULTS-------------------------------------
# retreive our best model since we saved it in a file
best_model = tf.keras.models.load_model(SAVE_PATH)
train_loss, train_accuracy = best_model.evaluate(train_dataset)
print(f"Train Loss: {train_loss:.3f}")
print(f"Train Accuracy: {train_accuracy*100:.2f}%")

val_loss, val_accuracy = best_model.evaluate(val_dataset)
print(f"Test Loss: {val_loss:.3f}")
print(f"Test Accuracy: {val_accuracy*100:.2f}%")

generate_csv(mlp_model, test_dataset, "linear_mlp_2.csv")