import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test  = (x_test.astype("float32") / 255.0)[..., np.newaxis]


#  Retina filters (ON/OFF)

on_center = np.array([[-1,-1,-1],
                      [-1, 8,-1],
                      [-1,-1,-1]], dtype=np.float32)
off_center = -on_center

retina_kernels = np.stack([on_center, off_center], axis=-1)
retina_kernels = np.expand_dims(retina_kernels, -2) 

# Thresholding function for ON/OFF
def threshold_retina(x, th_on=0.5, th_off=0.5):
    on = x[..., 0]
    off = x[..., 1]

    on_thr  = tf.where(on > th_on, on, tf.zeros_like(on))
    off_thr = tf.where(off > th_off, off, tf.zeros_like(off))

    return tf.stack([on_thr, off_thr], axis=-1)


# Edge filters

h_edge = np.array([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]], dtype=np.float32)
v_edge = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
d1_edge = np.array([[ 0, 1, 1],
                    [-1,0, 1],
                    [-1,-1,0]], dtype=np.float32)
d2_edge = np.array([[ 1, 1, 0],
                    [ 1, 0,-1],
                    [ 0,-1,-1]], dtype=np.float32)

edge_kernels = np.stack([h_edge, v_edge, d1_edge, d2_edge], axis=-1)  # (3,3,4)
edge_kernels = np.repeat(edge_kernels[:, :, np.newaxis, :], 2, axis=2) # (3,3,2,4)

inputs = layers.Input(shape=(28,28,1))

# Retina
retina = layers.Conv2D(2, (3,3), padding="same",
                       use_bias=False, trainable=False,
                       name="retina")(inputs)

# Apply threshold after retina
retina = layers.Lambda(lambda x: threshold_retina(x, th_on=0.5, th_off=0.5))(retina)
retina = layers.BatchNormalization()(retina)

# Edges
edges = layers.Conv2D(4, (3,3), padding="same",
                      use_bias=False, trainable=False,
                      name="edges")(retina)
edges = layers.BatchNormalization()(edges)

# Intersections
h  = layers.Lambda(lambda x: x[:,:,:,0:1])(edges)
v  = layers.Lambda(lambda x: x[:,:,:,1:2])(edges)
d1 = layers.Lambda(lambda x: x[:,:,:,2:3])(edges)
d2 = layers.Lambda(lambda x: x[:,:,:,3:4])(edges)

hv  = layers.Multiply()([h,v])    
hd1 = layers.Multiply()([h,d1])   
hd2 = layers.Multiply()([h,d2])   
vd1 = layers.Multiply()([v,d1])   
vd2 = layers.Multiply()([v,d2])   
d12 = layers.Multiply()([d1,d2])  

intersections = layers.Concatenate(name="intersections")([hv, hd1, hd2, vd1, vd2, d12])
intersections = layers.BatchNormalization()(intersections)

# Concatenate ALL features
all_features = layers.Concatenate()([retina, edges, intersections])

# Classifier
flat = layers.Flatten()(all_features)
x = layers.Dense(256, activation="relu")(flat)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Assign handcrafted weights
model.get_layer("retina").set_weights([retina_kernels])
model.get_layer("edges").set_weights([edge_kernels])

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#Train

history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=35,
                    batch_size=350)

#  Evaluate

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Confusion Matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Visualization Example (digit 4)

sample_idx = np.where(y_test==4)[0][0]  
sample_img = x_test[sample_idx:sample_idx+1]

# Forward pass to extract feature maps
extractor = models.Model(inputs, [retina, edges, intersections])
out_retina, out_edges, out_inters = extractor.predict(sample_img)

# Retina outputs with colorbar
fig, axes = plt.subplots(1, 3, figsize=(12,4))
im0 = axes[0].imshow(sample_img.squeeze(), cmap="gray")
axes[0].set_title("Original: 4")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(out_retina[0,:,:,0], cmap="seismic")
axes[1].set_title("ON-center (thr)")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(out_retina[0,:,:,1], cmap="seismic")
axes[2].set_title("OFF-center (thr)")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.show()

# Edges with colorbar
fig, axes = plt.subplots(1, 4, figsize=(16,4))
titles = ["Horizontal", "Vertical", "Diag 45°", "Diag 135°"]
for i in range(4):
    im = axes[i].imshow(out_edges[0,:,:,i], cmap="seismic")
    axes[i].set_title(titles[i])
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
plt.show()

# Intersections with colorbar
fig, axes = plt.subplots(1, 6, figsize=(20,4))
titles = ["hv (+)", "hd1 (L)", "hd2 (T)", "vd1", "vd2", "d12 (X)"]
for i in range(6):
    im = axes[i].imshow(out_inters[0,:,:,i], cmap="seismic")
    axes[i].set_title(titles[i])
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
plt.show()
