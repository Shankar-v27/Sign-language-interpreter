import tensorflow as tf
import os

# Parameters
image_size = (128, 128)
batch_size = 32
epochs = 10
data_dir = "emotions_data/"

# === Load and preprocess the dataset ===
def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label

def get_dataset(data_dir):
    class_names = sorted(os.listdir(data_dir))
    class_indices = dict((name, index) for index, name in enumerate(class_names))

    file_paths = []
    labels = []

    for label_name in class_names:
        label_dir = os.path.join(data_dir, label_name)
        for fname in os.listdir(label_dir):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                file_paths.append(os.path.join(label_dir, fname))
                labels.append(class_indices[label_name])

    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(class_names)

# Load dataset
dataset, num_classes = get_dataset(data_dir)
val_size = int(0.2 * len(dataset))
val_ds = dataset.take(val_size)
train_ds = dataset.skip(val_size)

# === Load existing model (pure TF) ===
model = tf.saved_model.load("saved_model/")
infer = model.signatures["serving_default"]

# Wrap infer in a function to use in training (if possible)
base_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
    tf.keras.layers.Lambda(lambda x: infer(x)["output_0"])  # Adjust based on your model's outputs
])

# === Add new layers (pure TF-style using keras.layers as building blocks) ===
full_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# === Optimizer, loss, metrics ===
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

# === Training loop ===
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            logits = full_model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, full_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, full_model.trainable_variables))
        train_acc.update_state(y_batch, logits)

    print(f"Train Accuracy: {train_acc.result().numpy():.4f}")
    train_acc.reset_states()

    # Validation
    for x_val, y_val in val_ds:
        val_logits = full_model(x_val, training=False)
        val_acc.update_state(y_val, val_logits)
    print(f"Val Accuracy: {val_acc.result().numpy():.4f}")
    val_acc.reset_states()

# === Save final model ===
tf.saved_model.save(full_model, "fine_tuned_emotion_model_tf_only/")
print("✅ Fine-tuned model saved.")
