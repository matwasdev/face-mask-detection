import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from src.models.mask_detector import MaskDetector
from src.utils.data_loader import get_dataset, change_class_occurrences, prepare_for_training


images, labels = get_dataset()
images, labels = change_class_occurrences(images, labels, max_occurrences=1200)
images, labels = prepare_for_training(images, labels)

X_train, X_test, y_train, y_test = train_test_split(images,labels, test_size=0.1, random_state=42)


model = MaskDetector().get_model()

augmented_data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

generator = augmented_data_gen.flow(X_train, y_train, batch_size=32)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(generator, epochs=20, validation_data=(X_test, y_test))


shouldSave = input("Do you want to save the model? YES/NO")

if shouldSave == "YES":
    model.save("../src/models/mask_detection_model.keras")




