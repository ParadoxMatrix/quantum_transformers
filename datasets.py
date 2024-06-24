import os
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

options = tf.data.Options()
options.deterministic = True

class ImageFolderDataset(tfds.core.GeneratorBasedBuilder):
    """
    A dataset consisting of images stored in folders (one folder per class).
    """
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, name, img_shape, num_classes, data_dir, **kwargs):
        """Creates an ImageFolderDataset."""
        self.name = name
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.data_dir = data_dir
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=self.img_shape),
                'label': tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            supervised_keys=('image', 'label')
        )

    def _split_generators(self, _):
        """Returns SplitGenerators."""
        dataset_path = tfds.core.Path(self.data_dir)
        return {
            'train': self._generate_examples(dataset_path / 'train'),
            'test': self._generate_examples(dataset_path / 'test'),
            'valid': self._generate_examples(dataset_path / 'valid'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        class_names = {c: i for i, c in enumerate(sorted([f.name for f in path.glob('*')]))}
        for class_folder in path.glob('*'):
            for f in class_folder.glob('*.jpg'):  # Assuming images are in jpg format
                try:
                    image = tf.io.read_file(f)
                    image = tf.image.decode_jpeg(image, channels=3)
                    image = tf.image.resize(image, self.img_shape[:2])
                    image = image / 255.0  # Normalize to [0, 1]
                    yield f"{class_folder.name}_{f.name}", {
                        'image': image.numpy(),
                        'label': class_names[class_folder.name],
                    }
                except Exception as e:
                    print(e)

def datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, drop_remainder=True, transform=None):
    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(10_000, reshuffle_each_iteration=True)

    # Batch
    train_dataset = train_dataset.batch(batch_size, drop_remainder=drop_remainder)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=drop_remainder)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Transform
    if transform is not None:
        train_dataset = train_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.map(transform, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # Convert to NumPy for JAX
    return tfds.as_numpy(train_dataset), tfds.as_numpy(val_dataset), tfds.as_numpy(test_dataset)

def get_real_fake_faces_dataloaders(data_dir: str, batch_size: int = 64, drop_remainder: bool = True):
    """
    Returns dataloaders for the Real and Fake Faces dataset
    """
    data_dir = os.path.expanduser(data_dir)

    # Load datasets
    real_fake_faces_builder = ImageFolderDataset(name="real_fake_faces", img_shape=(224, 224, 3), num_classes=2, data_dir=data_dir)
    real_fake_faces_builder.download_and_prepare(download_dir=data_dir)
    train_dataset, val_dataset, test_dataset = real_fake_faces_builder.as_dataset(split=['train', 'valid', 'test'], as_supervised=True, shuffle_files=True)
    train_dataset, val_dataset, test_dataset = train_dataset.with_options(options), val_dataset.with_options(options), test_dataset.with_options(options)
    print("Cardinalities (train, val, test):", train_dataset.cardinality().numpy(), val_dataset.cardinality().numpy(), test_dataset.cardinality().numpy())

    def preprocess(image, label):
        return image, label

    return datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size,
                                   drop_remainder=drop_remainder, transform=preprocess)
