import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from helper_functions2 import get_imgs, load_caps_img, create_image_gen


def load_mnist(batch_size, binary=True):
    train_ds, test_ds = tfds.load("mnist", split=["train", "test"], shuffle_files=True, batch_size=batch_size, as_supervised=True)

    def transform(image, label):
        image = tf.image.resize_with_crop_or_pad(image, 32, 32)
        image = tf.cast(image, dtype=tf.float32)
        if binary:
            image = tfp.distributions.Bernoulli(probs=image, dtype=tf.float32).sample()
        else:
            image /= 255
        return image, label

    return train_ds.map(transform), test_ds.map(transform)


def load_coco(use_all_data, image_shape, batch_size, drop_remain):
    dataset_name = 'train'
    train_capspath, train_imgspath = get_imgs(dataset_name)
    train_caps, train_imgs = load_caps_img(train_capspath, train_imgspath, dataset_name, use_all_data)
    
    dataset_name = 'val'
    val_capspath, val_imgspath = get_imgs(dataset_name)
    val_caps, val_imgs = load_caps_img(val_capspath, val_imgspath, dataset_name, use_all_data)

    dataset_name = 'test'
    test_capspath, test_imgspath = get_imgs(dataset_name)
    test_imgs = load_caps_img(test_capspath, test_imgspath, dataset_name, use_all_data)

    train_dataset = create_image_gen(set(train_imgs), batch_size, image_shape, drop_remain)
    val_dataset = create_image_gen(set(val_imgs), batch_size, image_shape, drop_remain)
    test_dataset = create_image_gen(set(test_imgs), batch_size, image_shape, drop_remain)
    
    return train_dataset, val_dataset, test_dataset, len(set(train_imgs))


def load_celeba():
    # tfds.load('celeb_a')
    pass


if __name__ == "__main__":
    load_celeba()
