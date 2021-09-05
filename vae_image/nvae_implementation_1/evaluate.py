import os
import numpy as np
from tqdm import tqdm
import skimage.transform
import scipy.linalg as scalg

import tensorflow as tf
from tensorflow_probability import distributions

from model import NVAE
import precision_recall as prec_rec
import perceptual_path_length as ppl
from fid_utils import calculate_fid_given_paths
from util import Metric, Metrics, ModelEvaluation, sample_to_dir, save_images_to_dir, tile_images



def evaluate_model(epoch, model, test_data, metrics_logger, batch_size, n_attempts=10, binary=False) -> ModelEvaluation:
    # PPL
    # slerp, slerp_perturbed = e.perceptual_path_length_init()
    # images1, images2 = model.sample(z=slerp), model.sample(z=slerp_perturbed)
    # TODO: Handle entire dataset
    # test_samples, _ = next(test_data.as_numpy_iterator())
    # test_samples = tf.convert_to_tensor(test_samples)
    evaluation = ModelEvaluation(nll=None, sample_metrics=[])
    
    for temperature in tqdm([0.6, 0.8, 1.0], desc="Temperature based tests (PPL/PR)", total=4):
        # TODO: Handle batches, perform 1000 attempts and average
        precisions = []
        recalls = []
        ppls = []
        for attempt in tqdm(range(n_attempts), desc="Sample attempt (PPL/PR)"):
            generated_images, last_s, z1, z2 = model.sample(temperature=temperature, n_samples=batch_size)
            precision, recall = 0, 0
            for test_batch, _ in test_data:
                for microbatch in tf.split(test_batch, 2):
                    pr_images, *_ = model.sample(temperature=temperature, n_samples=tf.shape(microbatch)[0])
                    
                    # PR
                    batch_precision, batch_recall = precision_recall(pr_images, microbatch)
                    precision += batch_precision.item()
                    recall += batch_recall.item()
                    
            # PPL
            slerp, slerp_perturbed = perceptual_path_length_init(z1, z2)
            images1, images2 = (model.sample_with_z(slerp, last_s), model.sample_with_z(slerp_perturbed, last_s))
            ppl = tf.reduce_mean(perceptual_path_length(images1, images2))
            
            ppls.append(ppl)
            precisions.append(precision / len(test_data))
            recalls.append(recall / len(test_data))
            
        fid = evaluate_fid(model, test_data, "mnist", batch_size=batch_size, temperature=temperature, binary=binary)
        
        evaluation.sample_metrics.append(Metrics(temperature=temperature, fid=fid, ppl=Metric.from_list(ppls), 
                                                 precision=Metric.from_list(precisions), recall=Metric.from_list(recalls)))
        
    # Negative log-likelihood
    evaluation.nll = neg_log_likelihood(model, test_data, n_attempts=n_attempts)
    return evaluation


def neg_log_likelihood(model: NVAE, test_data: tf.data.Dataset, n_attempts=10):
    nlls = []
    for batch, _ in tqdm(test_data, desc="NLL Batch", total=len(test_data)):
        batch_logs = []
        for _ in range(n_attempts):
            reconstruction, _, log_p, log_q = model(batch, nll=True)
            log_iw = -model.calculate_recon_loss(batch, reconstruction, crop_output=True) - log_q + log_p
            batch_logs.append(log_iw)
        nll = -tf.math.reduce_mean(tf.math.reduce_logsumexp(tf.stack(batch_logs), axis=0) - tf.math.log(float(n_attempts)))
        nlls.append(nll)
    return Metric.from_list(nlls)


# Takes 2 batches of images (b_size x 299 x 299 x 3) from different sources and calculates a FID score.
# Lower scores indicate closer resemblance in generated material to another data source.
# Inspired from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# TODO: support for batch size=1 (?) and progress logging
def fid_score(images1, images2):
    act1, act2 = latent_activations(images1, images2, "IV3")
    act1 = act1.numpy()
    act2 = act2.numpy()
    # model activations as gaussians
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate distance
    dotp = sigma1.dot(sigma2)
    covmean = scalg.sqrtm(dotp)
    return np.sum((mu1 - mu2) ** 2.0) + np.trace(sigma1 + sigma2 - 2.0 * (covmean.real))


def evaluate_fid(model: NVAE, dataset, dataset_name, batch_size, temperature, binary=False):
    dataset_dir = os.path.join("images", dataset_name, "actual")
    output_dir = os.path.join("images", dataset_name, f"generated_t_{temperature}")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if not os.listdir(dataset_dir):
        # We need to save the source images to the directory
        for image_batch, _ in tqdm(dataset, desc="Saving dataset (FID)"):
            save_images_to_dir(image_batch, dataset_dir)
    for filename in os.listdir(output_dir):
        # Delete all old generated images
        os.remove(os.path.join(output_dir, filename))
    # Recommended by FID author
    sample_size = 10000
    sample_to_dir(model, batch_size, sample_size, temperature, output_dir)
    os.makedirs("fid", exist_ok=True)
    
    print("[FID] Calculating FID")
    fid_value = calculate_fid_given_paths([dataset_dir, output_dir], inception_path="fid")
    return fid_value


# Evaluates the PR of images1 in reference to images2 using NVIDIAs implementation.
def precision_recall(images1, images2):
    act1, act2 = latent_activations(images1, images2, "VGG")
    # tf.compat.v1.disable_eager_execution()
    # act1 = tf.reshape(act1, (tf.shape(act1)[0], -1))
    # act2 = tf.reshape(act1, (tf.shape(act2)[0], -1))
    pr = prec_rec.knn_precision_recall_features(act1, act2)
    # tf.compat.v1.enable_eager_execution()
    return pr["precision"], pr["recall"]


# Calculates slerp from sampled latents. To continue PPL, generate images from the result of this function
# and call perceptual_path_length(images1,images2).
def perceptual_path_length_init(z1, z2, epsilon=1e-4):
    t = tf.random.uniform([tf.shape(z1)[0]], 0.0, 1.0)
    return ppl.slerp(z1, z2, t), ppl.slerp(z1, z2, t + epsilon)


# Takes generated images from interpolated latents and gives the PPL.
def perceptual_path_length(images1, images2):
    act1, act2 = latent_activations(images1, images2, "VGG")
    return ppl.evaluate(act1, act2)


# For comparing generated and real samples via Inception v3 latent representation.
# Returns latent activations from 2 sets of image batches.
def latent_activations(images1, images2, model_name):
    if not (images1.shape[1:] == (299, 299, 3) and images2.shape[1:] == (299, 299, 3)):
        images1, images2 = resize(images1), resize(images2)

    act1, act2 = 0, 0

    if model_name == "IV3":
        # TODO: Use
        model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(299, 299, 3),
        )
        act1 = tf.convert_to_tensor(model.predict(images1,), dtype=tf.float32)
        act2 = tf.convert_to_tensor(model.predict(images2), dtype=tf.float32)
        
    elif model_name == "VGG":
        model = tf.keras.applications.VGG16(include_top=False, pooling="avg")
        act1 = model(images1)
        act2 = model(images2)

    # latent representation

    return act1, act2


def gen_images(b_size, s1, s2, m1, m2):
    im1 = tf.random.normal(shape=[b_size, 32, 32, 3], stddev=s1, mean=m1, dtype=tf.dtypes.float32)
    im2 = tf.random.normal(shape=[b_size, 32, 32, 3], stddev=s2, mean=m2, dtype=tf.dtypes.float32)
    return im1, im2


def resize(images, target_shape=(299, 299, 3)):
    if tf.shape(images)[-1] == 1:
        images = tf.image.grayscale_to_rgb(images)
    resized_images = []
    for img in images:
        resized_images.append(skimage.transform.resize(img, target_shape, 0))
    return tf.convert_to_tensor(resized_images, dtype=tf.float32)


# -------For standalone debugging------


def main():
    # a,b=gen_images(20,0.1,0.1,0,0)
    # print(a.shape)
    # p,r=precision_recall(a,b)
    # print(str(p) + " - " + str(r))

    a, b = gen_images(20, 3, 3, 0, 0)
    print(a.shape)
    p, r = precision_recall(a, b)
    print(str(p) + " - " + str(r))


if __name__ == "__main__":
    main()
# -------For standalone debugging------

