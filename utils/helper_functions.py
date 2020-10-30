import numpy as np
import torch

"""
Converts input RGB image to grayscale.
@images: list of 4D tensor images 
@:return list of 4D tensor images with single channel 
"""
def convert2grayscale(images: list):
    assert len(images) >= 0, 'there should be at least 1 image batch'

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscaleImages = []
    for image_batch in images:
        assert len(image_batch.shape) == 4, 'valid image is batch x channel x height x width'
        assert image_batch.shape[1] == 3, 'valid image should have 3 channel'
        desired_shape = list(image_batch.shape)
        desired_shape[1] = 1
        result = np.zeros(desired_shape)
        for i in range(image_batch.shape[0]):
            image = image_batch[i].squeeze().detach().numpy().transpose(1, 2, 0)
            image = np.dot(image[..., :3], rgb_weights)
            result[i, :] = image
        result = torch.from_numpy(result)
        grayscaleImages.append(result)

    return grayscaleImages
