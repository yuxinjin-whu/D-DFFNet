import numpy as np
import imageio
import matplotlib.pyplot as plt

def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().detach().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, 128):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        imageio.imsave('/data/jyx/dbd/CPD/testSideFeature/rfb1/'+str(index)+".png", feature_map[index-1])
    plt.show()