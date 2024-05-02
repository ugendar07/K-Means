from model import KMeans
from utils import get_image, show_image, save_image, error
from matplotlib import pyplot as plt


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # create model
    # num_clusters = 2 # CHANGE THIS
    # kmeans = KMeans(num_clusters)
    cluster_size = [2,5,10,20,50]
    mse = []
    for i in cluster_size:
        kmeans = KMeans(i)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        image_clustered = kmeans.replace_with_cluster_centers(image)

        # Print the error
        print('MSE:', error(image, image_clustered))
        mse.append(error(image,image_clustered))


        # reshape image
        image_clustered = image_clustered.reshape(img_shape)

    
        # show/save image
        # show_image(image)
        save_image(image_clustered, f'image_clustered_{i}.jpg')



    plt.plot(cluster_size,mse)
    plt.xlabel('num of cluster')
    plt.ylabel('mse')
    plt.title('k vs mse')
    plt.show()

if __name__ == '__main__':
    main()
