import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

# Randomly select initial centroids from the dataset.
def initialize_centroids(X, K):
    # Randomly choose K indices from the number of samples
    centroids_idx = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[centroids_idx]
    return centroids

# Finds the closest centroid for each sample
def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    N = X.shape[0]
    idx = np.zeros(N, dtype=int)
    for i in range(N):
        d_min = np.linalg.norm(X[i] - centroids[0])**2
        c_i = 0
        for j in range(1, K):
            d_cur = np.linalg.norm(X[i] - centroids[j])**2
            if d_cur < d_min:
                d_min = d_cur
                c_i = j
        idx[i] = c_i
    return idx

# Compute the mean of samples assigned to each centroid
def compute_centroids(X, idx, K):
    D = X.shape[1]
    centroids = np.zeros((K, D))
    for j in range(K):
        indices = np.where(idx == j)[0]
        if len(indices) > 0:
            centroids[j] = np.mean(X[indices], axis=0)
    return centroids

# K-means algorithm for a specified number of iterations
def run_kmeans(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Initialize centroids randomly
def kmeans_init_centroids(X, K):
    ### YOUR CODE HERE

    initial_centroids =initialize_centroids(X, K)
    return initial_centroids

# Load the image
image = io.imread(r'D:\ChromeDownloads\PatReco2024HW2\exercise2_4\Fruit.png')
# image = resize(image, (256, 256))  # Resize for faster processing, if needed
img_size = image.shape

# Normalize image values in the range 0 - 1
image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Reshape the image to be a NxD matrix (N = num of pixels, D = num of features per pixel)
X = image.reshape(-1, 3)  # Reshape to N x D

# Perform K-means clustering
K = 16
max_iters = 10

# Initialize the centroids randomly
initial_centroids = kmeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kmeans(X, initial_centroids, max_iters)

# K-Means Image Compression
print('\nApplying K-Means to compress an image.\n')

# Find the closest cluster members
idx = find_closest_centroids(X, centroids)

# Recover the image from the indices
X_recovered = centroids[idx].reshape(img_size)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()
