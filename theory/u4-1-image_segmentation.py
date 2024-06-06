from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt

image =imread("image2.jpg")
print(image.shape)
print(image.reshape(-1,3))
plt.imshow(image)
plt.show()

X=image.reshape(-1,3)
kmeans = KMeans(n_clusters=10).fit(X)
print(kmeans.labels_)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
print(segmented_img)
segmented_img = segmented_img.reshape(image.shape)
print(segmented_img)
plt.imshow(segmented_img.astype('uint8'))
plt.show()
# plt.imshow(segmented_img)


kmeans = KMeans(n_clusters=8).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
print(segmented_img)
segmented_img = segmented_img.reshape(image.shape)
print(segmented_img)
plt.imshow(segmented_img.astype('uint8'))
plt.show()

kmeans = KMeans(n_clusters=6).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
print(segmented_img)
segmented_img = segmented_img.reshape(image.shape)
print(segmented_img)
plt.imshow(segmented_img.astype('uint8'))
plt.show()

kmeans = KMeans(n_clusters=4).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
print(segmented_img)
segmented_img = segmented_img.reshape(image.shape)
print(segmented_img)
plt.imshow(segmented_img.astype('uint8'))
plt.show()

kmeans = KMeans(n_clusters=2).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
print(segmented_img)
segmented_img = segmented_img.reshape(image.shape)
print(segmented_img)
plt.imshow(segmented_img.astype('uint8'))
plt.show()

