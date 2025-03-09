import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import io
from skimage.color import rgb2lab, lab2rgb  # Import lab2rgb directly
import cv2
from PIL import Image

def segment_image_with_kmeans(image_path, k_range=(2, 10), random_state=42):
    """
    Segment an image using K-means clustering with automatic K selection based on silhouette score.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    k_range : tuple
        Range of K values to try (min, max)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    segmented_image : numpy.ndarray
        The segmented image where each pixel is replaced with its cluster center
    best_k : int
        The optimal number of clusters
    silhouette_scores : dict
        Dictionary of silhouette scores for each K value
    """
    # Read the image
    img = io.imread(image_path)
    
    # Reshape the image to a 2D array of pixels
    height, width, channels = img.shape
    pixels = img.reshape(-1, channels)
    
    # Convert to LAB color space (better for color segmentation)
    pixels_lab = rgb2lab(pixels.reshape(-1, 3).astype(float) / 255.0)
    
    # Try different K values and compute silhouette scores
    silhouette_scores = {}
    labels_dict = {}
    kmeans_dict = {}
    
    for k in range(k_range[0], k_range[1] + 1):
        print(f"Trying K = {k}...")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(pixels_lab)
        
        # Calculate silhouette score if k > 1
        if k > 1:
            # Use a subset of pixels for silhouette calculation if the image is large
            sample_size = min(10000, len(pixels_lab))
            indices = np.random.choice(len(pixels_lab), sample_size, replace=False)
            score = silhouette_score(pixels_lab[indices], labels[indices])
            silhouette_scores[k] = score
            print(f"K = {k}, Silhouette Score = {score:.4f}")
        else:
            silhouette_scores[k] = -1  # Invalid score for k=1
            
        labels_dict[k] = labels
        kmeans_dict[k] = kmeans
    
    # Find the K with the highest silhouette score
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best K = {best_k} with Silhouette Score = {silhouette_scores[best_k]:.4f}")
    
    # Get the labels and cluster centers for the best K
    labels = labels_dict[best_k]
    kmeans = kmeans_dict[best_k]
    
    # Convert cluster centers back to RGB
    centers = kmeans.cluster_centers_
    centers_rgb = (np.clip(lab2rgb(centers.reshape(-1, 1, 3)), 0, 1) * 255).astype(np.uint8)  # Fixed line
    centers_rgb = centers_rgb.reshape(-1, 3)
    
    # Create segmented image by replacing each pixel with its corresponding cluster center
    segmented_pixels = centers_rgb[labels]
    segmented_image = segmented_pixels.reshape(height, width, channels)
    
    return segmented_image, best_k, silhouette_scores

def plot_results(original_image, segmented_image, silhouette_scores, best_k):
    """
    Plot the original image, segmented image, and silhouette scores.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot segmented image
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title(f'Segmented Image (K = {best_k})')
    plt.axis('off')
    
    # Plot silhouette scores
    plt.subplot(1, 3, 3)
    k_values = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    
    # Remove invalid score for k=1 if present
    if silhouette_scores.get(1, 0) == -1:
        k_values = k_values[1:]
        scores = scores[1:]
    
    plt.plot(k_values, scores, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. K')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the image segmentation process.
    """
    # Replace with your image path
    image_path = "flood_75.jpg"  
    
    # Read the original image for display
    original_image = io.imread(image_path)
    
    # Segment the image
    segmented_image, best_k, silhouette_scores = segment_image_with_kmeans(
        image_path=image_path,
        k_range=(2, 10),  # Try K from 2 to 10
        random_state=42
    )
    
    # Plot results
    plot_results(original_image, segmented_image, silhouette_scores, best_k)
    
    # Save the segmented image
    output_path = f"segmented_k{best_k}.jpg"
    io.imsave(output_path, segmented_image)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    main()