import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import io, color
from skimage.color import rgb2lab, lab2rgb
import cv2
from PIL import Image

def segment_image_with_kmeans(image_path, k_range=(2, 10), use_grayscale=True, random_state=42):
    """
    Segment an image using K-means clustering with automatic K selection based on silhouette score.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    k_range : tuple
        Range of K values to try (min, max)
    use_grayscale : bool
        Whether to convert the image to grayscale before segmentation
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
    original_img = io.imread(image_path)
    
    # Store original dimensions for later reshaping
    if len(original_img.shape) == 3:
        height, width, channels = original_img.shape
    else:
        height, width = original_img.shape
        channels = 1
    
    # Convert to grayscale if requested
    if use_grayscale and len(original_img.shape) == 3:
        # Convert RGB to grayscale
        img = color.rgb2gray(original_img)
        # Reshape to column vector
        pixels = img.reshape(-1, 1)
        print("Using grayscale for segmentation")
    else:
        # Use color image
        img = original_img.copy()
        if len(img.shape) == 3:
            # Reshape the image to a 2D array of pixels
            pixels = img.reshape(-1, channels)
            # Convert to LAB color space (better for color segmentation)
            pixels = rgb2lab(pixels.reshape(-1, 3).astype(float) / 255.0)
            print("Using LAB color space for segmentation")
        else:
            # Image is already grayscale
            pixels = img.reshape(-1, 1)
            print("Image is already grayscale")
    
    # Try different K values and compute silhouette scores
    silhouette_scores = {}
    labels_dict = {}
    kmeans_dict = {}
    
    for k in range(k_range[0], k_range[1] + 1):
        print(f"Trying K = {k}...")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Calculate silhouette score if k > 1
        if k > 1:
            # Use a subset of pixels for silhouette calculation if the image is large
            sample_size = min(10000, len(pixels))
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            score = silhouette_score(pixels[indices], labels[indices])
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
    
    # Create segmented image by replacing each pixel with its corresponding cluster center
    if use_grayscale or len(original_img.shape) == 2:
        # For grayscale images
        centers = kmeans.cluster_centers_
        segmented_pixels = centers[labels].reshape(-1)
        
        # Create a colored version to better visualize the segments
        # Generate distinct colors for each segment
        colors = plt.cm.tab10(np.linspace(0, 1, best_k))[:, :3]  # Using tab10 colormap
        
        # Map each label to a color
        colored_segments = colors[labels]
        
        # Reshape back to image dimensions
        segmented_image_gray = segmented_pixels.reshape(height, width)
        segmented_image_color = colored_segments.reshape(height, width, 3)
        
        return segmented_image_gray, segmented_image_color, best_k, silhouette_scores
    else:
        # For color images
        centers = kmeans.cluster_centers_
        # Convert cluster centers back to RGB
        centers_rgb = (np.clip(lab2rgb(centers.reshape(-1, 1, 3)), 0, 1) * 255).astype(np.uint8)
        centers_rgb = centers_rgb.reshape(-1, 3)
        
        # Create segmented image by replacing each pixel with its corresponding cluster center
        segmented_pixels = centers_rgb[labels]
        segmented_image = segmented_pixels.reshape(height, width, channels)
        
        return segmented_image, None, best_k, silhouette_scores

def plot_results(original_image, segmented_image_gray=None, segmented_image_color=None, silhouette_scores=None, best_k=None):
    """
    Plot the original image, segmented image, and silhouette scores.
    """
    if segmented_image_gray is not None and segmented_image_color is not None:
        # For grayscale processing
        plt.figure(figsize=(15, 10))
        
        # Plot original image
        plt.subplot(1, 4, 1)
        plt.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot grayscale segmented image
        plt.subplot(1, 4, 2)
        plt.imshow(segmented_image_gray, cmap='gray')
        plt.title(f'Segmented Image (K = {best_k})')
        plt.axis('off')
        
        # Plot colored segmentation for better visualization
        plt.subplot(1, 4, 3)
        plt.imshow(segmented_image_color)
        plt.title(f'Colored Segments (K = {best_k})')
        plt.axis('off')
    else:
        # For color processing
        plt.figure(figsize=(15, 10))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot segmented image
        plt.subplot(1, 3, 2)
        plt.imshow(segmented_image_gray)  # This is actually the color segmented image
        plt.title(f'Segmented Image (K = {best_k})')
        plt.axis('off')
    
    # Plot silhouette scores
    if silhouette_scores:
        plt.subplot(1, 4 if segmented_image_color is not None else 1, 4 if segmented_image_color is not None else 3)
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
    image_path = "your_image.jpg"  
    
    # Read the original image for display
    original_image = io.imread(image_path)
    
    # Choose whether to use grayscale
    use_grayscale = True
    
    # Segment the image
    if use_grayscale:
        segmented_image_gray, segmented_image_color, best_k, silhouette_scores = segment_image_with_kmeans(
            image_path=image_path,
            k_range=(2, 10),  # Try K from 2 to 10
            use_grayscale=True,
            random_state=42
        )
        
        # Plot results
        plot_results(original_image, segmented_image_gray, segmented_image_color, silhouette_scores, best_k)
        
        # Save the segmented images
        io.imsave(f"segmented_gray_k{best_k}.jpg", (segmented_image_gray * 255).astype(np.uint8))
        io.imsave(f"segmented_color_k{best_k}.jpg", (segmented_image_color * 255).astype(np.uint8))
        print(f"Segmented images saved with K = {best_k}")
    else:
        segmented_image, _, best_k, silhouette_scores = segment_image_with_kmeans(
            image_path=image_path,
            k_range=(2, 10),  # Try K from 2 to 10
            use_grayscale=False,
            random_state=42
        )
        
        # Plot results
        plot_results(original_image, segmented_image, None, silhouette_scores, best_k)
        
        # Save the segmented image
        io.imsave(f"segmented_k{best_k}.jpg", segmented_image)
        print(f"Segmented image saved with K = {best_k}")

if __name__ == "__main__":
    main()