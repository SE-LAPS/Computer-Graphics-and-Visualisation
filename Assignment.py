import cv2
import numpy as np
import os
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt

class CannyEdgeDetector:
    
    def __init__(
        self,
        gaussian_kernel_size: tuple = (5, 5),
        gaussian_sigma: float = 1.4,
        high_threshold_ratio: float = 0.09,
        low_threshold_ratio: float = 0.05
    ):
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.high_threshold_ratio = high_threshold_ratio
        self.low_threshold_ratio = low_threshold_ratio
        
        # Sobel kernels for gradient calculation
        self.Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        self.Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    def process_image(self, image_path: str, save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Read and validate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Noise Reduction
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel_size, self.gaussian_sigma)
        
        # Step 2: Gradient Calculation
        Ix = cv2.filter2D(blurred, -1, self.Kx)
        Iy = cv2.filter2D(blurred, -1, self.Ky)
        
        magnitude = np.hypot(Ix, Iy)
        magnitude = magnitude / magnitude.max() * 255
        angle = np.arctan2(Iy, Ix)
        
        # Step 3: Non-Maximum Suppression
        suppressed = self._non_maximum_suppression(magnitude, angle)
        
        # Step 4: Double Threshold
        strong_edges, weak_edges = self._double_threshold(suppressed)
        edges = strong_edges + weak_edges
        
        # Step 5: Edge Tracking by Hysteresis
        final_edges = self._hysteresis(edges)
        
        # Save result if requested
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_path, final_edges)
        
        return {
            'original': gray,
            'noise_reduced': blurred,
            'gradient': magnitude.astype(np.uint8),
            'suppressed': suppressed.astype(np.uint8),
            'threshold': edges.astype(np.uint8),
            'final': final_edges.astype(np.uint8)
        }
        
    def _non_maximum_suppression(self, magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression to the gradient magnitude."""
        height, width = magnitude.shape
        suppressed = np.zeros((height, width), dtype=np.float32)
        
        # Convert angle to degrees and normalize
        angle = angle * 180 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                # Determine gradient direction
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif (22.5 <= angle[i,j] < 67.5):
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif (67.5 <= angle[i,j] < 112.5):
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                    
                if magnitude[i,j] >= max(neighbors):
                    suppressed[i,j] = magnitude[i,j]
                    
        return suppressed
        
    def _double_threshold(self, img: np.ndarray) -> tuple:
        """Apply double threshold to the image."""
        highThreshold = img.max() * self.high_threshold_ratio
        lowThreshold = highThreshold * self.low_threshold_ratio
        
        strong_edges = np.zeros_like(img)
        weak_edges = np.zeros_like(img)
        
        strong = 255
        weak = 25
        
        strong_edges[img >= highThreshold] = strong
        weak_edges[(img >= lowThreshold) & (img < highThreshold)] = weak
        
        return strong_edges, weak_edges
        
    def _hysteresis(self, edges: np.ndarray) -> np.ndarray:
        """Apply hysteresis to connect edges."""
        height, width = edges.shape
        final_edges = edges.copy()
        
        weak = 25
        strong = 255
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if final_edges[i,j] == weak:
                    if np.any(final_edges[i-1:i+2, j-1:j+2] == strong):
                        final_edges[i,j] = strong
                    else:
                        final_edges[i,j] = 0
                        
        return final_edges

def visualize_stages(stages: Dict[str, np.ndarray], save_path: Optional[str] = None) -> None:
    
    titles = {
        'original': 'Original Image',
        'noise_reduced': 'Noise Reduction',
        'gradient': 'Gradient Calculation',
        'suppressed': 'Non-maximum Suppression',
        'threshold': 'Double Threshold',
        'final': 'Edge Tracking by Hysteresis'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Canny Edge Detection Images', fontsize=16)
    
    for idx, (key, img) in enumerate(stages.items()):
        ax = axes[idx//3, idx%3]
        ax.imshow(img, cmap='gray')
        ax.set_title(titles[key])
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
def main():
    """Main function demonstrating the usage of the Canny edge detector."""
    # Initialize detector with custom parameters if needed
    detector = CannyEdgeDetector(
        gaussian_kernel_size=(5, 5),
        gaussian_sigma=1.4,
        high_threshold_ratio=0.09,
        low_threshold_ratio=0.05
    )
    
    # Example usage
    try:
        # Process first image
        stages1 = detector.process_image(
            "charlie.jpg",
            save_path="output/image1_edges.jpg"
        )
        visualize_stages(stages1, save_path="output/Result1.png")
        
        # Process second image
        stages2 = detector.process_image(
            "GS.jpg",
            save_path="output/image2_edges.jpg"
        )
        visualize_stages(stages2, save_path="output/Result2.png")
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")

if __name__ == "__main__":
    main()