import cv2
import numpy as np
import os
import matplotlib as plt
#from markdown_it.rules_block import reference


#The ReferenceImage class encapsulates all the essential details of the reference image needed for flood depth estimation.
def resize_image(image, max_width=1024, max_height=1024):
    """Resizes an image while maintaining its aspect ratio."""
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)

    if scaling_factor < 1:  # Resize only if larger than the max size
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image, scaling_factor
    return image, 1  # Return original if no resizing is needed


# Preprocessing function
def preprocess_image(image):
    """Applies CLAHE and Gaussian blur only to the masked region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray

def get_equally_spaced_points(x_fit, y_fit):
    num_points = len(x_fit)

    if num_points < 3:
        raise ValueError("Not enough points to select three equally spaced ones.")

    # Compute three indices that are approximately equally spaced
    idx1 = 0  # First point (start)
    idx2 = num_points // 2  # Middle point
    idx3 = num_points - 1  # Last point (end)

    # Extract the corresponding (x, y) coordinates
    selected_points = [
        (x_fit[idx1], y_fit[idx1]),
        (x_fit[idx2], y_fit[idx2]),
        (x_fit[idx3], y_fit[idx3])
    ]

    return selected_points

class ReferenceImage:
    def __init__(self, path, baseline, pixel_width, pixel_height, ref_object_mask, img_size):
        """
        Initializes an instance of ReferenceImage
        :param path: The path to the reference image.
        :param baseline: A pair of points that define the baseline in the image.
        :param pixel_width: The real-world width corresponding to one pixel in the image
        :param pixel_height: The real-world height corresponding to one pixel in the image
        :param ref_object_mask: A mask indicating the region of the reference object in the image
        :param img_size: The width and height of the reference image.
        """
        self.path = path
        self.ref_object_mask = ref_object_mask
        self.img_width, self.img_height = img_size
        self.image = self.load_image(path)
        self.baseline = baseline
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height


    def load_image(self, image_source):
        """
        Checks if image_source is a file path (string) or an already loaded image.
        If a path is provided, it loads the image using cv2.imread(image_source).
        If an image object is provided, it returns it as is.
        :param image_source: file  path or cv2 image.
        :return: image object.
        """
        if isinstance(image_source, str):
            image =  cv2.imread(image_source)
            result = cv2.bitwise_and(image, image, mask=self.ref_object_mask)
            # display_image_w, scale = resize_image(image)
            # cv2.imshow("Original Image", display_image_w)
            # display_image_w, scale = resize_image(self.ref_object_mask)
            # cv2.imshow("Mask", display_image_w)
            # display_image_w, scale = resize_image(result)
            # cv2.imshow("Result", display_image_w)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return result
        return image_source


class FloodDepthEstimator:
    def __init__(self, reference, scene_image, reference_object_image, flood_region_image, scene_img_size, save_dir):
        self.reference = reference
        self.scene_image = scene_image
        self.reference_object_image = self.load_image(reference_object_image)
        self.flood_region_image = flood_region_image
        self.scene_img_width, self.scene_img_height = scene_img_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    def load_image(self, image_source):
        if isinstance(image_source, str):
            return cv2.imread(image_source)
        return image_source

    def compute_homography(self):
        # Preprocess images for better feature matching
        ref_gray = preprocess_image(self.reference.image)
        #display_image_w, scale = resize_image(ref_gray)
        #cv2.imshow("Grayed Reference", display_image_w)
        target_gray = preprocess_image(self.reference_object_image)
        #display_image_w, scale = resize_image(target_gray)
        #cv2.imshow("Grayed Object", display_image_w)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(target_gray, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) < 4:
            print("Not enough good matches found!")
            return None
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return homography_matrix

    def register_images(self):
        #homography_matrix = self.compute_homography()
        # Preprocess images for better feature matching
        ref_gray = preprocess_image(self.reference.image)
        #display_image_w, scale = resize_image(ref_gray)
        #cv2.imshow("Grayed Reference", display_image_w)
        target_gray = preprocess_image(self.reference_object_image)
        # display_image_w, scale = resize_image(target_gray)
        # cv2.imshow("Grayed Object", display_image_w)
        # cv2.waitKey(0)
        #cv2.destroyAllWindows()
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(target_gray, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = [m for match in matches if len(match) > 1 for m, n in [match] if m.distance < 0.6 * n.distance]
        if len(good_matches) < 4:
            print("Not enough good matches found!")
            return None
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print("matrix: \n", homography_matrix)
        #print("image width: ",self.reference.img_width)
        #print("image height: ",self.reference.img_height)
        registered_flood = cv2.warpPerspective(self.flood_region_image, homography_matrix,
                                               (self.reference.img_height, self.reference.img_width))
        display_image_w, scale = resize_image(registered_flood)
        cv2.imshow("Warped Flood", display_image_w)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        blended = cv2.addWeighted(self.reference.image, 1.0,registered_flood, 0.3, 0)
        registered_flood_gray = cv2.cvtColor(registered_flood, cv2.COLOR_BGR2GRAY)
        _, flood_mask = cv2.threshold(registered_flood_gray, 0, 255, cv2.THRESH_BINARY)
        ref_gray = cv2.cvtColor(self.reference.image,cv2.COLOR_BGR2GRAY)
        _, reference_mask = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY)
        display_image_w, scale = resize_image(reference_mask)
        cv2.imshow("Reference Mask", display_image_w)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        intersection = cv2.bitwise_and(reference_mask, flood_mask)
        display_image_w, scale = resize_image(intersection)
        cv2.imshow("Intersection", display_image_w)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Get the height and width of the intersection image
        height, width = intersection.shape
        # Store the first non-zero pixel in each column
        points = []
        for col in range(width):
            non_zero_pixels = np.where(intersection[:, col] > 0)[0]  # Get row indices of non-zero pixels
            if len(non_zero_pixels) > 0:
                first_non_zero = non_zero_pixels[0]  # First non-zero pixel in the column
                points.append((col, first_non_zero))  # (x, y) format

        # Convert to NumPy array
        points = np.array(points)
        if len(points) > 1:  # Ensure we have enough points to fit a line
            x_coords = points[:, 0]  # Extract x-coordinates
            y_coords = points[:, 1]  # Extract y-coordinates

            # Fit a line (y = mx + c) using np.polyfit (degree=1 for linear fit)
            m, c = np.polyfit(x_coords, y_coords, 1)

            # Generate points along the best-fit line
            x_fit = np.linspace(x_coords.min(), x_coords.max(), num=width).astype(int)  # Evenly spaced x-values
            y_fit = (m * x_fit + c).astype(int)  # Compute corresponding y-values

            # **Clamp values to ensure they are within valid image bounds**
            x_fit = np.clip(x_fit, 0, width - 1)
            y_fit = np.clip(y_fit, 0, height - 1)

            # Draw the best-fit line on the image
            for i in range(len(x_fit) - 1):
                cv2.line(blended, (x_fit[i], y_fit[i]), (x_fit[i + 1], y_fit[i + 1]), (0, 0, 255), 10)  # Red line
            # Show result
        # Draw lines connecting these points
        #for i in range(len(points) - 1):
            #cv2.line(blended, points[i], points[i + 1], (255, 255, 255), 2)  # White line
        # Example usage
        selected_points = get_equally_spaced_points(x_fit, y_fit)
        print("Selected points:", selected_points)
        #overlay = cv2.bitwise_or(self.reference.image, registered_flood)
        gradient = (self.reference.baseline[1][1] - self.reference.baseline[0][1]) / (self.reference.baseline[1][0] - self.reference.baseline[0][0])
        intercept = self.reference.baseline[0][1] - gradient * self.reference.baseline[0][0];
        print("gradient",gradient)
        print("intercept",intercept)
        height, width = self.reference.image.shape[:2]  # Get image dimensions
        new_coordinates = []
        for (x_fit_i, y_fit_i) in selected_points:
            y_at = x_fit_i * gradient  + intercept # Compute x_at
            y_at = int(round(y_at))  # Round and convert to int

            # Ensure valid image indices
            y_at = max(0, min(y_at, height - 1))
            x_fit_i = max(0, min(x_fit_i, width - 1))
            new_coordinates.append((x_fit_i, y_at))
        for i in range(len(new_coordinates)):
            cv2.arrowedLine(blended, selected_points[i],new_coordinates[i] ,(255, 0, 0), 10)
        cv2.line(blended, self.reference.baseline[0], self.reference.baseline[1], (0, 255, 0), 5) #green
        display_image_w, scale = resize_image(blended)
        cv2.imshow("Blended", display_image_w)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("New coordinates:", new_coordinates)
        homography_inv = np.linalg.inv(homography_matrix)  # Compute inverse of homography mat

        # Transform each point using the inverse homography
        transformed_points = []
        average_depth =0
        for i in range(len(new_coordinates)):
            x, y = selected_points[i]  # Extract the coordinate
            pixel_depth = new_coordinates[i][1] - selected_points[i][1]
            real_depth = pixel_depth * self.reference.pixel_height
            average_depth += real_depth
            print("Depth : ", i, pixel_depth, real_depth)
            # Convert to homogeneous coordinates
            point = np.array([[[x, y]]], dtype=np.float32)
            # Apply inverse homography transformation
            transformed_point = cv2.perspectiveTransform(point, homography_inv)
            # Extract transformed (x', y') and ensure they are valid image indices
            x_trans, y_trans = map(int, transformed_point[0][0])  # Ensure integer coordinates
            if np.isnan(x_trans) or np.isnan(y_trans):
                print("Invalid transformed coordinates:", transformed_point)
            start_y = 100
            start_x =  min(x_trans + 10, self.scene_img_width - 1)
            if 0 <= x_trans < self.scene_img_width and 0 <= y_trans < self.scene_img_height:
                cv2.arrowedLine(self.scene_image, (x_trans , start_y), (x_trans, y_trans),(0, 255, 0), 2)
                cv2.putText(self.scene_image, f"{real_depth:.2f} mm", (start_x, start_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            transformed_points.append((int(x_trans), int(y_trans)))

        # Draw the transformed points as a line on the image
        for i in range(len(transformed_points) - 1):
            cv2.line(self.scene_image, transformed_points[i], transformed_points[i + 1], (0, 0, 255), 2)  # Red line
        display_image_w, scale = resize_image(self.scene_image)
        cv2.imshow("Blended", display_image_w)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(self.save_dir, "registered_flood.png"), blended)
        cv2.imwrite(os.path.join(self.save_dir, "depth_overlay.png"), self.scene_image)


    def process(self):
        self.register_images()



class Helper:

    @staticmethod
    def select_baseline(image_source):
        """Allows the user to select two points to define a baseline and see it before confirming."""
        image = cv2.imread(image_source) if isinstance(image_source, str) else image_source
        display_image, scale_factor = resize_image(image)
        clone = display_image.copy()
        points = []

        def draw_baseline(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                if len(points) == 2:
                    cv2.line(clone, points[0], points[1], (0, 0, 255), 2)
                    cv2.imshow("Draw Baseline", clone)
                    print("Press any key to confirm baseline selection.")

        cv2.imshow("Draw Baseline", clone)
        cv2.setMouseCallback("Draw Baseline", draw_baseline)

        # Wait for the user to confirm after drawing
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(points) != 2:
            raise ValueError("You must select exactly two points for the baseline.")

        # Scale points back to original size
        baseline_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in points]
        return baseline_points

    @staticmethod
    def create_reference_image(path, pixel_width, pixel_height):
        """Creates a reference image structure with object mask and baseline."""
        image = cv2.imread(path)

        # Select the reference object region
        ref_object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi = Helper.select_polygon_roi(image)
        print("ROI shape:", roi.shape)
        print("ROI coordinates:", roi)
        cv2.fillPoly(ref_object_mask, [roi], 255)

        # Select baseline
        baseline_points = Helper.select_baseline(image)

        return ReferenceImage(path, baseline_points, pixel_width, pixel_height, ref_object_mask, image.shape[:2])
    #This decorator allows the method to be called on the class itself (Helper.select_region_of_interest(...))
    # without needing to instantiate an object of the Helper class.
    @staticmethod
    def select_rectangular_region_of_interest(image_source):
        #If image_source is a file path (string), it loads the image using OpenCV’s cv2.imread(image_source)
        #If image_source is already an image object (NumPy array, as returned by cv2.imread), it uses it directly
        image = cv2.imread(image_source) if isinstance(image_source, str) else image_source
        #Opens a GUI window titled "Select Region of Interest", where the user can click and drag to
        # select a rectangular portion of the image
        #cv2.selectROI returns a tuple: (x, y, width, height), which specifies:
        #(x, y): The top-left corner of the selected region.
        #width, height: The dimensions of the selected region.

        roi = cv2.selectROI("Select Region of Interest", image, False, False)
        #Uses NumPy array slicing to extract the selected region from the image:
        #roi[1] (y-coordinate of the top-left corner) to roi[1] + roi[3] (y + height)
        #roi[0] (x-coordinate of the top-left corner) to roi[0] + roi[2] (x + width)
        roi_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        #The method returns the cropped region of interest (ROI) as an image (NumPy array)
        return roi_image

    @staticmethod
    def select_polygon_roi(image_source):
        """Allows the user to select a polygon region of interest (ROI)."""
        image = cv2.imread(image_source) if isinstance(image_source, str) else image_source

        # Resize image for display
        display_image, scale_factor = resize_image(image)

        clone = display_image.copy()
        points = []

        def draw_polygon(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to select points
                points.append((x, y))
                cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow("Select Region", clone)
            elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click to finalize selection
                if len(points) > 2:
                    cv2.line(clone, points[-1], points[0], (0, 255, 0), 2)
                    cv2.imshow("Select Region", clone)
                cv2.destroyAllWindows()

        cv2.imshow("Select Region", clone)
        cv2.setMouseCallback("Select Region", draw_polygon)
        cv2.waitKey(0)

        # Scale points back to the original image size
        scaled_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in points]

        return np.array(scaled_points, np.int32).reshape((-1, 1, 2))

    @staticmethod
    def select_region_of_interest(image_source):
        image = cv2.imread(image_source) if isinstance(image_source, str) else image_source
        roi = Helper.select_polygon_roi(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi], 255)
        roi_image = cv2.bitwise_and(image, image, mask=mask)
        return roi_image


def find_intersection(x0, y0, baseline_start, baseline_end):
    """
    Finds the intersection of a vertical line from (x0, y0) onto an inclined baseline.

    :param x0: x-coordinate of the dropping point
    :param y0: y-coordinate of the dropping point (unused but given)
    :param baseline_start: (x1, y1) - first point on the baseline
    :param baseline_end: (x2, y2) - second point on the baseline
    :return: (x_intersect, y_intersect) if valid, else None
    """
    x1, y1 = baseline_start
    x2, y2 = baseline_end

    # Compute slope m of the baseline
    if x1 == x2:
        return None  # Baseline is vertical, no unique intersection

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1  # Compute y-intercept

    # Compute intersection point
    x_intersect = x0
    y_intersect = m * x_intersect + c

    # Check if the intersection is within the baseline segment
    if min(x1, x2) <= x_intersect <= max(x1, x2):
        return (x_intersect, y_intersect)
    else:
        return None  # Intersection is outside the baseline segment







# Example Usage:
baseline_start = (0, 767)
baseline_end = (1831, 767)  # Not perfectly horizontal
x0 = 1500  # Example dropping point x-coordinate
y0 = 3200  # Example dropping point y-coordinate

intersection = find_intersection(x0, y0, baseline_start, baseline_end)
if intersection:
    print("Intersection point:", intersection)
else:
    print("No intersection within the baseline range.")

REFERENCE_IMG_PATH = "To Embed/Reference_Taxi_Body_Outline.jpg"
TARGET_IMG_PATH = "To Embed/624_marked.jpg"
reference_image = Helper.create_reference_image(REFERENCE_IMG_PATH, 2.5163,2.5163);
reference_object = Helper.select_region_of_interest(TARGET_IMG_PATH )
flood_image = Helper.select_region_of_interest(TARGET_IMG_PATH )
h , w , _ = reference_object.shape
print("Baseline:", reference_image.baseline)
estimator = FloodDepthEstimator(reference_image,TARGET_IMG_PATH,reference_object,flood_image,(w, h),"results");
estimator.process();