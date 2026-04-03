# One detected button in the inference image frame (e.g. 640×640 pixels).
int32 class_id
float64 score
float64 center_x
float64 center_y
float64 theta
float64 size_x
float64 size_y
# Horizontal offset from vertical image midline (pixels, inference frame). Positive = centroid right of center; drive until ~0 to align with the button.
float64 offset_from_center_x
