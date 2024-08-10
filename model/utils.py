def calculate_overlap_ratio(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    
    # Calculate the (x, y) coordinates of the intersection rectangle
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    # Calculate the area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate the area of both bounding boxes
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_ - x1_) * (y2_ - y1_)
    
    # Calculate the overlap ratio for the smaller box
    if bbox1_area < bbox2_area:
        overlap_ratio = inter_area / bbox1_area
    else:
        overlap_ratio = inter_area / bbox2_area
    
    return overlap_ratio

def remove_overlapping_bboxes(output, threshold):
    texts = output['texts']
    bboxes = output['bboxes']
    
    # List to keep track of indices to remove
    remove_indices = set()
    
    # Iterate through each pair of bboxes
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if i in remove_indices or j in remove_indices:
                continue
            overlap_ratio = calculate_overlap_ratio(bboxes[i], bboxes[j])
            if overlap_ratio >= threshold:
                # Remove the smaller bbox
                area_i = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
                area_j = (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] - bboxes[j][1])
                if area_i > area_j:
                    remove_indices.add(j)
                else:
                    remove_indices.add(i)
    
    # Filter the bboxes and texts
    output['texts'] = [texts[i] for i in range(len(texts)) if i not in remove_indices]
    output['bboxes'] = [bboxes[i] for i in range(len(bboxes)) if i not in remove_indices]
    
    return output