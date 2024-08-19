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

def formula_and_equation(output):
    # output is a list of tuples (category, bbox)
    # if the list contains both formula and equation, and the overlap ratio is greater than 0.5, get rid of the formula 
    # and keep the equation
    formula_indices = []
    equation_indices = []
    for i, (category, bbox) in enumerate(output):
        if category == 'formula':
            formula_indices.append(i)
        elif category == 'equation':
            equation_indices.append(i)
    
    if len(formula_indices) == 0 or len(equation_indices) == 0:
        return output
    
    for i in formula_indices:
        for j in equation_indices:
            overlap_ratio = calculate_overlap_ratio(output[i][1], output[j][1])
            if overlap_ratio >= 0.5:
                output.pop(i)
                break
            
    return output

def post_process_bbox(output):
    # get the 'Text' bbox, [x1, y1, x2, y2]
    # get the 'Equation', 'Figure', 'Table' bbox [x1_, y1_, x2_, y2_]
    # if the 'Equation', 'Figure', 'Table' bbox is inside 'Text' bbox, 
    # cut the 'Text' bbox to be [x1, y1, x2, y1_], 
    # 'Equation', 'Figure', 'Table' bbox is [x1_, y1_, x2_, y2_], 
    # 'Text' bbox is [x1, y2_, x2, y2]
    
    # output is a list of tuples (category, bbox)
    
    for i, (category, bbox) in enumerate(output):
        if category == 'Text':
            x1, y1, x2, y2 = bbox
            for j, (category_, bbox_) in enumerate(output):
                if category_ in ['Equation', 'Figure', 'Table']:
                    x1_, y1_, x2_, y2_ = bbox_
                    if x1_ > x1 and x2_ < x2 and y1_ > y1 and y2_ < y2:
                        output[i] = ('Text', [x1, y1, x2, y1_])
                        output.append(('Text', [x1, y2_, x2, y2]))
                        break
                    
    return output

def sort_layout_by_columns(output, threshold: float): 
    left_column = []
    right_column = []
    
    for (category, bbox) in output:
        if bbox[0] > threshold and bbox[2] > threshold:
            right_column.append((category, bbox))
        else:
            left_column.append((category, bbox))
        
    left_sorted = sorted(left_column, key=lambda x: (x[1][0], x[1][1]))
    right_sorted = sorted(right_column, key=lambda x: (x[1][0], x[1][1]))
    
    sorted_layout = left_sorted + right_sorted
    return sorted_layout

def final_remove_overlapping_bboxes(final_output, threshold):
    # final_output에서 카테고리와 바운딩 박스 분리
    categories = [item[0] for item in final_output]
    bboxes = [item[1] for item in final_output]
    
    # 제거할 인덱스를 추적하는 리스트
    remove_indices = set()
    
    # 바운딩 박스 쌍을 순차적으로 비교
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if i in remove_indices or j in remove_indices:
                continue
            overlap_ratio = calculate_overlap_ratio(bboxes[i], bboxes[j])
            if overlap_ratio >= threshold:
                # 작은 바운딩 박스를 제거
                area_i = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
                area_j = (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] - bboxes[j][1])
                if area_i > area_j:
                    remove_indices.add(j)
                else:
                    remove_indices.add(i)
    
    # 중복이 제거된 결과 생성
    filtered_output = [(categories[i], bboxes[i]) for i in range(len(final_output)) if i not in remove_indices]
    
    return filtered_output
