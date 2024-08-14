import argparse
import json
import logging
import re
from pathlib import Path

import layoutparser as lp
import numpy as np
import pdf2image
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, AutoProcessor

from utils import remove_overlapping_bboxes


class DocumentProcessor:

    def __init__(self, pdf_path: Path, model_path: Path, result_path: Path, visualize=False, save_result=False):
        self.pdf_path = pdf_path
        self.model_path = model_path
        self.result_path = result_path
        self.visualize = visualize
        self.pdf_id = pdf_path.stem
        self.save_result = save_result

        # Labels
        self.label_list = [
            'Caption', 
            'Footnote', 
            'Formula', 
            'List-item', 
            'Page-footer', 
            'Page-header', 
            'Picture', 
            'Section-header', 
            'Table', 
            'Text', 
            'Title'
        ]
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

        # Models
        self.parser_model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', 
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5], 
            label_map={
                0: "Text", 
                1: "Title", 
                2: "List", 
                3: "Table", 
                4: "Figure"
            })
        self.lm_model = AutoModelForTokenClassification.from_pretrained(
            model_path)
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False)
        self.formula_model = lp.Detectron2LayoutModel(
            config_path='lp://MFD/faster_rcnn_R_50_FPN_3x/config', 
            label_map={1: "Equation"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5])

    def sort_layout_by_columns(self, layout, threshold):
        left, right = [], []
        for block in layout:
            if block.block.x_1 > threshold and block.block.x_2 > threshold:
                right.append(block)
            else:
                left.append(block)
        left_sorted = sorted(left, key=lambda blk: (blk.block.y_1, blk.block.x_1))
        right_sorted = sorted(right, key=lambda blk: (blk.block.y_1, blk.block.x_1))
        sorted_layout = lp.Layout()
        sorted_layout.extend(left_sorted)
        sorted_layout.extend(right_sorted)

        logging.debug(f'self.sort_layout_by_columns :: sorted_layout = {sorted_layout}')
        return sorted_layout
    
    def extract_bboxes(self, layout_result):
        return [list(block.block.coordinates) for block in layout_result]
    
    def ocr(self, image, layout_result):
        texts = []
        for block in layout_result:
            x1, y1, x2, y2 = map(int, block.block.coordinates)
            segment_image = np.asarray(image)[y1:y2, x1:x2]
            segment_image_pil = Image.fromarray(segment_image)
            ocr_agent = lp.TesseractAgent(languages='eng')
            text = ocr_agent.detect(segment_image_pil) if block.type != "Figure" else f'{segment_image_pil}'
            text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
            texts.append(text)

        logging.debug(f'self.ocr :: texts = {texts}')
        return texts
    
    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]
    
    def normalize(self, examples):
        coco_width = examples['width']
        coco_height = examples['height']
        
        normalized_bboxes = []
        ## bboxes_block가 비어 있는지 확인해봅시다.
        if examples['bboxes']:    
            for bbox in examples['bboxes']:
    		        ## (x,y,w,h) -> (x,y,x+w,y+h)
    		        ## 가끔 음수가 뜨기도 해서 max(0,val) 적용해줍니다.
                x1 = max(0,bbox[0])
                y1 = max(0,bbox[1])
                x2 = max(0,bbox[2])
                y2 = max(0,bbox[3])
    
                normalized_bbox = [
                    int(np.rint(x1 / coco_width * 1000)),
                    int(np.rint(y1 / coco_height * 1000)),
                    int(np.rint(x2 / coco_width * 1000)),
                    int(np.rint(y2 / coco_height * 1000))
                ]
                
                normalized_bboxes.append(normalized_bbox)
                
        return {
            "bboxes": normalized_bboxes,
            "texts" : examples["texts"],
            "height" : examples["height"],
            "width" : examples["width"]
        }

    def layout_parser(self, img):
        img_np = np.asarray(img)
        layout_result = self.parser_model.detect(img_np)
        logging.debug(f'self.layout_parser :: layout_result = {layout_result}')
    
        layout_result = self.sort_layout_by_columns(layout_result, threshold=img_np.shape[1] // 2)
        return {"bboxes": self.extract_bboxes(layout_result), "texts": self.ocr(img, layout_result), "width": img_np.shape[1], "height": img_np.shape[0]}

    def math_formula(self, img):
        bboxes = []
        layout = self.formula_model.detect(img)
        for block in layout._blocks:
            print(block)
            bbox = list(block.block.coordinates)

            bboxes.append(bbox)

        logging.debug(f'self.math_formula :: bboxes = {bboxes}')
        return bboxes
    
    def segment_image(self, img, output, page_number):
        ocr_result = []
        img_np = np.array(img)  # PIL 이미지를 NumPy 배열로 변환
        img_height, img_width = img_np.shape[:2]  # 이미지 크기 가져오기

        save_path: Path = self.result_path / "visualize" / self.pdf_id / str(page_number)
        save_path.mkdir(parents=True, exist_ok=True)

        for id, (category, bbox) in enumerate(output):
            x1, y1, x2, y2 = map(int, bbox)
            
            # 바운딩 박스 좌표를 이미지 범위 내에 있도록 조정
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            if x1 >= x2 or y1 >= y2:
                continue  # 바운딩 박스가 유효하지 않은 경우 건너뜀
            
            segment_image = img_np[y1:y2, x1:x2]
            segment_image_pil = Image.fromarray(segment_image)
            ocr_agent = lp.TesseractAgent(languages='eng')
            text = ocr_agent.detect(segment_image_pil)
            text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
            ocr_result.append((id, category, bbox, text))
            
            if category in ["Figure", "Table"]:
                segment_image_pil.save(save_path / Path(str(id)).with_suffix(".png"))

        logging.debug(f'self.segment_image :: ocr_result = {ocr_result}')
        return ocr_result
    
    def visualize_image(self, img, output, page_number):
        img_np = np.array(img)  # PIL 이미지를 NumPy 배열로 변환
        img_height, img_width = img_np.shape[:2]  # 이미지 크기 가져오기

        category_colors = {
            "Text": (0, 255, 0), 
            "Title": (255, 0, 0), 
            "List": (0, 0, 255), 
            "Table": (255, 255, 0),
            "Figure": (255, 0, 255), 
            "Caption": (0, 255, 255), 
            "Footnote": (128, 0, 128),
            "Formula": (128, 128, 0), 
            "Page-footer": (0, 128, 128), 
            "Page-header": (128, 0, 0),
            "Section-header": (0, 128, 0), 
            "Equation": (255, 165, 0)  # Orange
        }

        # PIL 이미지 객체 생성
        image = Image.fromarray(img_np)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for category, bbox in output:
            x1, y1, x2, y2 = map(int, bbox)
            
            # 바운딩 박스 좌표를 이미지 범위 내에 있도록 조정
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            if x1 >= x2 or y1 >= y2:
                continue  # 바운딩 박스가 유효하지 않은 경우 건너뜀
            
            color = category_colors.get(category, (255, 255, 255))  # Default to white if category not found
            draw.rectangle([x1, y1, x2, y2], outline=color)
            draw.text((x1 + 10, y1 - 10), category, fill=color, font=font)

        output_dir = self.result_path / "visualize" / self.pdf_id
        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / Path(str(page_number)).with_suffix(".png"))

    def create_json_from_output(self, output, save=False):
        json_data = {"id": self.pdf_id, "elements": []}
        for page_number, page_output in enumerate(output):
            for id, category, bbox, text in page_output:
                json_data["elements"].append({
                    "bounding_box": [{"x": bbox[0], "y": bbox[1]}, {"x": bbox[2], "y": bbox[1]}, {"x": bbox[2], "y": bbox[3]}, {"x": bbox[0], "y": bbox[3]}],
                    "category": category,
                    "id": id,
                    "page": page_number + 1,
                    "text": text,
                })

        if save:
            json_path = self.result_path / "json"
            json_path.mkdir(parents=True, exist_ok=True)

            with open(json_path / Path(self.pdf_id).with_suffix(".json"), 'w') as f:
                json.dump(json_data, f, indent=2)
        
        logging.debug(f'self.create_json_from_output :: json_data = {json_data}')
        return json_data

    def process(self):
        results = []
        images = pdf2image.convert_from_path(self.pdf_path)
        logging.info(f'Processing started for {self.pdf_id}')

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Device: {device}')

        for page_number, img in enumerate(images):
            logging.info(f'Processing {self.pdf_id}, page {page_number + 1}')

            logging.debug(f'Calling `layout_parser` on {self.pdf_id}, page {page_number + 1}')
            output = self.layout_parser(img)
            
            logging.debug(f'Calling `remove_overlapping_boxes` on {self.pdf_id}, page {page_number + 1}')
            threshold = 0.6
            output = remove_overlapping_bboxes(output, threshold)

            logging.debug(f'Calling `normalize` on {self.pdf_id}, page {page_number + 1}')
            output = self.normalize(output)
            words, boxes = output["texts"], output["bboxes"]
            words = [word[:80] for word in words]
            width, height = output["width"], output["height"]

            logging.debug(f'Calling `self.processor` on {self.pdf_id}, page {page_number + 1}')
            encoding = self.processor(
                img, words, boxes=boxes, return_offsets_mapping=True, 
                return_tensors="pt", truncation=True, padding="max_length")
            offset_mapping = encoding.pop('offset_mapping')
            
            logging.debug(f'Predicting on {self.pdf_id}, page {page_number + 1}')
            self.lm_model.to(device)
            encoding.to(device)
            outputs = self.lm_model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_boxes = encoding.bbox.squeeze().tolist()
            is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
            true_predictions = [self.id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
            true_boxes = [self.unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
            final_output = [(category, bbox) for category, bbox in zip(true_predictions, true_boxes) if bbox != [0, 0, 0, 0]]

            bboxes = self.math_formula(img)
            if bboxes:
                final_output += [("Equation", box) for box in bboxes]

            if self.visualize:
                self.visualize_image(img, final_output, page_number + 1)
            results.append(self.segment_image(img, final_output, page_number + 1))
        
        return self.create_json_from_output(results, save=self.save_result)


# Run module
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Layout Parser for LayoutLMv3')
    parser.add_argument('-p', '--pdf_path', type=str, default='/root/paper_pdf/eess', help='Path to the PDF Directory')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the Model Directory')
    parser.add_argument('-r', '--result_path', type=str, default='/root/result/eess', help='Path to the Result Directory')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug Mode: Process only 1 PDF files')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize Mode: Visualize the result of detection')
    args = parser.parse_args()
    
    pdf_directory = Path(args.pdf_path)
    model_path = args.model_path
    result_directory = Path(args.result_path)
    debug = args.debug
    visualize = args.visualize
    
    # Logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    # Read PDF files from directory
    for pdf_file in pdf_directory.glob('**/*.pdf'):
        json_filename = pdf_file.with_suffix('json')
        json_path = result_directory / 'json' / json_filename

        # Create if not exists, else continue
        if json_path.exists():
            logging.warning(f'{json_filename} JSON already exists')
            continue
        
        # Process files
        processor = DocumentProcessor(
            pdf_file, model_path, result_directory, visualize, save_result=True)
        result = processor.process()
        logging.info(f'Processed output: result = {result}')

        # Debug
        if debug:
            break
