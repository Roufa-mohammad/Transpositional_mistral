import io
import torch
import pdfplumber
from word2number import w2n
import pdfplumber
import fitz
import PyPDF2

import time
from PIL import Image
from PIL import Image, ImageDraw
from pypdf import PdfReader
from pdf2image import convert_from_path
from transformers import DetrImageProcessor, DetrForObjectDetection
from dummy_4_1 import is_super_Script_present,extract_references_text_indexes,filter_references,\
                                extract_references_text_from_details,remove_references_from_doc

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import re
import pdfplumber
from fuzzywuzzy import fuzz
from bounding_box import merge_bboxes
from json_output import return_json_result



def find_starting_page_TOC(path):
    reader = pdfplumber.open(path)

    # get contents list till index
    num_pages = len(reader.pages)
    (
        # page_content,
        page_content_dict,
        common_text,
        text_list_compare_prev,
        content_page_flg,
        prev_break_flg,
        cur_break_flg,
        content_page_cnt_flg,
        content_page_cnt,
        
    ) = (
        # [],
        {},
        [],
        [],
        False,
        False,
        False,
        True,
        0,
    
    )
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text_data = page.within_bbox((0, 0, 550, 770)).extract_text_lines()
        text = [td.get("text") for td in text_data]
        text = "\n".join(text)
        
        text_dict = {str(page_num+1)+'~'+str(n)+"~"+td.get("text"): td.get("x0") for n,td in enumerate(text_data)}
 
        
        text_list_compare = re.sub(r"[^a-zA-Z0-9 .:)\]\n]", "", text)
        text_list_compare = text_list_compare.split("\n")

        text_list_compare = [
            i.strip().lower() for i in text_list_compare if i.strip() != ""
        ]
        text_list_compare_prev.extend(text_list_compare)
    
        if (len(text_list_compare) > 0) and (((len(text_list_compare[0].split()) < 5)
            and (
                "content" in "".join(text_list_compare[0])
                or "index" in "".join(text_list_compare[0])
    
            )) or content_page_flg):
                    
            if content_page_cnt_flg:
                content_page_cnt += 1
                
            if prev_break_flg:
                if (
                    (len(text_list_compare) > 0)
                    and (len(text_list_compare[0].split()) < 5)
                    and (
                        "content" in "".join(text_list_compare)
                        or "index" in "".join(text_list_compare)

                    )
                ):
                    cur_break_flg = True
                    prev_break_flg = False
                    content_page_cnt_flg = True
                    content_page_cnt = 0
                    page_content_dict.clear()
                else:
                    break
                
            content_page_flg = True
            page_content_dict.update(text_dict)
            
            num = page_num

            for i in [
                "appendix",
                "bibliography",
                "index",
                "list of",
                "references",
                "contributors",
            ]:
                if ((fuzz.partial_ratio(i, text_list_compare[-1]) > 89) and (len(text_list_compare[-1].split()) < 4)) or (content_page_cnt > 10):
                    prev_break_flg = True
                    if content_page_cnt > 10:
                        content_page_cnt_flg = False
                    break
            if prev_break_flg and cur_break_flg:
                break
        elif (len(text_list_compare) == 0) and content_page_flg:
            break

    return page_content_dict,page_num+1


processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

def detr_table_boxes(image):
    '''detr model to extract bounding box for bordered and boardedless tables'''
    image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    final_bbox = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        final_bbox.append(box)


    return final_bbox

def not_within_bboxes(obj, bboxes):
    """Check if the object is in any of the table's bbox."""
    def obj_in_bbox(_bbox):

        
        v_mid = (obj["top"] + obj["bottom"]) / 2
        h_mid = (obj["x0"] + obj["x1"]) / 2
        x0, top, x1, bottom = _bbox
        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
    return not any(obj_in_bbox(__bbox) for __bbox in bboxes)


def img_cords_to_page_cords(boxes, img_width, pdf_width):
    '''this code convert image coordinate table bounding box to pdf coordiante bounding box'''
    x1, y1, x2, y2 = map(float, boxes)
    
    x1 = x1 / img_width * pdf_width
    y1 = y1 / img_width * pdf_width
    x2 = x2 / img_width * pdf_width
    y2 = y2 / img_width * pdf_width
    
    return (x1,y1,x2,y2)


def get_text_without_references(path):
    pdf = pdfplumber.open(path)
    super_script_results =  is_super_Script_present(pdf)
    references_indexes = extract_references_text_indexes(pdf)
    filtered_references_indexes = filter_references(references_indexes)
    reference_details = extract_references_text_from_details(pdf, filtered_references_indexes)
    clean_lines_per_page = remove_references_from_doc(pdf, reference_details, super_script_results)
    return clean_lines_per_page


def transposition_details(pdf_path,pdf,start_num,text_ref_dictionary):
    '''
    This fucntions find page has border or borderless table and removing the table,
    this code takes each line pass though number-treatment code and return final_transposition_context  which is having pageno,line_no,and there corresponding text  
    '''
    sample_dict = {}
    final_transposition_context = {}
    output_dict = {}
    

    for page_no, page in enumerate(pdf.pages):
       
        
        #if page_no > start_num:
        if page_no >= 75 and page_no <= 76:
            if page.find_tables():
                images = convert_from_path(pdf_path, first_page=page_no+1 , last_page=page_no+1 )
                for j, img in enumerate(images):
                    # Save the image as JPG and pass it to detr_table_boxes
                    img_as_bytes = io.BytesIO()
                    # Convert the current page to images

                    img.save(img_as_bytes, format="JPEG")
                    img_as_bytes.seek(0)
                    
                    # Detect and get bounding boxes
                    bboxes = detr_table_boxes(img_as_bytes)
                    if bboxes:
                        pdf_width  = page.width
                        pdf_height = page.height
                        image = Image.open(img_as_bytes).convert("RGB")
                        width,height = image.size
                    

                        convert_box =[]
                        for boxes in bboxes:
                            new = img_cords_to_page_cords(boxes, width, pdf_width)
                            convert_box.append(new)

                        for line_no, dictionary in enumerate(page.extract_text_lines()):
                            del dictionary["chars"]
                            
                            for page_cords in convert_box:
                                if dictionary['top'] < int(page_cords[1]) or dictionary["top"] > int(page_cords[3]):
                                    sample_dict[line_no] = dictionary["text"]
                        val_text_if = text_ref_dictionary.get(page_no+1)
                        for key,val_if in sample_dict.items():
                            try:                                
                                if val_if in val_text_if["text"] :
                                  
                                    if page_no+1 not in final_transposition_context:
                                        final_transposition_context[page_no+1] = {}
                                    if key  not in final_transposition_context[page_no+1]:
                                        final_transposition_context[page_no+1][key] = []
                                    final_transposition_context[page_no+1][key].append(val_if)
                     
                            except:
                                pass
                                
                    else:
                        for line_no, dictionary in enumerate(page.extract_text_lines()):
                            del dictionary["chars"]
                            sample_dict[line_no] = dictionary["text"]
                        val_text_else = text_ref_dictionary.get(page_no+1)
                        for key,val_else in sample_dict.items():
                            try:
                                if val_else in val_text_else["text"]:
                                    if page_no+1 not in final_transposition_context:
                                        final_transposition_context[page_no+1] = {}
                                    if key  not in final_transposition_context[page_no+1]:
                                        final_transposition_context[page_no+1][key] = []
                                    final_transposition_context[page_no+1][key].append(val_else)

                            except:
                                pass                 
                                          
            else:
                
                for line_no, dictionary in enumerate(page.extract_text_lines()):
                    del dictionary["chars"]              
             
                    sample_dict[line_no] = dictionary["text"]

                val_text = text_ref_dictionary.get(page_no+1)
                for key,val in sample_dict.items():
                    try:
                        if val in val_text["text"]:
                            if page_no+1 not in final_transposition_context:
                                final_transposition_context[page_no+1] = {}
                            if key  not in final_transposition_context[page_no+1]:
                                    final_transposition_context[page_no+1][key] = []
                            final_transposition_context[page_no+1][key].append(val)
    
                    except:
                        pass
                                    
    return final_transposition_context
          

def math_line_in_sentece(output_dict):
    '''  This function check for line in sentence and sentece in line in order to get line no for text  '''
    new_dict = {}
    for smaple_k,smaple_v in output_dict.items():
        lines_list = list(smaple_v.values())
        line_nos_list = list(smaple_v.keys())
        lines_list = [lines_Data for lines_Data in lines_list if not lines_Data[0].startswith('.')]


        all_text = list(smaple_v.values())
        all_text = sum(all_text,[] )
        all_text = " ".join(all_text)
        all_sentences = [row_line.lstrip() for row_line in all_text.split(".")]
        all_sentences = [f'{sentence}.' for sentence in all_sentences ]

        # count = 0

        for line1 in all_sentences:
            matched_lines = [[]]
            
            for line_no, line in zip(line_nos_list, lines_list):
                
                if line[0] in line1:

                    matched_lines.append(line_no)
                    
                elif line1 in line[0]:
                    if not line1.startswith('.'):
                        matched_lines.append(line_no)
                else:
                    split_line =line[0].split('.')
                    split_line = [split_l.lstrip() for split_l in split_line if split_l.lstrip()]
                    split_line = [f'{split_l}.' for split_l in split_line]
                    for line_split in split_line:
                        if line_split in line1:
                            
                            matched_lines.append(line_no)
                              
            if smaple_k not in new_dict:
                new_dict[smaple_k] = {}
                
            if not line1.startswith('.'):    
                new_dict[smaple_k][line1] = matched_lines

    return new_dict


def mistral(model_name_or_path):
    ''' mistral model loading'''
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model_2,tokenizer_2

def Generate_response(prompt,new_dict,model_2,tokenizer_2):
    ''' This fucntion is to Generate response for the sentence and add this responses to the yes_no_Dictionary '''
   
    
    yes_no_dictionary = {}
    for pno in new_dict:
        
        sentences = list(new_dict[pno].keys())
        page_nos = list(new_dict[pno].values())
        sentences = [f"Sentence {i} "+sent for i,sent in enumerate(sentences)]
        paragraph =  "\n".join(sentences)
        
        new_prompt = prompt.format(paragraph)
        prompt_template = f'''<s>[INST] {new_prompt} +'~' [/INST]'''

        pipe = pipeline(
        "text-generation",
        model=model_2,
        tokenizer=tokenizer_2,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
        )

        
        response = pipe(prompt_template)[0]['generated_text']

        
        split_respose = response.split('~')[-1]
        
        list_of_sentence = []
        for each_sentence in split_respose.split('\n'):
            
            try:
                recover_sentece = each_sentence.split(":")[1]
                list_of_sentence.append(recover_sentece.split()[0])
            except:
                list_of_sentence.append('No')
        yes_no_dictionary[pno]=list_of_sentence

    return yes_no_dictionary

def complete_call(path):
    
    pdf = pdfplumber.open(path)

    text_ref_dictionary = get_text_without_references(path)                
    page_content_dict,page_num = find_starting_page_TOC(path)
    output_dict = transposition_details(path,pdf,page_num,text_ref_dictionary)
    return output_dict



def get_transposition_error_details(path, prompt):
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

    output_dict = complete_call(path)
    new_dict = math_line_in_sentece(output_dict)
    model_2,tokenizer_2 = mistral(model_name_or_path)
    yes_no_dictionary = Generate_response(prompt, new_dict, model_2, tokenizer_2)
    new_dict_copy = new_dict.copy()

    for key, sub_dict in new_dict_copy.items():
        if key in yes_no_dictionary:
            yes_no_list = yes_no_dictionary[key]
            for sub_key, sub_value in sub_dict.items():
                # Append corresponding values from yes_no_list
                res = sub_dict.get(sub_key)
                res.append(yes_no_list.pop(0) if yes_no_list else [None])
            
                sub_dict[sub_key] = res
                
    return new_dict_copy,output_dict


def get_all_bboxes(new_dict_copy, path):
    pdf_document = fitz.open(path)
    num_pages = pdf_document.page_count
    """Below code is to add coodinate of the text"""
    for dit_k, dit_v in new_dict_copy.items():
    
        for key, val in dit_v.items():
            for page_number in range(num_pages):
                if page_number+1 == dit_k:
                    page = pdf_document[page_number]
                    
                    text_instances = page.search_for(key)
                    merge_boxes = merge_bboxes(text_instances)

                    if merge_boxes:
                        # Append text_instances to the sub-dictionary
                        page_width = page.rect.width
                        page_height = page.rect.height
                        val.append((page_width,page_height))
                        val[0].extend([coordinate[0], coordinate[1], coordinate[2], coordinate[3] ]  for coordinate in merge_boxes )
    
    return new_dict_copy,pdf_document


def process_ouput(new_dict_copy):
    filtered_dict = {}
    '''Below code take only that dictinary which is having response yes '''
    for key, value in new_dict_copy.items():
        if any(('Yes' in str(v) or 'YES' in str(v) or 'Yes,' in str(v) or 'YES,' in str(v)) for v in value.values()):
            filtered_dict[key] = {k: v for k, v in value.items() if ('Yes' in str(v) or 'YES' in str(v) or 'Yes,' in str(v) or 'YES,' in str(v))}
    return filtered_dict







path = '/data/copy_assessment_tool/modules/data/15031-4988-FullBook.pdf'


prompt = ''' A transposition error refers to the rearrangement or swapping of words Below are the some of the example of sentence
            Questions: Does the following sentences have transpositional error(): 'I love to books read in the evening.'
            Answer :Yes There is Transpostional error in a Sentece
            Questions: Does the following sentences have transpositional error(): 'Birds soared gracefully in the distance, adding to the scene serene.'
            Answer: NO Transpositinal error 
            Question: Does the following sentences have transpositional error:"The quick lazy fox jumped over the brown dog. It was a day sunny in the forest, and all the were animals enjoying their time."
            Answer: :Yes There is Transpostional error in a Sentece 
            Question:Does the following sentences have transpositional error:'She danced across the stage gracefully, capturing attention the audience's with every move. The played music in perfect with harmony her movements.'
            Answer: :Yes There is Transpostional error in a Sentece
            Question: Does the following sentences have transpositional error:The old bookstore at the end of the street is a charming place to discover hidden literary gems. People often spend hours browsing through its dusty shelves
            Answer: No Transpositional error
            Now please generate Answer for below text such that each sentece respose would be Yes Trasposition error if sentence has and NO Transpositional error if sentece not have Transpostional error
            
            Question:Does the following sentences have transpositional error(check each sentences):{}
            '''


new_dict_copy,output_dict = get_transposition_error_details(path, prompt)

filtered_dict =  process_ouput(new_dict_copy)

output_dict = complete_call(path)


new_dict_copy , pdf_document= get_all_bboxes(filtered_dict, path)


def generate_dict_format(new_dict_copy, output_dict,pdf_document):
    final_result = []
    for pgno in new_dict_copy:
        linenum_line_dict = output_dict[pgno]
        sentence_cord_dict = new_dict_copy[pgno]
        page = pdf_document[pgno-1]
        
        for setence, detail_info in sentence_cord_dict.items():
            cords_and_linenos = detail_info[:-2]
            response, page_dims = detail_info[-2:]
            cords, line_nos = cords_and_linenos[0], cords_and_linenos[1:]
     
            for i,cord in enumerate(cords):
                text = page.get_textbox(cords[i])
                final_dict = {}
                final_dict["page"] = pgno
                final_dict["text"] = text
                final_dict["bbox"] = {"x1":cords[i][0],
                                    "y1":cords[i][1],
                                    "x2":cords[i][2],
                                    "y2":cords[i][3],
                                    "width":page_dims[0],
                                    "height":page_dims[1]
                                    }
                final_dict["response"] = response
                if i < len(line_nos):
                    final_dict["line_no"] = line_nos[i]
                final_result.append(final_dict)
    return final_result


results = generate_dict_format(new_dict_copy, output_dict,pdf_document)
transpositional_boxes = return_json_result(results)
print("append_json_results",transpositional_boxes)

