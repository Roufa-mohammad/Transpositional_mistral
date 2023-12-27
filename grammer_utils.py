import time

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import io
from transformers import pipeline
import sys
import fitz
#sys.path.append('/data/copy_editing_prod')


from copy_editing_tool.data_science.density_extraction.references.extract_reference_details import (
    extract_references_from_text,
) 
from copy_editing_tool.data_science.density_extraction.references.extract_superscripts import (
    is_super_Script_present,
)
from copy_editing_tool.data_science.density_extraction.TOC.extract_TOC import (
    TOC_num,
)                                

from copy_editing_tool.data_science.density_extraction.TOC.utils import (
    detr_table_boxes,
    img_cords_to_page_cords
)

from copy_editing_tool.data_science.density_extraction.grammar.merge_bboxes import (
    merge_bboxes,
)



#if any superscript/footnotes in pdf then remove them as well
def clean_line(line, results):
    for res in results:
        if res[0] in line['text']:
            line['text'] = line['text'].replace(res[0], '')
    return line['text']


def remove_references_from_doc(pdf, reference_details, super_script_results):
    '''
    This function removes the references from the text and returns other text details.
    '''
    start_page = 1
    start_idx = 1
    clean_lines_per_page = {}

    if not reference_details:  # If there are no reference details, process all pages
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = page.extract_text_lines()
            cleaned_lines = []

            for line in lines:
                cleaned_line = clean_line(line, super_script_results)
                cleaned_lines.append(cleaned_line)

            clean_lines_per_page[page_num] = {'text': cleaned_lines}

    else:
        for ref_detail in reference_details:
            end_page = ref_detail['start_page_num']
            end_idx = ref_detail['start_idx']


            for page_num in range(start_page, end_page + 1):
                page = pdf.pages[page_num - 1]
                lines = page.extract_text_lines()

                if page_num == start_page:
                    lines = lines[start_idx - 1:]

                if page_num == end_page:
                    lines = lines[:end_idx]
                
                cleaned_lines = []

                for line in lines:
                    cleaned_line = clean_line(line, super_script_results)
                    cleaned_lines.append(cleaned_line)
                clean_lines_per_page[page_num] = {'text': cleaned_lines}


        start_page = ref_detail['end_page_num']
        start_idx = ref_detail['end_idx']

    return clean_lines_per_page



def remove_tables_data(pdf_path,pdf,start_num,text_ref_dictionary, detr_processor, detr_model):
    '''
    This fucntions find page has border or borderless table and removing the table,
    this code takes each line pass though number-treatment code and return final_transposition_context  which is having pageno,line_no,and there corresponding text  
    '''
    sample_dict = {}
    final_transposition_context = {}
    output_dict = {}

    for page_no, page in enumerate(pdf.pages):
        
        if page_no > start_num:
        #if page_no >= 50 and page_no <= 52:
            print("start_num-->",start_num)
            if page.find_tables():
                images = convert_from_path(pdf_path, first_page=page_no+1 , last_page=page_no+1 )
                for j, img in enumerate(images):
                    # Save the image as JPG and pass it to detr_table_boxes
                    img_as_bytes = io.BytesIO()
                    # Convert the current page to images

                    img.save(img_as_bytes, format="JPEG")
                    img_as_bytes.seek(0)
                    
                    # Detect and get bounding boxes
                    bboxes = detr_table_boxes(img_as_bytes, detr_processor, detr_model)
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
        
        else:
            if page_no > 15:
                print("start_num-->",start_num)
                if page.find_tables():
                    images = convert_from_path(pdf_path, first_page=page_no+1 , last_page=page_no+1 )
                    for j, img in enumerate(images):
                        # Save the image as JPG and pass it to detr_table_boxes
                        img_as_bytes = io.BytesIO()
                        # Convert the current page to images

                        img.save(img_as_bytes, format="JPEG")
                        img_as_bytes.seek(0)
                        bboxes = detr_table_boxes(img_as_bytes, detr_processor, detr_model)
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


def match_line_in_sentece(output_dict):
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





def Generate_response(prompt, new_dict, model_2, tokenizer_2):
    ''' This fucntion is to Generate response for the sentence and add this responses to the yes_no_Dictionary '''
    start = time.time()
    
    yes_no_dictionary = {}
    for pno in new_dict:

        #print("pg-", pno)
        
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
        #print("response:", response)

        split_respose = response.split('~')[-1]
        
        list_of_sentence = []
        for each_sentence in split_respose.split('\n'):
            
            try:
                recover_sentece = each_sentence.split(":")[1]
                list_of_sentence.append(recover_sentece.split()[0])
            except:
                list_of_sentence.append('No')
        yes_no_dictionary[pno]=list_of_sentence
    end = time.time()

    return yes_no_dictionary

def get_processed_data(path, detr_processor, detr_model):
    start = time.time()
    pdf = pdfplumber.open(path)
    super_script_results =  is_super_Script_present(pdf)
    print("ss:", super_script_results)
    reference_details = extract_references_from_text(pdf)
    print("ref:", reference_details)
    text_ref_dictionary = remove_references_from_doc(pdf, reference_details, super_script_results)
    # print("text_ref:", text_ref_dictionary)
    _, page_num = TOC_num(pdf) #get TOC starting page
    # print("page_num---->")
    processed_dict = remove_tables_data(path, pdf, page_num,text_ref_dictionary, detr_processor, detr_model)
    # print("processed_dict---->",processed_dict)
    final_cleaned_dict = match_line_in_sentece(processed_dict)
    end = time.time()
    return final_cleaned_dict


def process_ouput(new_dict_copy):
    filtered_dict = {}
    '''Below code take only that dictinary which is having response yes '''
    for key, value in new_dict_copy.items():
        if any(('Yes' in str(v) or 'YES' in str(v) or 'Yes,' in str(v) or 'YES,' in str(v)) for v in value.values()):
            filtered_dict[key] = {k: v for k, v in value.items() if ('Yes' in str(v) or 'YES' in str(v) or 'Yes,' in str(v) or 'YES,' in str(v))}
    return filtered_dict

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
    
    return new_dict_copy, pdf_document

def generate_dict_format(new_dict_copy, pdf_document):
    final_result = []
    for pgno in new_dict_copy:
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
    return len(final_result), final_result
