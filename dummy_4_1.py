import re
import pdfplumber  
from word2number import w2n
import pdfplumber
from fuzzywuzzy import fuzz,process
import re
import subprocess
# import fitz
import cv2
import numpy as np

from PIL import Image
import io
import matplotlib.pyplot as plt





from PyPDF2 import PdfReader
import unidecode
from typing import List
import nltk
import pdfplumber

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from pdf2image import convert_from_path
from PIL import Image, ImageDraw 




#based on top/bottom coordinates of a character, in a line
#if they are different to each other then considering that as superscript. code to check for 1,2.. 10,11..


#based on top/bottom coordinates of a character, in a line
#if they are different to each other then considering that as superscript. code to check for 1,2.. 10,11..
def are_chars_different(chars):
    if len(chars) < 1:
        return []

    first_char = chars[0]
    first_char_text = first_char['text']
    first_char_top = first_char['top']
    first_char_bottom = first_char['bottom']

    if len(chars) >= 2:
        second_char = chars[1]
        second_char_text = second_char['text']

        if second_char_text.strip() and \
                all(is_char_different(char, first_char) for char in chars[2:]) and \
                all(is_char_different(char, second_char) for char in chars[2:]) and \
                first_char_text.isdigit() and second_char_text.isdigit(): 
            return [(first_char_text + second_char_text)]
        elif any(is_char_different(char, first_char) for char in chars[1:]) and \
                first_char_text.isdigit():
            return [(first_char_text)]
        else:
            return []
    else:
        return [(first_char_text)]

def is_char_different(char1, char2):
    return char1['top'] != char2['top'] or char1['bottom'] != char2['bottom']

#identify any superscripts in the next few pages after references.
def super_Script_present(lines, next_page_num):
    results = []
    for idx, line in enumerate(lines):
        first_five_chars = line['chars'][:5] #considering only 1st 5 chars in line to check if a superscript is present
        different_chars = are_chars_different(first_five_chars)
        starts_with_ordered_number = re.match(r'^\d{1,3}(st|nd|rd|th)[, ]', line['text'])  # Check if line starts with 1st, 2nd, ..., 10th, etc.
        if different_chars and len(line['text']) > 5 and not starts_with_ordered_number:
            results.append((different_chars[0], next_page_num, idx))
    return results

#check if a bold font is present in line
def is_bold_font(line):
    return 'Bold' in line["chars"][0]["fontname"]

#extract all the found search term indexes from the pdf
def extract_references_text_indexes(pdf):
    references_indexes = []
    search_terms = ["references", "bibliography", "works cited", "citations", "reference list"]
    loop_break_terms  = ["abstract","abstract:"]

    for page in pdf.pages:
        lines = page.extract_text_lines()

        for idx, line in enumerate(lines):            
            for term in search_terms:
                if line["text"].strip().lower() == term.strip():
                    references_indexes.append({'start_page_num': page.page_number, 'start_idx': idx, 'keyword': term})
                    break  # Stop searching for other keywords on this line
   
    for i in range(len(references_indexes)):
        page_num = references_indexes[i]['start_page_num']
        start_idx = references_indexes[i]['start_idx']
        
        #once the search keyword found then extract references from that index in the current page.
        current_page_lines = pdf.pages[page_num - 1].extract_text_lines()
        end_page_num = None
        for idx in range(start_idx + 1, len(current_page_lines)):
            if is_bold_font(current_page_lines[idx]): #break if bold font is found
                end_idx = idx
                end_page_num = page_num
                break

        # Search for bold font or superscript or any loop_break_terms in the subsequent pages
        #to stop extracting the references.
        next_page_num = page_num + 1
        while end_page_num is None and next_page_num <= len(pdf.pages):
            next_page_lines = pdf.pages[next_page_num - 1].extract_text_lines()

            for idx in range(len(next_page_lines)):
                results = super_Script_present(next_page_lines, next_page_num)
                if results:
                    for result in results:
                        sc_page_num = result[1]
                        sc_page_idx = result[2]
                        end_idx = sc_page_idx 
                        end_page_num = sc_page_num
                        break
                elif is_bold_font(next_page_lines[idx]):
                    end_idx = idx
                    end_page_num = next_page_num
                    break
                else:
                    found_match = False
                    for term in loop_break_terms:
                        for line_idx, line in enumerate(next_page_lines):
                            #if line['text'].lower().startswith(term):
                            if line['text'].lower() == term:
                                end_idx = line_idx
                                end_page_num = next_page_num
                                found_match = True
                                break
                                
                        if found_match:
                            break

            next_page_num += 1

        if end_page_num is None:
            # If no bold font or horizontal edges found in subsequent pages, use the last page
            references_indexes[i]['end_page_num'] = len(pdf.pages)
            references_indexes[i]['end_idx'] = len(pdf.pages[-1].extract_text_lines()) - 1
        else:
            references_indexes[i]['end_page_num'] = end_page_num
            references_indexes[i]['end_idx'] = end_idx

    return references_indexes

#if any reference keyword present at start of page after the references are found then ignore those indexes.
def filter_references(references_indexes):
    # Sort the references based on 'start_page_num'
    sorted_references = sorted(references_indexes, key=lambda x: x['start_page_num'])

    final_reference_indexes = []
    last_end_page_num = -1

    for ref in sorted_references:
        if ref['start_page_num'] > last_end_page_num:
            # Add the reference to the final list if it starts after the last identified reference
            final_reference_indexes.append(ref)
            last_end_page_num = ref['end_page_num']

    return final_reference_indexes


#extract all the reference details based on start and end page numbers
def extract_references_text_from_details(pdf, details):
    references_text = []

    for detail in details:
        start_page_num = detail['start_page_num']
        start_idx = detail['start_idx']
        end_page_num = detail['end_page_num']
        end_idx = detail['end_idx']

        # Extract text from the start page
        start_page = pdf.pages[start_page_num - 1]
        lines = start_page.extract_text_lines()
        extracted_references = []
        for idx in range(start_idx + 1, len(lines)):
            extracted_references.append(lines[idx]["text"].strip())

        # Extract text from subsequent pages until end_page_num or end_idx is reached
        current_page_num = start_page_num + 1
        while current_page_num <= end_page_num:
            current_page = pdf.pages[current_page_num - 1]
            lines = current_page.extract_text_lines()
            for idx in range(len(lines)):
                if current_page_num == end_page_num and idx >= end_idx:
                    break
                extracted_references.append(lines[idx]["text"].strip())
            current_page_num += 1

        references_text.append({'start_page_num': start_page_num,
                                'start_idx': start_idx,
                                'end_page_num': end_page_num,
                                'end_idx': end_idx,
                                'references': extracted_references})

    return references_text

#is superscript present in the first 5 chars in each line
def is_super_Script_present(pdf):
    results = []
    for page in pdf.pages:
        lines = page.extract_text_lines()
        for idx, line in enumerate(lines):
            first_five_chars = line['chars'][:5]
            different_chars = are_chars_different(first_five_chars)
            starts_with_ordered_number = re.match(r'^\d{1,3}(st|nd|rd|th)[, ]', line['text'])  # Check if line starts with 1st, 2nd, ..., 10th, etc.
            if different_chars and len(line['text']) > 5 and not starts_with_ordered_number:
                results.append((line['text'], page.page_number, idx))
    return results


def clean_line(line, results):
    for res in results:
        if res[0] in line['text']:
            line['text'] = line['text'].replace(res[0], '')
    return line["text"]


def remove_references_from_doc(pdf, reference_details, super_script_results):
    '''
    This function remove the reference the text from the page and give remaining text
    '''
    start_page = 1
    start_idx = 1
    clean_lines_per_page = {}

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
                # print(cleaned_line)
            clean_lines_per_page[page_num] = {'text': cleaned_lines}


        start_page = ref_detail['end_page_num']
        start_idx = ref_detail['end_idx']

    return clean_lines_per_page
    




















