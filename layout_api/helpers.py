import pymupdf
import copy
import base64
from models import GeometricParser

def filter_overlap(predictions:list[dict]) -> list[dict]:
    """
    Given a set of predictions for bounding-boxes we need to check if 
    there are some overlap: it should not be, but depends on the inference
    server.
    """
    pending = predictions
    merged = True
    
    while merged:
        merged = False
        result = []
        
        while pending:
            base = pending.pop()
            i=0
            while i< len(pending):
                # TODO: what happens if labels are not the same?
                if base['coordinate'].intersects(pending[i]['coordinate']):
                    base['coordinate'] |= pending.pop(i)['coordinate']
                    merged = True
                else:
                    i += 1
            result.append(base)
        pending = result
    
    return pending

def sanitize_orphans(orphans=list[pymupdf.Rect], predictions=list[dict])-> tuple:
    """
    Given a set of regions known to be orphans we will check if they overlap
    with the predictions from the model.
    """
    orphan_result = copy.deepcopy(orphans)
    prediction_result = []
    
    # Iterate over predictions:
    for prediction in predictions:
        matches = [(i,r) for i,r in enumerate(orphan_result) if prediction['coordinate'].intersects(r)]
        # TODO: Are all the cases covered?
        if len(matches) == 0: # No overlap for this prediction
            prediction_result.append(prediction)
        elif len(matches) == 1: # Overlap with one orphan
            pred_= copy.deepcopy(prediction)
            match_index, match = matches[0]
            pred_['coordinate'] |= match
            orphan_result.pop(match_index) # Delete orphan from list
            prediction_result.append(pred_)
    return orphan_result, prediction_result

def custom_preparation(file_stream:bytes):
    """
    In previous versions I used to send the images sequentially to the
    inference server. That's why I had this function: to manage the 
    img extraction and further postprocessing.
    
    At this moment is not needed anymore. But I will keep it.
    """
    # Step 1 Open the PDF (although type is always inferred)
    doc = pymupdf.open(stream=file_stream, filetype='pdf')
    # Step 2 Iterates over the pages:
    for page in doc:
        # Step 3 Convert pdf page to image
        page_pix = page.get_pixmap()
        # Step 4 Convert PIX to bytes
        page_pix_bytes = page_pix.pil_tobytes(format="PNG")
        # Step 5 bases4 encoded bytes
        page_encoded = base64.b64encode(page_pix_bytes).decode("ascii")
        
        # Now we yield three elements that we will use across the processing:
        # (Page, Pix, Base64Image)
        yield page, page_pix, page_encoded

def custom_page_postprocess(page:pymupdf.Page, 
                            pix:pymupdf.Pixmap, 
                            pruned_results:list[dict])->list[pymupdf.Rect]:
    """
    Given a set of bounding boxes from images we will transform them 
    into pdf coordinates
    """
    # Step 1 We create a matrix using Pixmap in order to transform
    mat = pymupdf.IRect(pix.irect).torect(page.rect)
    result = []
    # Step 2 transform the BBOXES
    for pruned_result in pruned_results:
        prunned_ = copy.deepcopy(pruned_result)
        x0, y0, x1, y1 = pruned_result['coordinate']
        rect = pymupdf.Rect(x0, y0, x1, y1)
        pdf_coordinates = rect * mat
        # Update the copy so we can override only the coordinates
        prunned_['coordinate'] = pdf_coordinates
        result.append(prunned_)
    return result

def text_extraction(page:pymupdf.Page,
                    prunned_results:list[dict])->list[dict]:
    """
    Function to extract the text given a set of Rectangles containing
    the layout description of the pdf. The text is added to the 
    dictionary of results.

    """
    result = []
    orphans = []
    page_text = page.get_textpage()
    
    # Filter overlap in predictions
    prunned_results_ = filter_overlap(copy.deepcopy(prunned_results))
    
    # Look for orphan Rectangles
    parser = GeometricParser([box['coordinate'] for box in prunned_results_])
    for word in page.get_text('words', sort=True, textpage=page_text):
        word_rect = pymupdf.Rect(word[0], word[1], word[2], word[3])
        # Check if the word is not contained in predictions
        matches = [(i,r) for i,r in enumerate(prunned_results_) if r['coordinate'].intersects(word_rect)]
        if len(matches) == 0: # Orphan word
            parser.append(word_rect)
    
    # Generate orphan regions
    orphan_regions = []
    if not parser.is_empty():
        for region in parser.generate_regions():
            orphan_regions.append(region)
    
    # orphan regions should not overlap with each other
    orphan_regions = [r['coordinate'] for r in filter_overlap([{'coordinate':region} for region in orphan_regions])]
    
    # Now check that there are no overlap between orphans and results
    orphan_regions, prunned_results_ = sanitize_orphans(orphan_regions, prunned_results_)
    
    # Extract the text for orphans
    for region in orphan_regions:
        text = page.get_text("text", clip=region, sort=True, textpage=page_text)
        orphans.append({'cls_id':30, 
                        'label': 'unknown', 
                        'score': 100.0, 
                        'coordinate':region,
                        'text':text})

    # Check last time for overlap in predictions
    prunned_results_ = filter_overlap(copy.deepcopy(prunned_results_))
    
    # Extract the text for results
    for prunned_result in prunned_results_:
        prunned_ = copy.deepcopy(prunned_result)
        text = page.get_text("text", clip=prunned_result['coordinate'], sort=True)
        prunned_['text']= text
        result.append(prunned_)
    
    return orphans + result
