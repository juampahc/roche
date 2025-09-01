from managers import bytes_io_manager
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from helpers import custom_page_postprocess, text_extraction, custom_preparation
from paddlex_hps_client import triton_request
from tritonclient import grpc as triton_grpc
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from configuration import get_settings
import base64
import logging
import pymupdf

# Load uvicorn logger
logger = logging.getLogger('uvicorn.error')

# ======================================================================
# LifeSpan for configs
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start by loading the settings
    settings = get_settings()
    # Create the tritonclient for gRPC server
    logger.info('Creatin Triton Client:')
    tritoncli = triton_grpc.InferenceServerClient(settings.TRITON_URL)

    logger.info(f'REST-API startup with given config: {settings.model_dump()}')
    app.state.settings = settings
    logger.info('TritonClient created')
    app.state.tritoncli = tritoncli
    yield
    
    # Delete global objects
    del app.state.settings
    del app.state.tritoncli
    logger.info("Configuration deleted.")

# Set up the FastAPI app and define the endpoints
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post('/v1/page-dla', summary="Performs DLA sequentially over a PDF document")
async def extract_page_text(file: UploadFile, 
                              tritoncli:triton_grpc.InferenceServerClient = Depends(lambda: app.state.tritoncli)):
    """
    Perform Document Layout Analysis sequentially (page by page) over the stream of a PDF file.
    
    :param file: PDF FILE
    :return: A JSON payload.
    """
    try:
        # Read the bytes
        pdf_content = file.file.read()
        document = []
        with bytes_io_manager(pdf_content) as pdf_stream:
            for page, pix, encoded_image in custom_preparation(pdf_stream):
                input_ = {"file": encoded_image}
                input_["fileType"] = 1 # IMG
                output = triton_request(tritoncli, "formula-recognition", input_)
                if output["errorCode"] != 0:
                    raise HTTPException(status_code=500, detail=f"ErrorCode: {output['errorCode']} - Error Msg: {output['errorMsg']}")
                results = [box for formula_result in output["result"]["formulaRecResults"] 
                           for box in formula_result["prunedResult"]['layout_det_res']['boxes'] ]
                processed_results = custom_page_postprocess(page, pix, results)
                results_with_text = text_extraction(page, processed_results)
                document.append({'page': page.number,
                                'content': results_with_text})
        return {'document':document}
    except Exception as e:
        # Handle Exceptions with a generic error
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post('/v1/doc-dla', summary="Performs DLA over a PDF document")
async def extract_pdf_text(file: UploadFile, 
                           tritoncli:triton_grpc.InferenceServerClient = Depends(lambda: app.state.tritoncli)):
    """
    Perform Document Layout Analysis over the stream of a PDF file.
    
    :param file: PDF FILE
    :return: A JSON payload.
    """
    try:
        # Read the bytes
        pdf_content = file.file.read()
        encoded_document = base64.b64encode(pdf_content).decode("ascii") # What we send to the model
        pdf_document = pymupdf.open(stream=pdf_content, filetype='pdf')  # Document / PyMuPDF
        
        # Set parameters for inference request
        input_ = {"file": encoded_document}
        input_["fileType"] = 0 #PDF
        
        # Send the whole document in one request
        output = triton_request(tritoncli, "formula-recognition", input_)
        if output["errorCode"] != 0:
            raise HTTPException(status_code=500, detail=f"ErrorCode: {output['errorCode']} - Error Msg: {output['errorMsg']}")
        
        # Good to check
        if pdf_document.page_count != output['result']['dataInfo']['numPages']:
            raise HTTPException(status_code=500, detail='Missmatch size between PDF reader and predictions for PDF object')
        
        # Proceed to post-process SEQUENTIALLY
        document = []
        for pdf_page, pdf_result  in zip([page for page in pdf_document], output['result']['formulaRecResults']):
            boxes = [box for box in pdf_result['prunedResult']['layout_det_res']['boxes']]
            img = pdf_result['outputImages']['layout_det_res']
            pixmap = pymupdf.Pixmap(base64.b64decode(img))
            processed_results = custom_page_postprocess(pdf_page, pixmap, boxes)
            results_with_text = text_extraction(pdf_page, processed_results)
            document.append({'page': pdf_page.number,
                            'content': results_with_text})
        return {'document': document}
    except Exception as e:
        # Handle Exceptions with a generic error
        raise HTTPException(status_code=500, detail=str(e)) from e