"""Shared, cached CPU inference for Streamlit; no laptop API is required."""
import io
import threading
import time
import warnings
from pathlib import Path
from zipfile import ZipFile
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
from engine import Models

ROOT=Path(__file__).resolve().parent

class Runtime:
    def __init__(self):
        torch.set_num_threads(2)
        self.models=Models(ROOT/'models',ROOT,ROOT/'refinement')
        self.lock=threading.Lock()
        with ZipFile(ROOT/'assets/gallery.zip') as archive:
            expected={uid+'.jpg' for uid in self.models.gallery_ids}
            if set(archive.namelist())!=expected:
                raise ValueError('The reference images do not match the model gallery.')

    def predict(self,content,mode):
        if mode not in {'classification','identification'}:
            raise ValueError('Choose classification or identification.')
        if not content:
            raise ValueError('Choose an image first.')
        if len(content)>8*1024*1024:
            raise ValueError('Use an image smaller than 8 MB.')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    if image.format not in {'JPEG','PNG','WEBP'}:
                        raise ValueError('Use a JPG, PNG or WebP image.')
                    if image.width*image.height>20_000_000:
                        raise ValueError('Use an image smaller than 20 megapixels.')
                    image.load()
                    with self.lock:
                        start=time.perf_counter()
                        result=self.models.predict(image,mode)
                        result['processing_ms']=round((time.perf_counter()-start)*1000,2)
                    return result
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
            raise ValueError('This image could not be read. Choose another JPG, PNG or WebP.') from error

    def reference_image(self,uid):
        if uid not in self.models.gallery_rows:
            raise ValueError('Reference image not found.')
        with ZipFile(ROOT/'assets/gallery.zip') as archive:
            return archive.read(uid+'.jpg')

@st.cache_resource(show_spinner=False)
def get_runtime():
    return Runtime()
