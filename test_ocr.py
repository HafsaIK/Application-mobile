import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    img = Image.new('RGB', (300, 100), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Default font might not be available or might be too small, but let's try
    try:
        # Try to load a font, or use default
        font = ImageFont.load_default()
    except:
        font = None
    
    d.text((10,10), "Hello World", fill=(0,0,0), font=font)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_ocr():
    url = "http://127.0.0.1:8000/ocr"
    img_bytes = create_test_image()
    
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ocr()
