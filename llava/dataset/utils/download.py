import requests
from io import BytesIO
from PIL import Image

def load_remote_json(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            return json_data
        else:
            print("Failed to fetch data. Status code:", response.status_code)
            return None
    except Exception as e:
        print("json An error occurred:", e)
        raise e
    
def download_image(image_url):
    import logging
    # if 'kg-t2i-mark.bj.bcebos.com' in image_url:
    #     return Image.open('/mnt/pexels/' + image_url.split('http://kg-t2i-mark.bj.bcebos.com')[-1].split('?autho')[0])
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            logging.error(f"下载失败，HTTP响应状态码:{response.status_code}")
            raise Exception("下载失败")
    except:
        return Image.open('/mnt/pexels/' + image_url.split('http://kg-t2i-mark.bj.bcebos.com')[-1].split('?autho')[0])
    

def download_image_for_yewu(image_url):
    import logging
    # if 'kg-t2i-mark.bj.bcebos.com' in image_url:

    # response = requests.get(image_url)
    # Image.open(BytesIO(response.content))
    #     return Image.open('/mnt/pexels/' + image_url.split('http://kg-t2i-mark.bj.bcebos.com')[-1].split('?autho')[0])
    # try:

    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        logging.error(f"下载失败，HTTP响应状态码:{response.status_code}")
        raise Exception("下载失败")
    # except Exception as e:
    #     print(e)
    #     print("jj")
    #     return Image.open('/mnt/pexels/' + image_url.split('http://kg-t2i-mark.bj.bcebos.com')[-1].split('?autho')[0])

def download_image_for_yewu_human(image_url):
    import logging
    # if 'kg-t2i-mark.bj.bcebos.com' in image_url:

    # response = requests.get(image_url)
    # Image.open(BytesIO(response.content))
    #     return Image.open('/mnt/pexels/' + image_url.split('http://kg-t2i-mark.bj.bcebos.com')[-1].split('?autho')[0])
    # try:
    try:
        return Image.open(image_url)
    except:
        print("打开失败")