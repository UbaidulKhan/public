import requests # request img from web
import shutil # save img locally

ubuntu_url = 'https://releases.ubuntu.com/23.04/ubuntu-23.04-netboot-amd64.tar.gz'
# url = input('Please enter an image URL (string):') #prompt user for img url
url = ubuntu_url
file_name = input('Save image as (string):') #prompt user for file_name

res = requests.get(url, stream = True)

if res.status_code == 200:
    with open(file_name,'wb') as f:
        shutil.copyfileobj(res.raw, f)
    print('Image sucessfully Downloaded: ',file_name)
else:
    print('Image Couldn\'t be retrieved')